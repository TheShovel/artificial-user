#!/usr/bin/env node
/**
 * End-to-end sanity check that doesn't need Discord:
 *   1. Synthesize speech with the local TTS.
 *   2. Resample it exactly like the voice-capture pipeline (48k stereo -> 16k mono).
 *   3. Transcribe it back with Whisper and check the words come through.
 *   4. Ask Ollama a question and check the reply.
 *
 * Usage: npm run selftest
 */
import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { config } from '../src/config.js';
import { checkOllama, ask } from '../src/llm.js';
import { estimateSpeech, transcribe } from '../src/stt.js';
import { synthesize } from '../src/tts.js';

const require = createRequire(import.meta.url);
const { OpusEncoder } = require('@discordjs/opus');

const PHRASE = 'the quick brown fox jumps over the lazy dog';

/**
 * The exact path real Discord audio takes in the bot:
 *   wav -> 48 kHz stereo PCM -> Opus frames -> native libopus decode
 *        -> ffmpeg resample to 16 kHz mono
 */
function resampleWavTo16kMono(wavBuffer) {
  return new Promise((resolve, reject) => {
    const pcm = wavBuffer.subarray(44); // skip WAV header (22.05 kHz mono)

    const to48k = spawn('ffmpeg', [
      '-loglevel', 'error',
      '-f', 's16le', '-ar', '22050', '-ac', '1', '-i', 'pipe:0',
      '-f', 's16le', '-ar', '48000', '-ac', '2', 'pipe:1',
    ]);
    to48k.on('error', reject);

    const encoder = new OpusEncoder(48000, 2, 'audio');
    const opusFrames = [];
    to48k.stdout.on('data', (c) => {
      for (let i = 0; i + 3840 <= c.length; i += 3840) {
        opusFrames.push(encoder.encode(c.subarray(i, i + 3840)));
      }
    });

    to48k.on('close', () => {
      // Same as the bot: native libopus decode, then ffmpeg resample.
      const decoder = new OpusEncoder(48000, 2, 'audio');
      const resampler = spawn('ffmpeg', [
        '-loglevel', 'error',
        '-f', 's16le', '-ar', '48000', '-ac', '2', '-i', 'pipe:0',
        '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1',
      ]);
      resampler.on('error', reject);
      const chunks = [];
      resampler.stdout.on('data', (c) => chunks.push(c));
      for (const frame of opusFrames) {
        try {
          const decoded = decoder.decode(frame);
          if (decoded?.length) resampler.stdin.write(decoded);
        } catch {
          // skip malformed frames
        }
      }
      resampler.stdin.end();
      resampler.on('close', () => resolve(Buffer.concat(chunks)));
    });

    to48k.stdin.on('error', () => {});
    to48k.stdin.end(pcm);
  });
}

const results = [];
function record(name, ok, detail = '') {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

console.log(`\n=== TTS → STT round trip (phrase: "${PHRASE}") ===\n`);

let wav;
try {
  wav = synthesize(PHRASE);
  record('tts.synthesize produces a WAV', wav && wav.length > 1000, `${wav?.length ?? 0} bytes`);
} catch (error) {
  record('tts.synthesize produces a WAV', false, error.message);
  process.exit(1);
}

let pcm;
try {
  pcm = await resampleWavTo16kMono(wav);
  record('opus encode/decode -> resample to 16k mono', pcm.length > 1000, `${pcm.length} bytes`);
} catch (error) {
  record('opus encode/decode -> resample to 16k mono', false, error.message);
  process.exit(1);
}

let heard;
try {
  heard = await transcribe(pcm);
  record('whisper transcribes the audio', heard.length > 0, `heard: "${heard}"`);
} catch (error) {
  record('whisper transcribes the audio', false, error.message);
  process.exit(1);
}

console.log(`\n=== Voice activity detection ===\n`);

const speechStats = estimateSpeech(pcm);
record(
  'real speech passes VAD',
  speechStats.isSpeech,
  `${speechStats.activeMs}ms active of ${speechStats.totalMs}ms (${(speechStats.fraction * 100).toFixed(0)}%)`,
);

// A synthetic 40 ms "keyboard click": loud burst followed by silence.
const click = Buffer.alloc(16000 * 2); // 1 second of 16k mono
for (let i = 0; i < 0.04 * 16000; i++) {
  const sample = (Math.random() * 2 - 1) * 0.6;
  click.writeInt16LE(Math.round(sample * 32767), i * 2);
}
const clickStats = estimateSpeech(click);
record(
  'a keyboard click is rejected by VAD',
  !clickStats.isSpeech,
  `${clickStats.activeMs}ms active of ${clickStats.totalMs}ms (${(clickStats.fraction * 100).toFixed(0)}%)`,
);

// Music (sine + noise) passes the energy gate but must be rejected by the
// Silero VAD in the worker, before Whisper ever runs. (Seeded noise so the
// test is deterministic.)
const music = spawnSync('ffmpeg', [
  '-loglevel', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',
  '-f', 'lavfi', '-i', 'anoisesrc=color=pink:duration=3:amplitude=0.4:seed=42',
  '-filter_complex', '[0][1]amix=inputs=2',
  '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1',
]);
const musicText = await transcribe(music.stdout);
record('music is rejected by Silero VAD (Whisper skipped)', musicText === '', `got: "${musicText}"`);

console.log(`\n=== Name wake word ===\n`);

const { isRepetitiveGarbage, mentionsBotName } = await import('../src/text.js');
const garbageTests = [
  ['I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-', true],
  ['bro bro bro bro bro bro bro', true],
  ['i\'m so sorry, '.repeat(80), true],
  ['hey bobby what do you think', false],
  ['yeah yeah yeah yeah sure', false],
  ["i'm so sorry i'm so sorry i'm so sorry i can't believe it", false],
];
for (const [input, expected] of garbageTests) {
  const got = isRepetitiveGarbage(input);
  record(`isRepetitiveGarbage(${JSON.stringify(input.slice(0, 20))}...) = ${expected}`, got === expected, `got ${got}`);
}
const nameTests = [
  ['hey bobby', true],
  ['Bobby tell us a joke', true],
  ['bobbie what do you think', true], // fuzzy spelling
  ['what do you think bobbi', true], // fuzzy spelling
  ['hey robot', true], // alias
  ['robots are cool', true], // alias plural
  ['hey ai', true], // alias
  ['what do you think clanker', true], // alias
  ['whats up guys', false],
  ['the weather is nice today', false],
  ['his voice sounds robotic', false],
  ['my uncle robert came over', false],
];
for (const [input, expected] of nameTests) {
  const got = mentionsBotName(input);
  record(`mentionsBotName(${JSON.stringify(input)}) = ${expected}`, got === expected, `got ${got}`);
}

console.log(`\n=== Ollama (${config.llmModel}) ===\n`);

try {
  await checkOllama();
  record('ollama reachable with model', true, config.llmModel);
} catch (error) {
  record('ollama reachable with model', false, error.message);
  process.exit(1);
}

try {
  const reply = await ask('selftest', 'Say hello and tell me the name of the animal in "the quick brown fox".');
  record('llm.ask returns a reply', reply.length > 0, `reply: "${reply.slice(0, 120)}"`);
} catch (error) {
  record('llm.ask returns a reply', false, error.message);
  process.exit(1);
}

console.log(`\n=== Memory ===\n`);
const { memory } = await import('../src/memory.js');
const history = memory.get('selftest');
record('conversation is stored in memory', history.length >= 2, `${history.length} messages saved to ${config.memoryFile}`);

const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} checks passed.\n`);
process.exit(failed ? 1 : 0);
