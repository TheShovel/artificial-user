#!/usr/bin/env node
/**
 * Quick validation of Silero VAD on real speech vs. music/noise.
 * Usage: node scripts/vad-test.mjs
 */
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import ort from 'onnxruntime-node';
import { synthesize } from '../src/tts.js';

async function vadProbs(pcm16kMono) {
  const session = await ort.InferenceSession.create('./models/silero-vad.onnx', {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
  });
  const CHUNK = 512;
  let state = new Float32Array(2 * 1 * 128);
  const probs = [];
  for (let off = 0; off + CHUNK * 2 <= pcm16kMono.length; off += CHUNK * 2) {
    const input = new Float32Array(CHUNK);
    for (let i = 0; i < CHUNK; i++) input[i] = pcm16kMono.readInt16LE(off + i * 2) / 32768;
    const feeds = {
      input: new ort.Tensor('float32', input, [1, CHUNK]),
      state: new ort.Tensor('float32', state, [2, 1, 128]),
      sr: new ort.Tensor('int64', BigInt64Array.from([16000n]), []),
    };
    const out = await session.run(feeds);
    probs.push(out.output.data[0]);
    state = Float32Array.from(out.stateN.data);
  }
  return probs;
}

function summarize(name, probs) {
  const over = probs.filter((p) => p >= 0.5).length;
  const max = probs.length ? Math.max(...probs) : 0;
  const mean = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0;
  console.log(
    `${name.padEnd(16)} frames=${String(probs.length).padStart(4)} over0.5=${String(over).padStart(4)} max=${max.toFixed(2)} mean=${mean.toFixed(2)}`,
  );
  return { over, max };
}

// 1) Real speech via mespeak -> 16k mono PCM
const wav = synthesize('hey bobby what do you think about this');
const r = spawnSync('ffmpeg', [
  '-loglevel', 'error', '-i', 'pipe:0',
  '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1',
], { input: wav });
const speech = r.stdout;

// 2) "Music": sustained sine + noise (no speech)
const music = spawnSync('ffmpeg', [
  '-loglevel', 'error', '-f', 'lavfi',
  '-i', "sine=frequency=440:duration=3",
  '-f', 'lavfi', '-i', "anoisesrc=color=pink:duration=3:amplitude=0.4",
  '-filter_complex', '[0][1]amix=inputs=2',
  '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1',
]);
const musicPcm = music.stdout;

// 3) Keyboard-click-ish: short bursts of noise
const click = Buffer.alloc(16000 * 2); // 1s
for (let i = 0; i < 0.04 * 16000; i++) {
  click.writeInt16LE(Math.round((Math.random() * 2 - 1) * 0.6 * 32767), i * 2);
}

// 4) Game audio-ish: noise bursts (staccato, non-speech)
const staccato = Buffer.alloc(16000 * 2); // 1s
for (let i = 0; i < 16000; i += 1600) { // 100ms bursts
  for (let j = 0; j < 800; j++) {
    staccato.writeInt16LE(Math.round((Math.random() * 2 - 1) * 0.5 * 32767), (i + j) * 2);
  }
}

// 5) Video-game SFX: square-wave beeps
const beeps = spawnSync('ffmpeg', [
  '-loglevel', 'error', '-f', 'lavfi',
  '-i', "sine=frequency=880:duration=0.15",
  '-f', 'lavfi', '-i', "sine=frequency=660:duration=0.15",
  '-f', 'lavfi', '-i', "sine=frequency=440:duration=0.15",
  '-filter_complex', '[0][1][2]concat=n=3:v=0:a=1',
  '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1',
]);
const beepsPcm = beeps.stdout;

for (const [name, pcm] of [
  ['speech', speech],
  ['music', musicPcm],
  ['click', click],
  ['noise bursts', staccato],
  ['game beeps', beepsPcm],
]) {
  const probs = await vadProbs(pcm);
  summarize(name, probs);
  const over5 = probs.filter((p) => p >= 0.5).length;
  const over7 = probs.filter((p) => p >= 0.7).length;
  const over9 = probs.filter((p) => p >= 0.9).length;
  console.log(
    `  over0.5=${over5} over0.7=${over7} over0.9=${over9} | ` +
      `rule(>=12 frames >=0.7): ${over7 >= 12 ? 'SPEECH' : 'NOT SPEECH'}`,
  );
}
process.exit(0);
