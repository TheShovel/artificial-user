import { parentPort } from 'node:worker_threads';
import { pipeline } from '@huggingface/transformers';
import * as ort from 'onnxruntime-node';
import { config } from './config.js';
import { ensureVadModel } from './vad.js';
import { isRepetitiveGarbage, mentionsBotName } from './text.js';

// Runs Whisper inference off the main thread so the bot never freezes while
// transcribing. Jobs are processed one at a time (serialized), which also
// keeps many-people-talking-at-once from thrashing the CPU.
//
// Before Whisper, a Silero VAD pass rejects non-speech (music, game audio,
// keyboard noise) in milliseconds — Whisper is only run on actual speech.
//
// Speed design: the main model is whisper-base (~2.5x faster than small).
// Whisper-base occasionally muffles the wake word, so when the fast pass
// finds no wake word, a whisper-tiny.en pass double-checks — tiny.en is fast
// AND hears "bobby" in cases base misses. Worst case ~1s, common case ~0.7s.

function pcmToFloat32(buffer) {
  const samples = new Float32Array(buffer.length / 2);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = buffer.readInt16LE(i * 2) / 32768;
  }
  return samples;
}

const transcriberCache = new Map(); // model id -> promise
function getTranscriber(modelId, dtype) {
  if (!transcriberCache.has(modelId)) {
    console.log(`[stt-worker] loading "${modelId}" (dtype: ${dtype})...`);
    transcriberCache.set(
      modelId,
      pipeline('automatic-speech-recognition', modelId, { dtype }),
    );
  }
  return transcriberCache.get(modelId);
}

async function transcribeWith(modelId, transcriber, buffer) {
  const audio = pcmToFloat32(buffer);
  const options = { chunk_length_s: 30, stride_length_s: 5 };
  // English-only models (.en) reject a language hint.
  if (config.sttLanguage && !modelId.toLowerCase().includes('.en')) {
    options.language = config.sttLanguage;
  }
  const output = await transcriber(audio, options);
  return (output?.text ?? '').trim();
}

let vadSessionPromise = null;
function getVad() {
  if (!vadSessionPromise) {
    vadSessionPromise = ensureVadModel().then((modelPath) =>
      ort.InferenceSession.create(modelPath, {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all',
      }),
    );
  }
  return vadSessionPromise;
}

async function handleTranscribe(message) {
  try {
    const buffer = Buffer.from(message.buffer);

    // Stage 1: Silero VAD — reject non-speech without touching Whisper.
    const speech = await vadSpeech(buffer);
    if (!speech.isSpeech) {
      parentPort.postMessage({ id: message.id, text: '', wake: false });
      return;
    }

    if (message.model === 'fast') {
      // Detection pass: whisper-tiny.en (~0.3 s) is fast AND hears the wake
      // word more reliably than bigger models. Returns whether it was hit.
      const fast = await getTranscriber(config.sttFallbackModel, config.sttDtype);
      let text = await transcribeWith(config.sttFallbackModel, fast, buffer);
      // Whisper sometimes locks onto a syllable ("I-I-I-I-I..."); that's not
      // speech worth acting on.
      if (isRepetitiveGarbage(text)) text = '';
      parentPort.postMessage({ id: message.id, text, wake: mentionsBotName(text) });
    } else {
      // Full-quality pass (whisper-base) for actual responses.
      const main = await getTranscriber(config.sttModel, config.sttDtype);
      let text = await transcribeWith(config.sttModel, main, buffer);
      if (isRepetitiveGarbage(text)) text = '';
      parentPort.postMessage({ id: message.id, text });
    }
  } catch (error) {
    parentPort.postMessage({ id: message.id, error: error.message, wake: false });
  }
}

/**
 * Silero VAD over 16 kHz mono 16-bit PCM. A clip is speech when it has:
 *  - at least `vadMinSpeechFrames` frames at >= `vadProbThreshold`, making up
 *    at least `vadMinSpeechFraction` of the clip, AND
 *  - at least `vadMinConfidentFrames` frames at >= `vadConfidenceThreshold`.
 *
 * The confidence floor is the music discriminator: real speech (even quiet or
 * short — "yeah", "okay") hits 0.9+ on several frames, while music/tones
 * rarely ever do, so it rejects music without rejecting real speech.
 */
export async function vadSpeech(pcm16kMono) {
  const session = await getVad();
  const CHUNK = 512; // 32 ms at 16 kHz
  let state = new Float32Array(2 * 1 * 128);
  let total = 0;
  let strong = 0;
  let confident = 0;
  let maxProb = 0;

  for (let off = 0; off + CHUNK * 2 <= pcm16kMono.length; off += CHUNK * 2) {
    const input = new Float32Array(CHUNK);
    for (let i = 0; i < CHUNK; i++) input[i] = pcm16kMono.readInt16LE(off + i * 2) / 32768;
    const feeds = {
      input: new ort.Tensor('float32', input, [1, CHUNK]),
      state: new ort.Tensor('float32', state, [2, 1, 128]),
      sr: new ort.Tensor('int64', BigInt64Array.from([16000n]), []),
    };
    const out = await session.run(feeds);
    const prob = out.output.data[0];
    maxProb = Math.max(maxProb, prob);
    if (prob >= config.vadProbThreshold) strong++;
    if (prob >= config.vadConfidenceThreshold) confident++;
    total++;
    state = Float32Array.from(out.stateN.data);
  }

  const fraction = total > 0 ? strong / total : 0;
  return {
    isSpeech:
      strong >= config.vadMinSpeechFrames &&
      fraction >= config.vadMinSpeechFraction &&
      confident >= config.vadMinConfidentFrames,
    frames: total,
    strong,
    confident,
    maxProb,
  };
}

// Serialize jobs so only one inference runs at a time.
let chain = Promise.resolve();
parentPort.on('message', (message) => {
  if (message.type !== 'transcribe') return;
  chain = chain.then(() => handleTranscribe(message));
});

// Load the VAD + model at startup so the first utterance doesn't wait.
getVad()
  .then(() => {
    parentPort.postMessage({ type: 'ready' });
    console.log('[stt-worker] model + VAD ready');
  })
  .catch((error) => {
    console.error('[stt-worker] startup failed:', error.message);
    parentPort.postMessage({ type: 'ready' }); // still come up; VAD will be retried per job
  });
