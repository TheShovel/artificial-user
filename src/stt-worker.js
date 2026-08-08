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
// This worker handles ONE role, chosen by the parent process:
//   "fast" -> whisper-tiny.en, the wake-word detector (every utterance)
//   "main" -> whisper-base, the quality pass (only for actual replies)
// Splitting the models across dedicated pools means a slow quality pass on
// one server never delays wake-word detection on another, and each worker
// only loads the one model it needs.
//
// Before Whisper, a Silero VAD pass rejects non-speech (music, game audio,
// keyboard noise) in milliseconds — Whisper is only run on actual speech.

const ROLE = process.env.STT_WORKER_ROLE ?? 'main';
const MODEL_ID = ROLE === 'fast' ? config.sttFallbackModel : config.sttModel;

function pcmToFloat32(buffer) {
  const samples = new Float32Array(buffer.length / 2);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = buffer.readInt16LE(i * 2) / 32768;
  }
  return samples;
}

let transcriberPromise = null;
function getTranscriber() {
  if (!transcriberPromise) {
    console.log(`[stt-worker:${ROLE}] loading "${MODEL_ID}" (dtype: ${config.sttDtype})...`);
    transcriberPromise = pipeline('automatic-speech-recognition', MODEL_ID, {
      dtype: config.sttDtype,
    }).catch((error) => {
      // Allow a later job to retry (e.g. after a download failure).
      transcriberPromise = null;
      throw error;
    });
  }
  return transcriberPromise;
}

async function transcribeWith(transcriber, buffer) {
  const audio = pcmToFloat32(buffer);
  const options = { chunk_length_s: 30, stride_length_s: 5 };
  // English-only models (.en) reject a language hint.
  if (config.sttLanguage && !MODEL_ID.toLowerCase().includes('.en')) {
    options.language = config.sttLanguage;
  }
  const output = await transcriber(audio, options);
  return (output?.text ?? '').trim();
}

let vadSessionPromise = null;
function getVad() {
  if (!vadSessionPromise) {
    vadSessionPromise = ensureVadModel()
      .then((modelPath) =>
        ort.InferenceSession.create(modelPath, {
          executionProviders: ['cpu'],
          graphOptimizationLevel: 'all',
        }),
      )
      .catch((error) => {
        // Allow a later job to retry (e.g. after a download failure).
        vadSessionPromise = null;
        throw error;
      });
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

    const transcriber = await getTranscriber();
    let text = await transcribeWith(transcriber, buffer);
    // Whisper sometimes locks onto a syllable ("I-I-I-I-I..."); that's not
    // speech worth acting on.
    if (isRepetitiveGarbage(text)) text = '';

    if (ROLE === 'fast') {
      // The detection pass also reports whether the bot's name was said.
      parentPort.postMessage({ id: message.id, text, wake: mentionsBotName(text) });
    } else {
      // Full-quality pass (whisper-base) for actual responses.
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

// Load the VAD + this worker's model at startup so the first utterance
// doesn't wait for a download or a cold model load.
getVad()
  .then(() => getTranscriber())
  .then(() => {
    parentPort.postMessage({ type: 'ready' });
    console.log(`[stt-worker:${ROLE}] model + VAD ready`);
  })
  .catch((error) => {
    console.error(`[stt-worker:${ROLE}] startup failed:`, error.message);
    parentPort.postMessage({ type: 'ready' }); // still come up; models retried per job
  });
