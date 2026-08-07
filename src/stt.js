import { Worker } from 'node:worker_threads';
import { config } from './config.js';

/**
 * Whisper transcription, run in a pool of worker threads so inference never
 * blocks the bot's event loop (it would otherwise freeze for ~3 s per
 * utterance, stutter playback, and pile up when several people talk at once).
 * Each worker serializes its own jobs; the pool lets multiple speakers
 * transcribe in parallel on multi-core machines.
 */

let workers = [];
let readyPromise = null;
let nextId = 1;
let dispatch = 0; // round-robin across the pool
const pending = new Map(); // job id -> { resolve, reject }

/** Spawn the STT worker pool and wait for the models to load. Cached. */
export function initStt() {
  if (workers.length > 0) return readyPromise;

  const poolSize = Math.max(1, config.sttWorkers);
  readyPromise = new Promise((resolve, reject) => {
    let ready = 0;
    for (let i = 0; i < poolSize; i++) {
      // Bound ONNX threads per worker: onnxruntime's default (all cores) thrashes
      // the cache and runs *slower*; poolSize x sttThreads uses the machine well.
      const worker = new Worker(new URL('./stt-worker.js', import.meta.url), {
        execArgv: [],
        env: { ...process.env, OMP_NUM_THREADS: String(config.sttThreads) },
      });

      worker.on('message', (message) => {
        if (message.type === 'ready') {
          ready++;
          if (ready === poolSize) {
            console.log(`[stt] ready (${poolSize} worker${poolSize > 1 ? 's' : ''})`);
            resolve();
          }
          return;
        }
        const job = pending.get(message.id);
        if (!job) return;
        pending.delete(message.id);
        if (message.error) job.reject(new Error(message.error));
        else job.resolve({ text: message.text ?? '', wake: message.wake ?? false });
      });

      worker.on('error', (error) => {
        console.error('[stt] worker error:', error.message);
      });

      worker.on('exit', (code) => {
        console.error(`[stt] worker exited (code ${code}) — restarting on next use`);
        for (const job of pending.values()) job.reject(new Error('STT worker exited'));
        pending.clear();
        workers = [];
        readyPromise = null;
      });

      workers.push(worker);
    }
  });
  return readyPromise;
}

/** Transcribe a 16 kHz mono 16-bit PCM buffer to text (or '' if nothing heard). */
export async function transcribe(pcm16kMono) {
  return (await runJob(pcm16kMono, 'main')).text;
}

/**
 * Detection pass: quick tiny.en transcription. Returns { text, wake } where
 * `wake` is whether the transcript mentions the bot's name.
 */
export async function transcribeFast(pcm16kMono) {
  return runJob(pcm16kMono, 'fast');
}

async function runJob(pcm16kMono, model) {
  await initStt(); // ensures the pool is up; no-op once loaded

  const id = nextId++;
  const job = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
  const worker = workers[dispatch % workers.length];
  dispatch++;

  const buffer = pcm16kMono.buffer.slice(
    pcm16kMono.byteOffset,
    pcm16kMono.byteOffset + pcm16kMono.byteLength,
  );
  worker.postMessage({ type: 'transcribe', model, id, buffer }, [buffer]);
  return job; // { text, wake }
}

/**
 * Simple energy-based voice activity detection.
 *
 * Whisper hallucinates words from non-speech audio (keyboard clicks, mouse
 * clicks, music), so we gate transcriptions on this: real speech is sustained
 * audio (hundreds of ms of active frames), while clicks are brief bursts that
 * leave most of the buffer silent.
 *
 * Works on the same 16 kHz mono 16-bit PCM buffers that Whisper consumes.
 */
export function estimateSpeech(pcm16kMono) {
  const WINDOW_SAMPLES = 320; // 20 ms at 16 kHz
  const totalFrames = Math.floor(pcm16kMono.length / 2 / WINDOW_SAMPLES);
  let activeFrames = 0;

  for (let frame = 0; frame < totalFrames; frame++) {
    const base = frame * WINDOW_SAMPLES * 2;
    let sumSq = 0;
    for (let i = 0; i < WINDOW_SAMPLES; i++) {
      const sample = pcm16kMono.readInt16LE(base + i * 2) / 32768;
      sumSq += sample * sample;
    }
    const rms = Math.sqrt(sumSq / WINDOW_SAMPLES);
    if (rms >= config.vadRmsThreshold) activeFrames++;
  }

  const activeMs = activeFrames * 20;
  const totalMs = totalFrames * 20;
  const fraction = totalFrames > 0 ? activeFrames / totalFrames : 0;

  return {
    activeMs,
    totalMs,
    fraction,
    isSpeech: activeMs >= config.vadMinActiveMs && fraction >= config.vadMinActiveFraction,
  };
}
