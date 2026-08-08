import { Worker } from 'node:worker_threads';
import { config } from './config.js';

/**
 * Speech-to-text runs in two dedicated worker pools so inference never blocks
 * the bot and servers don't starve each other:
 *
 *   fast — whisper-tiny.en, the wake-word detector. EVERY utterance hits this
 *          pass, so it gets its own workers and is never queued behind slow
 *          quality transcriptions from another server.
 *   main — whisper-base, the quality pass that only runs when Bobby is about
 *          to answer. Multiple workers let several servers (or speakers)
 *          transcribe at the same time.
 *
 * Each worker loads only its own model, so a pool of N costs ~N × one model
 * in memory instead of N × both models.
 *
 * Dispatch picks the worker with the fewest jobs in flight; ties go to the
 * most recently used worker (warm CPU cache), so a single server keeps hitting
 * the same worker while several servers spread across the pool.
 */

const pools = {
  fast: createPool('fast', Math.max(1, config.sttFastWorkers)),
  main: createPool('main', Math.max(1, config.sttMainWorkers)),
};

let nextId = 1;
const pending = new Map(); // job id -> { resolve, reject, pool, entry }

function createPool(role, size) {
  const pool = {
    role,
    entries: [], // { worker, inFlight, lastUse }
    readyPromise: null,
    resolveReady: null, // resolves the current readyPromise (unblocks waiters if the pool dies)
  };

  const ensure = () => {
    if (pool.readyPromise) return pool.readyPromise;
    pool.readyPromise = new Promise((resolve) => {
      pool.resolveReady = resolve;
      let ready = 0;
      const spawn = () => {
        const entry = {
          worker: new Worker(new URL('./stt-worker.js', import.meta.url), {
            // Bound ONNX threads per worker: onnxruntime's default (all cores)
            // thrashes the cache and runs *slower*; workers × sttThreads uses
            // the machine well. The role picks which model the worker loads.
            env: {
              ...process.env,
              OMP_NUM_THREADS: String(config.sttThreads),
              STT_WORKER_ROLE: role,
            },
          }),
          inFlight: 0,
          lastUse: 0,
        };

        entry.worker.on('message', (message) => {
          if (message.type === 'ready') {
            ready++;
            if (ready === size) {
              console.log(`[stt] ${role} pool ready (${size} worker${size > 1 ? 's' : ''})`);
              resolve();
            }
            return;
          }
          const job = pending.get(message.id);
          if (!job || job.pool !== pool) return;
          pending.delete(message.id);
          entry.inFlight = Math.max(0, entry.inFlight - 1);
          if (message.error) job.reject(new Error(message.error));
          else job.resolve({ text: message.text ?? '', wake: message.wake ?? false });
        });

        entry.worker.on('error', (error) => {
          console.error(`[stt:${role}] worker error:`, error.message);
        });

        entry.worker.on('exit', (code) => {
          console.error(`[stt:${role}] worker exited (code ${code})`);
          const idx = pool.entries.indexOf(entry);
          if (idx !== -1) pool.entries.splice(idx, 1);
          // Only jobs that were on the dead worker fail; everything else carries on.
          for (const job of [...pending.values()]) {
            if (job.pool === pool && job.entry === entry) {
              pending.delete(job.id);
              job.reject(new Error('STT worker exited'));
            }
          }
          if (pool.entries.length === 0) {
            // Whole pool is down. Unblock anyone waiting on it — their next
            // utterance rebuilds the pool from scratch.
            pool.readyPromise = null;
            pool.resolveReady?.();
            pool.resolveReady = null;
          } else {
            spawn(); // keep the pool at its configured size
          }
        });

        pool.entries.push(entry);
      };
      for (let i = 0; i < size; i++) spawn();
    });
    return pool.readyPromise;
  };
  pool.ensure = ensure;
  return pool;
}

/** Worker with the fewest in-flight jobs; ties go to the warmest cache. */
function pick(pool) {
  let best = null;
  for (const entry of pool.entries) {
    if (!best || entry.inFlight < best.inFlight) {
      best = entry;
    } else if (entry.inFlight === best.inFlight && entry.lastUse > best.lastUse) {
      best = entry;
    }
  }
  return best;
}

/** Spawn both pools and wait for their models to load. Cached. */
export async function initStt() {
  await Promise.all([pools.fast.ensure(), pools.main.ensure()]);
}

/** Transcribe a 16 kHz mono 16-bit PCM buffer to text (or '' if nothing heard). */
export async function transcribe(pcm16kMono) {
  return (await runJob(pcm16kMono, pools.main)).text;
}

/**
 * Detection pass: quick tiny.en transcription. Returns { text, wake } where
 * `wake` is whether the transcript mentions the bot's name.
 */
export async function transcribeFast(pcm16kMono) {
  return runJob(pcm16kMono, pools.fast);
}

async function runJob(pcm16kMono, pool) {
  await pool.ensure(); // no-op if already up; rebuilds if the pool died

  const entry = pick(pool);
  if (!entry) throw new Error('STT pool unavailable — try again');

  const id = nextId++;
  entry.inFlight++;
  entry.lastUse = Date.now();
  const job = new Promise((resolve, reject) =>
    pending.set(id, { resolve, reject, pool, entry }),
  );

  const buffer = pcm16kMono.buffer.slice(
    pcm16kMono.byteOffset,
    pcm16kMono.byteOffset + pcm16kMono.byteLength,
  );
  entry.worker.postMessage({ type: 'transcribe', id, buffer }, [buffer]);
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
