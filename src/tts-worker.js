import { parentPort } from 'node:worker_threads';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

// meSpeak: eSpeak compiled to JavaScript. Robotic, but fully local and
// dependency-free. (GPL — see mespeak package for details.)
// Runs in its own worker because synthesis is synchronous and can take a few
// hundred ms; on the main thread that would freeze audio capture in every
// server the bot is connected to.
let mespeak = require('mespeak');
mespeak.loadConfig(require('mespeak/src/mespeak_config.json'));
mespeak.loadVoice(require('mespeak/voices/en/en-us.json'));
mespeak.setDefaultVoice('en/en-us');

// eSpeak's WASM heap corrupts after many calls (the known "~80th call" bug),
// which makes it throw RangeErrors mid-synthesis. Reloading the whole mespeak
// module gives a fresh engine, so we do that periodically and after failures.
let speakCalls = 0;
const REBUILD_EVERY = 40;

function rebuildEngine() {
  try {
    const espeakPath = require.resolve('mespeak/src/ESpeak.js');
    const indexPath = require.resolve('mespeak');
    delete require.cache[espeakPath];
    delete require.cache[indexPath];
    mespeak = require('mespeak');
    mespeak.loadConfig(require('mespeak/src/mespeak_config.json'));
    mespeak.loadVoice(require('mespeak/voices/en/en-us.json'));
    mespeak.setDefaultVoice('en/en-us');
  } catch (error) {
    console.error('[tts] engine rebuild failed:', error.message);
  }
}

function trySpeak(phrase, speed, pitch, variant) {
  return mespeak.speak(phrase, {
    rawdata: 'buffer',
    speed,
    pitch,
    variant,
  });
}

/** Synthesize a phrase to a WAV Buffer (22.05 kHz mono PCM), or null. */
function synthesize(phrase, speed, pitch, variant) {
  try {
    if (++speakCalls % REBUILD_EVERY === 0) rebuildEngine(); // prevent corruption
    const wav = trySpeak(phrase, speed, pitch, variant);
    if (wav?.length) return wav;
  } catch (error) {
    console.error('[tts] synthesis failed, rebuilding engine:', error.message);
  }

  // Retry once after rebuilding the engine (self-heals the ~80th-call bug).
  rebuildEngine();
  try {
    const wav = trySpeak(phrase, speed, pitch, variant);
    if (wav?.length) return wav;
  } catch (error) {
    console.error('[tts] synthesis failed after rebuild:', error.message);
  }
  return null;
}

parentPort.on('message', (message) => {
  if (message.type !== 'synthesize') return;
  const wav = synthesize(message.text, message.speed, message.pitch, message.variant);
  if (wav) {
    // Copy into an exactly-sized buffer and transfer it (zero-copy to main).
    const buf = Buffer.alloc(wav.length);
    buf.set(wav);
    parentPort.postMessage({ id: message.id, wav: buf }, [buf.buffer]);
  } else {
    parentPort.postMessage({ id: message.id, wav: null });
  }
});

parentPort.postMessage({ type: 'ready' });
