import { Worker } from 'node:worker_threads';
import { config } from './config.js';
import { stripEmojis, stripHtml } from './text.js';

// Text-to-speech runs in a worker thread: eSpeak synthesis is synchronous and
// can take a few hundred ms, which would otherwise freeze the bot's event loop
// (and with it, audio capture in every server it is connected to). The worker
// owns the mespeak engine and its periodic rebuild, and returns WAV buffers.

let ttsWorker = null;
let ttsReady = null;
let ttsResolveReady = null;
let nextId = 1;
const pending = new Map(); // job id -> resolve

function getWorker() {
  if (ttsWorker) return ttsReady;
  ttsReady = new Promise((resolve) => {
    ttsResolveReady = resolve;
  });
  ttsWorker = new Worker(new URL('./tts-worker.js', import.meta.url));
  ttsWorker.on('message', (message) => {
    if (message.type === 'ready') {
      ttsResolveReady?.();
      return;
    }
    const resolve = pending.get(message.id);
    if (!resolve) return;
    pending.delete(message.id);
    resolve(message.wav ?? null);
  });
  ttsWorker.on('error', (error) => console.error('[tts] worker error:', error.message));
  ttsWorker.on('exit', (code) => {
    console.error(`[tts] worker exited (code ${code}) — restarting on next use`);
    for (const resolve of pending.values()) resolve(null);
    pending.clear();
    ttsWorker = null;
    ttsReady = null;
  });
  return ttsReady;
}

// Words a robotic voice would mangle if read literally; expanded first.
// (The model occasionally still emits these.)
const TEXTISMS = {
  idk: 'I don\'t know',
  idc: 'I don\'t care',
  btw: 'by the way',
  'im': 'I am',
  'ive': 'I have',
  imma: 'I will',
  gonna: 'going to',
  wanna: 'want to',
  'u': 'you',
  ur: 'your',
  rn: 'right now',
  fr: 'for real',
  lol: 'haha',
  lmao: 'haha',
  nvm: 'never mind',
  omg: 'oh my god',
  k: 'okay',
};

/** Expand textisms ("idk", "lol") so the robotic voice reads them properly. */
function expandTextisms(text) {
  return text
    .split(/(\s+)/)
    .map((word) => {
      const bare = word.toLowerCase().replace(/[^a-z']/g, '');
      const punct = word.replace(/[a-z']/gi, '');
      const expanded = TEXTISMS[bare];
      if (!expanded) return word;
      return punct ? `${punct}${expanded}${punct}` : expanded;
    })
    .join('');
}

/** Make text easier for a robotic voice to read aloud. */
function cleanForSpeech(text) {
  return stripEmojis(stripHtml(text))
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[*_~`#>|]/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\([^)]*\)/g, ' ') // drop parenthetical asides like "(hey)"
    .replace(/w\//g, ' with ')
    .replace(/&/g, ' and ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Cut long replies at a sentence boundary so TTS doesn't drone on. */
function truncateForSpeech(text) {
  const max = config.ttsMaxChars;
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const boundary = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('! '), cut.lastIndexOf('? '));
  const end = boundary > max * 0.4 ? boundary + 1 : max;
  return text.slice(0, end);
}

/**
 * Synthesize text and return a Promise<Buffer|null> for the WAV file
 * (22.05 kHz mono PCM). Runs in the TTS worker so the event loop stays
 * responsive even when several servers are talking at once.
 */
export async function synthesize(text) {
  const phrase = truncateForSpeech(expandTextisms(cleanForSpeech(text)));
  if (!phrase) return null;

  try {
    await getWorker();
  } catch {
    return null;
  }
  const worker = ttsWorker;
  if (!worker) return null; // died between await and post

  const id = nextId++;
  const wavPromise = new Promise((resolve) => pending.set(id, resolve));
  worker.postMessage({
    type: 'synthesize',
    id,
    text: phrase,
    speed: config.ttsSpeed,
    pitch: config.ttsPitch,
    variant: config.ttsVariant,
  });
  return wavPromise;
}
