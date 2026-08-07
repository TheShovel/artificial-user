import { createRequire } from 'node:module';
import { config } from './config.js';
import { stripEmojis, stripHtml } from './text.js';

const require = createRequire(import.meta.url);

// meSpeak: eSpeak compiled to JavaScript. Robotic, but fully local and
// dependency-free. (GPL — see mespeak package for details.)
const mespeak = require('mespeak');
mespeak.loadConfig(require('mespeak/src/mespeak_config.json'));
mespeak.loadVoice(require('mespeak/voices/en/en-us.json'));
mespeak.setDefaultVoice('en/en-us');

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

/** Synthesize text and return a WAV file as a Buffer (22.05 kHz mono PCM). */
export function synthesize(text) {
  const phrase = truncateForSpeech(expandTextisms(cleanForSpeech(text)));
  if (!phrase) return null;
  const wav = mespeak.speak(phrase, {
    rawdata: 'buffer',
    speed: config.ttsSpeed,
    pitch: config.ttsPitch,
    variant: config.ttsVariant,
  });
  return wav?.length ? wav : null;
}
