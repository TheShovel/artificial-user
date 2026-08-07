import { config } from './config.js';

// Removes emoji (including flags, skin tones, ZWJ sequences, and variation
// selectors) from bot output so replies stay plain text.
const EMOJI_RE =
  /[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}\u{1F3FB}-\u{1F3FF}\u{200D}\u{FE0F}\u{20E3}]/gu;

export function stripEmojis(text) {
  return text.replace(EMOJI_RE, '').replace(/\s+/g, ' ').trim();
}

/** Removes bracketed segments like "[Sam]" or "[laughs]" from model output. */
export function stripBrackets(text) {
  return text.replace(/\[[^\]]*\]/g, ' ').replace(/\s+/g, ' ').trim();
}

/** Removes HTML/XML tags like "<br>" that small models sometimes leak. */
export function stripHtml(text) {
  return text.replace(/<\/?[a-z][^>]*>/gi, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * Detects whisper stutter-loops: transcripts where one word makes up most of
 * the clip ("I-I-I-I-I...", "bro bro bro bro..."). Real speech doesn't do
 * this; audio that slips past the VAD often does.
 */
export function isRepetitiveGarbage(text) {
  const flat = text.toLowerCase().replace(/[^a-z\s]/g, ' ');
  const words = flat.split(/\s+/).filter(Boolean);
  if (words.length === 0) return true;

  // One word dominates the whole clip.
  const counts = new Map();
  for (const word of words) counts.set(word, (counts.get(word) ?? 0) + 1);
  const maxCount = Math.max(...counts.values());
  if (words.length >= 6 && maxCount / words.length >= 0.75) return true;

  // One character repeated in a row ("aaaaaaa").
  if (/([a-z])\1{9,}/.test(flat)) return true;

  return false;
}

let nameRegex = null;

function getNameRegex() {
  if (!nameRegex) {
    const names = [config.botName, ...config.botAliases].map((n) => n.toLowerCase());
    const parts = [];
    for (const name of names) {
      parts.push(name, `${name}s`);
      if (name.endsWith('y')) parts.push(`${name.slice(0, -1)}\\w*`);
    }
    nameRegex = new RegExp(`\\b(?:${[...new Set(parts)].join('|')})\\b`, 'i');
  }
  return nameRegex;
}

/** Which wake word did the transcript use, if any (lowercased, e.g. "clanker")? */
export function botNameMentioned(text) {
  return text.toLowerCase().match(getNameRegex())?.[0] ?? null;
}

/**
 * Wake-word check: does this transcript talk to/about the bot?
 * Matches the bot's name plus aliases with word boundaries. Names ending in
 * "y" also get a fuzzy stem match so Whisper's spelling slips ("bobbi",
 * "bobbie") still trigger, while lookalikes like "robotic" or "robert"
 * don't.
 */
export function mentionsBotName(text) {
  return botNameMentioned(text) !== null;
}
