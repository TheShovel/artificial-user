import fs from 'node:fs';
import path from 'node:path';
import { config } from './config.js';

/**
 * Conversation memory, persisted to a JSON file so the bot remembers across
 * restarts. Keyed per conversation (the bot uses the guild id); `/forget`
 * clears the same key.
 *
 * Shape: { [convoKey]: { summary, notes, keywords, messages } }
 *
 * Tiered compression so nothing important is lost while older content takes
 * less and less space:
 *   - `messages`  — the most recent turns, raw
 *   - `notes`     — short fact-dense lines from slightly older turns
 *   - `keywords`  — frequency-ranked content words from everything older;
 *                   words that stop mattering get dropped first, so the list
 *                   compresses more and more as the conversation grows
 */
class Memory {
  constructor(filePath) {
    this.filePath = filePath;
    this.data = {};
    this.saveTimer = null;
    this.load();
  }

  load() {
    try {
      this.data = JSON.parse(fs.readFileSync(this.filePath, 'utf8'));
    } catch {
      this.data = {};
    }
    // Migrate older formats to { summary, notes, keywords, messages }.
    for (const [channelId, value] of Object.entries(this.data)) {
      if (Array.isArray(value)) {
        this.data[channelId] = { summary: null, notes: null, keywords: {}, messages: value };
      } else if (!value || !Array.isArray(value.messages)) {
        this.data[channelId] = { summary: null, notes: null, keywords: {}, messages: [] };
      }
      // Seed the keyword frequency map from an existing keyword summary string.
      if (value && typeof value === 'object' && value.summary && !value.keywords) {
        const words = String(value.summary).replace(/^Topics: /, '').split(', ').filter(Boolean);
        this.data[channelId].keywords = Object.fromEntries(words.map((w) => [w, 1]));
      }
    }
  }

  /** Write to disk immediately (used on shutdown). */
  flush() {
    clearTimeout(this.saveTimer);
    this.saveTimer = null;
    try {
      fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
      fs.writeFileSync(this.filePath, JSON.stringify(this.data, null, 2));
    } catch (error) {
      console.error('[memory] save failed:', error.message);
    }
  }

  /** Debounced write: keeps sync disk I/O off the hot path during conversation. */
  save() {
    clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => this.flush(), 500);
  }

  /** Recent messages plus any earlier-conversation summary. */
  getState(channelId) {
    const state = this.data[channelId] ?? { summary: null, notes: null, keywords: {}, messages: [] };
    return {
      summary: state.summary ?? null,
      notes: state.notes ?? null,
      keywords: state.keywords ?? {},
      messages: state.messages ?? [],
    };
  }

  /** Recent raw messages only. */
  get(channelId) {
    return this.getState(channelId).messages;
  }

  add(channelId, role, content) {
    const state = this.data[channelId] ?? { summary: null, notes: null, keywords: {}, messages: [] };
    // Cap each message: whisper occasionally produces huge degenerate loops
    // (e.g. "merci d'être vraiment" x40) that would blow up the context.
    state.messages.push({ role, content: String(content).slice(0, config.maxMessageChars) });

    // Tiered fold: when raw history exceeds the window, older messages are
    // compressed progressively — recent facts stay as short notes, everything
    // older becomes frequency-ranked keywords that compress more and more.
    if (state.messages.length >= config.summarizeAfter) {
      const keep = config.summaryKeep;
      const old = state.messages.slice(0, state.messages.length - keep);
      // Only fold what PEOPLE said — the bot's own replies can be hallucinated
      // garbage and would pollute long-term memory if extracted as facts.
      const userMessages = old.filter((m) => m.role === 'user');

      // Keywords: accumulate frequencies across folds; when over budget, drop
      // the least-mentioned words first ("compresses more and more").
      for (const [word, count] of extractKeywords(userMessages)) {
        state.keywords[word] = (state.keywords[word] ?? 0) + count;
      }
      const ranked = Object.entries(state.keywords).sort((a, b) => b[1] - a[1]);
      state.keywords = Object.fromEntries(ranked.slice(0, config.summaryKeywordCap));
      state.summary = ranked
        .slice(0, config.summaryShownKeywords)
        .map(([word]) => word)
        .join(', ');

      // Notes: keep the newest short fact-dense user lines, capped.
      const freshNotes = extractShortNotes(userMessages)
        .split(' | ')
        .filter(Boolean);
      const allNotes = [...(state.notes ? state.notes.split(' | ') : []), ...freshNotes];
      state.notes = [...new Set(allNotes)].slice(-config.summaryNoteCap).join(' | ');

      state.messages = state.messages.slice(-keep);
      if (old.length >= 3) {
        console.log(`[memory] folded ${old.length} messages (conversation ${channelId})`);
      }
    }

    this.data[channelId] = state;
    this.save();
  }

  clear(channelId) {
    delete this.data[channelId];
    this.save();
  }
}

/** Words too common to be useful memory hooks. */
const STOPWORDS = new Set([
  'the','a','an','and','or','but','so','for','with','about','from','that','this','these',
  'those','have','has','had','was','were','been','being','they','them','their','there',
  'just','like','yeah','dude','man','guy','guys','you','your','yours','youre','im','ive',
  'its','it','is','are','not','no','yes','what','when','where','why','who','how','get',
  'got','going','go','come','coming','know','think','thinkin','talk','talking','say',
  'said','thing','things','one','two','out','up','down','all','some','any','really',
  'pretty','kinda','little','big','right','ok','okay','hey','oh','lol','haha','hahaha',
  'much','many','more','most','other','another','still','even','well','good','bad',
  'also','would','could','should','will','can','cant','want','wanna','need','let','us',
]);

/**
 * Keep the shortest fact-dense user lines from folded messages verbatim,
 * e.g. "my dog is named biscuit" — short statements are usually the facts.
 */
function extractShortNotes(messages) {
  return messages
    .filter((m) => m.role === 'user')
    .map((m) => ({
      text: m.content.replace(/^\[[^\]]*\]\s*/, '').trim(),
      words: m.content.trim().split(/\s+/).length,
    }))
    .filter((m) => m.text.length > 0 && m.words <= 8)
    .sort((a, b) => a.words - b.words)
    .slice(0, 3)
    .map((m) => m.text)
    .join(' | ')
    .slice(0, 150);
}

/**
 * Extract distinctive content words from a set of messages, returning
 * [word, count] pairs, e.g. [["biscuit", 3], ["chicago", 1]] — cheap
 * memory hooks for the model, weighted by how often they were said.
 */
function extractKeywords(messages) {
  const seen = new Map();
  for (const m of messages) {
    // Drop the "[Name]" prefix, keep the body.
    const body = m.content.replace(/^\[[^\]]*\]\s*/, '');
    const words = body.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/);
    for (const word of words) {
      if (word.length > 3 && !STOPWORDS.has(word)) {
        seen.set(word, (seen.get(word) ?? 0) + 1);
      }
    }
  }
  return [...seen.entries()];
}

export const memory = new Memory(config.memoryFile);
