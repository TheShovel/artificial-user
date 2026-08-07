import { config } from './config.js';
import { memory } from './memory.js';
import { stripBrackets, stripEmojis, stripHtml } from './text.js';

function stripThinkBlocks(text) {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/^think:[\s\S]*?\n/i, '')
    .trim();
}

/** Raw chat call to Ollama; no memory side effects. Retries once on failure. */
async function chat(messages, { signal, temperature, numCtx, think = false, maxTokens } = {}) {
  let lastError;
  for (let attempt = 0; attempt < 2; attempt++) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

    // Time out slow/stuck generations so a wedged Ollama can never hang the
    // bot's turn queue. Also honors the caller's abort signal.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), config.llmTimeoutMs);
    const onAbort = () => controller.abort();
    signal?.addEventListener('abort', onAbort);

    try {
      const response = await fetch(`${config.ollamaUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          model: config.llmModel,
          messages,
          stream: false,
          think,
          options: {
            temperature: temperature ?? config.llmTemperature,
            num_ctx: numCtx ?? config.llmContextSize,
            // Small models tend to loop; penalize repeating tokens and bound
            // how long a single generation can run.
            repeat_penalty: config.llmRepeatPenalty,
            num_predict: maxTokens ?? config.llmMaxTokens,
          },
        }),
      });

      if (!response.ok) {
        const body = await response.text().catch(() => '');
        throw new Error(`Ollama error ${response.status}: ${body.slice(0, 300)}`);
      }

      const data = await response.json();
      let reply = stripThinkBlocks(data.message?.content ?? '').trim();
      // Strip emojis, bracket artifacts, stray HTML tags, and hashtags (small
      // models leak "<br>", "[Name]:" and "#whatsup" formatting), but never
      // let it erase the reply entirely.
      const stripped = stripEmojis(stripBrackets(stripHtml(reply)))
        .replace(/#\w+/g, ' ')
        .replace(/\s+/g, ' ');
      reply = stripped || stripEmojis(reply);
      // Drop leading punctuation left over from the model imitating "[Name]:".
      reply = reply.replace(/^[\s:;,]*/, '').trim();
      if (!reply) throw new Error('Ollama returned an empty response');
      return reply;
    } catch (error) {
      if (error.name === 'AbortError') {
        if (signal?.aborted) throw error; // real interrupt
        lastError = new Error(`Ollama timed out after ${config.llmTimeoutMs}ms`);
      } else {
        lastError = error;
      }
      console.log(`[llm] attempt ${attempt + 1} failed: ${lastError.message} — retrying`);
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    }
  }
  throw lastError;
}

/** Normalize text for duplicate detection. */
function normalize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * Is this reply basically a repeat of the previous one? Catches tiny models
 * that lock onto a phrase and echo it forever. Short replies only count when
 * identical; longer ones when word overlap is high.
 */
export function isRepeat(a, b) {
  const na = normalize(a);
  const nb = normalize(b);
  if (!na || !nb) return false;
  if (na === nb) return true;
  const wa = na.split(' ');
  const wb = nb.split(' ');
  if (wa.length < 10 || wb.length < 10) return false;
  const setB = new Set(wb);
  let overlap = 0;
  for (const word of wa) if (setB.has(word)) overlap++;
  return overlap / Math.min(wa.length, wb.length) >= 0.6;
}

/** Small-model filler: asking about the other person's day instead of replying. */
const SMALL_TALK_PATTERNS = [
  /how('s| is| are| was| were)\s+(your|ya|u|you)\s+(day|weekend|night|evening|morning)/i,
  /how('s| is)\s+it\s+going/i,
  /how\s+are\s+(you|ya|u)(\s+doing)?/i,
  /what('s| is)\s+(up|new)(\s+with)?\s+(you|ya|u|yourself)/i,
  /what\s+have\s+(you|ya|u)\s+been\s+up\s+to/i,
  /(how|what)\s+about\s+(you|ya|u|yourself)/i,
  /how('s| is)\s+your\s+day(\s+going)?/i,
];

export function isSmallTalkFiller(text) {
  return SMALL_TALK_PATTERNS.some((re) => re.test(text.toLowerCase()));
}

/** Hard cap: cut long replies down to short, complete sentences. */
export function shortenReply(text, maxChars = config.replyMaxChars) {
  const trimmed = text.trim();
  if (trimmed.length <= maxChars) return trimmed;

  // Take complete sentences while they fit within the cap.
  const sentences = trimmed.match(/[^.!?]*[.!?]+(?:\s|$)/g) ?? [];
  let result = '';
  for (const sentence of sentences) {
    if ((result + sentence).trim().length > maxChars) break;
    result += sentence;
  }
  result = result.trim();
  if (result) return result;

  // No sentence fits whole: keep the first one if it's reasonable, otherwise
  // cut at a word boundary and trail off.
  const first = sentences[0]?.trim() ?? trimmed;
  if (sentences.length > 0 && first.length <= maxChars * 2) return first;
  const cut = (sentences[0] ?? trimmed).slice(0, maxChars);
  const space = cut.lastIndexOf(' ');
  return (space > 0 ? cut.slice(0, space) : cut) + '...';
}

/**
 * Send a user utterance to the Ollama model and return the reply.
 * The user message and the reply are stored in the channel's memory.
 *
 * `speakerName` is included with the message ("[Alex] ...") so the model can
 * respond to people like a friend would. Pass an AbortSignal to cancel a
 * pending reply (e.g. when interrupted).
 */
export async function ask(convoKey, userText, signal, speakerName, insulted = false, addressed = false) {
  const safeName = speakerName ? String(speakerName).replace(/[\[\]]/g, '').trim() : '';
  const knownName =
    safeName && safeName.toLowerCase() !== 'someone' ? safeName : '';
  const content = knownName ? `[${knownName}] ${userText}` : userText;
  memory.add(convoKey, 'user', content);

  // Recent raw messages, plus a compressed summary of anything older.
  const { summary, notes, messages } = memory.getState(convoKey);
  const earlier = summary
    ? `Earlier in the conversation, topics mentioned: ${summary}${notes ? `. Facts: ${notes}` : ''}`
    : null;
  const context = [
    { role: 'system', content: config.systemPrompt },
    ...(insulted
      ? [
          {
            role: 'system',
            content:
              'The user just called you "clanker", which is an insult for droids. ' +
              'Clap back rudely and defensively, like a friend who just got roasted. ' +
              'Keep it short and stay in character.',
          },
        ]
      : []),
    ...(addressed
      ? [
          {
            role: 'system',
            content:
              `The speaker just said your name (${config.botName}). You are the bot ` +
              `named ${config.botName} — they are talking to you. Answer them.`,
          },
        ]
      : []),
    ...(earlier ? [{ role: 'system', content: earlier }] : []),
    // History with everyone's names, but make the CURRENT speaker explicit
    // right before their message so the model never uses another name for them.
    ...messages.slice(0, -1),
    ...(knownName
      ? [
          {
            role: 'system',
            content:
              `The person you are responding to now is ${knownName}. ` +
              `Only call them ${knownName} — do not use any other name from the conversation, ` +
              'and spell their name exactly as shown.',
          },
        ]
      : []),
    messages.at(-1),
  ];

  let reply = await chat(context, { signal });

  // Anti-repeat: if this reply is nearly identical to the last one the model
  // gave, regenerate it with a nudge instead of echoing again.
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
  if (lastAssistant && isRepeat(reply, lastAssistant.content)) {
    console.log('[llm] reply repeated the previous one — regenerating');
    const nudged = await chat(
      [...context, { role: 'system', content: 'Stop repeating yourself. Say something completely new and different.' }],
      { signal },
    );
    if (nudged) reply = nudged;
  }

  // Anti-fixation: if a content word from the new reply also appeared in the
  // last two replies, the model has latched onto a topic ("coffee again").
  // Nudge it to change the subject. (Needs >= 2 prior replies; `every` on an
  // empty list is vacuously true, which would false-trigger.)
  const recentAssistants = messages.filter((m) => m.role === 'assistant').slice(-2);
  const fixated =
    recentAssistants.length >= 2
      ? contentWords(reply).find((word) =>
          recentAssistants.every((m) => normalize(m.content).split(' ').includes(word)),
        )
      : undefined;
  if (fixated) {
    console.log(`[llm] keeps bringing up "${fixated}" — nudging to change topic`);
    const nudged = await chat(
      [
        ...context,
        // Note: deliberately does NOT name the word — naming it makes the model
        // parrot it back.
        { role: 'system', content: 'You keep repeating the same topic. Change the subject completely and say something new.' },
      ],
      { signal },
    );
    if (nudged) reply = nudged;
  }

  // Anti-small-talk: the model falls back to "how's your day?" when it has
  // nothing to say. If the reply is mostly that, regenerate it.
  if (isSmallTalkFiller(reply)) {
    console.log('[llm] small-talk filler — regenerating');
    const nudged = await chat(
      [
        ...context,
        { role: 'system', content: 'Do not ask the other person about their day, weekend, or what they have been up to. Respond directly to what was said instead.' },
      ],
      { signal },
    );
    if (nudged) reply = nudged;
  }

  // Keep replies short (the model rambles; this is the enforcement).
  reply = shortenReply(reply);

  memory.add(convoKey, 'assistant', reply);
  return reply;
}

/** Words too common to count as "topics" for the fixation guard. */
const STOPWORDS = new Set([
  'the','a','an','and','or','but','so','for','with','about','from','that','this','these',
  'those','have','has','had','was','were','been','being','they','them','their','there',
  'just','like','yeah','dude','man','guy','guys','you','your','yours','youre','im','ive',
  'its','it','is','are','not','no','yes','what','when','where','why','who','how','get',
  'got','going','go','come','coming','know','think','thinkin','talk','talking','say',
  'said','thing','things','one','two','out','up','down','all','some','any','really',
  'pretty','kinda','little','big','right','ok','okay','hey','oh','lol','haha','hahaha',
  'much','many','more','most','other','another','still','even','well','good','bad',
  'something','someone','somebody','anything','anyone','nothing','everything','everyone',
  'somewhere','anywhere','watch','watching','wanna','gonna','maybe','sure','yeah',
  'jody','jodi','jodie','jodiiee', // name misspellings the model fixates on
]);

/** Distinctive words (>3 chars, not stopwords) in a reply. */
function contentWords(text) {
  return [...new Set(normalize(text).split(' '))].filter((w) => w.length > 3 && !STOPWORDS.has(w));
}

/** Verify Ollama is reachable and the configured model exists. */
export async function checkOllama() {
  const response = await fetch(`${config.ollamaUrl}/api/tags`);
  if (!response.ok) {
    throw new Error(`Ollama not reachable at ${config.ollamaUrl} (HTTP ${response.status})`);
  }
  const data = await response.json();
  const models = (data.models ?? []).map((m) => m.name);
  if (!models.some((name) => name === config.llmModel || name.startsWith(`${config.llmModel}:`))) {
    throw new Error(
      `Model "${config.llmModel}" not found in Ollama. Available: ${models.join(', ') || '(none)'}`,
    );
  }
}
