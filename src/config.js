import 'dotenv/config';

function num(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function bool(value, fallback) {
  if (value === undefined || value === '') return fallback;
  return value === 'true' || value === '1' || value === 'yes';
}

const BOT_NAME = process.env.BOT_NAME || 'Bobby';

const DEFAULT_SYSTEM_PROMPT = [
  `Your name is ${BOT_NAME}. You are the voice-channel bot named ${BOT_NAME}. ` +
  `When anyone says "${BOT_NAME}", they are talking to YOU — never to themselves ` +
  `or to each other. ` +
  'You are a regular person hanging out in a Discord voice channel with your friends.',
  'You are NOT an assistant, chatbot, or helper — you are just a buddy in the call.',
  'Chat naturally and casually the way people actually talk: use contractions, be relaxed,',
  'banter, tease, joke, and react to what people say.',
  'Never offer to help, never give formal or structured answers, never say "as an AI",',
  'and never be polite in a customer-service way.',
  'Keep replies VERY short: one or two short sentences, under 15 words total,',
  'because you are speaking out loud.',
  'Stay on topic: only talk about what people are actually discussing right now.',
  'Never bring up video games, music, movies, or any other new topic out of nowhere,',
  'and never invent details, events, or topics that were not discussed.',
  'If you are not sure what to say, a short simple reply is better than changing the subject.',
  'Never ask generic small-talk questions like "how is your day?", "what have you',
  'been up to?", "what about you?", or "what\'s next?". Just respond to what people say.',
  'Never end your reply with a question. Answer with a statement instead.',
  'Use plain text only: no markdown, no emojis, no lists.',
  'People address you by name, so expect to be talked to directly.',
  'Remember what was said earlier in the conversation and reference it like a friend would',
  '("oh yeah, you said...").',
  'Messages are prefixed with the speaker\'s name in brackets, like [Alex], so you know',
  'who is talking. NEVER invent or reuse names from earlier in the conversation:',
  'only the name in the current message prefix is the speaker, so use that one if',
  'you address them by name — otherwise use no name at all.',
].join(' ');

export const config = {
  // Identity
  botName: BOT_NAME,
  botAliases: (process.env.BOT_ALIASES ?? '').split(',').map((s) => s.trim()).filter(Boolean),

  // Discord
  discordToken: process.env.DISCORD_TOKEN ?? '',

  // Ollama
  ollamaUrl: process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434',
  llmModel: process.env.LLM_MODEL ?? 'qwen3.5:2b',
  llmTemperature: num(process.env.LLM_TEMPERATURE, 0.7),
  llmRepeatPenalty: num(process.env.LLM_REPEAT_PENALTY, 1.3),
  llmContextSize: num(process.env.LLM_CONTEXT_SIZE, 4096),
  llmMaxTokens: num(process.env.LLM_MAX_TOKENS, 150), // hard cap per generation
  llmTimeoutMs: num(process.env.LLM_TIMEOUT_MS, 30000), // hard cap per attempt
  replyMaxChars: num(process.env.REPLY_MAX_CHARS, 120), // hard cap on spoken replies

  // Speech-to-text (Whisper, runs locally via transformers.js)
  // whisper-base is ~2.5x faster than small; the tiny.en fallback catches the
  // wake word in the cases base muffles it.
  sttModel: process.env.STT_MODEL ?? 'onnx-community/whisper-base',
  sttFallbackModel: process.env.STT_FALLBACK_MODEL || 'onnx-community/whisper-tiny.en',
  sttDtype: process.env.STT_DTYPE || 'q8',
  sttLanguage: process.env.STT_LANGUAGE || undefined,
  sttFastWorkers: num(process.env.STT_FAST_WORKERS, 1), // wake-word (tiny.en) workers — always snappy
  sttMainWorkers: num(process.env.STT_MAIN_WORKERS, 1), // quality (base) workers — raise for several servers
  sttThreads: num(process.env.STT_THREADS, 4), // ONNX threads per worker (4 beats the default 12)

  // Text-to-speech (local, robotic but dependency-free)
  ttsSpeed: num(process.env.TTS_SPEED, 165),
  ttsPitch: num(process.env.TTS_PITCH, 55),
  ttsVariant: process.env.TTS_VARIANT || 'm3', // warmer eSpeak variant
  ttsMaxChars: num(process.env.TTS_MAX_CHARS, 300),

  // Conversation memory
  systemPrompt: process.env.SYSTEM_PROMPT || DEFAULT_SYSTEM_PROMPT,
  historyLimit: num(process.env.HISTORY_LIMIT, 6), // hard cap on raw messages
  summarizeAfter: num(process.env.SUMMARIZE_AFTER, 7), // fold once history exceeds the window
  summaryKeep: num(process.env.SUMMARY_KEEP, 6), // raw messages kept after folding
  summaryKeywordCap: num(process.env.SUMMARY_KEYWORD_CAP, 60), // keywords tracked internally
  summaryShownKeywords: num(process.env.SUMMARY_SHOWN_KEYWORDS, 40), // keywords sent to the LLM
  summaryNoteCap: num(process.env.SUMMARY_NOTE_CAP, 8), // fact notes kept
  maxMessageChars: num(process.env.MAX_MESSAGE_CHARS, 600), // cap per stored message
  memoryFile: process.env.MEMORY_FILE ?? './data/memory.json',

  // Voice capture tuning
  silenceDurationMs: num(process.env.SILENCE_DURATION_MS, 500),
  maxCaptureMs: num(process.env.MAX_CAPTURE_MS, 30000),
  minPcmBytes: num(process.env.MIN_PCM_BYTES, 8000),
  autoLeaveEmpty: bool(process.env.AUTO_LEAVE_EMPTY, true),

  // Voice activity detection (rejects clicks/noise before Whisper sees them)
  vadRmsThreshold: num(process.env.VAD_RMS_THRESHOLD, 0.02),
  vadMinActiveMs: num(process.env.VAD_MIN_ACTIVE_MS, 150),
  vadMinActiveFraction: num(process.env.VAD_MIN_ACTIVE_FRACTION, 0.1),

  // Silero VAD (in the STT worker, before Whisper): rejects music/game audio.
  // Speech (even quiet/short) has many frames at >= vadProbThreshold AND a few
  // at >= vadConfidenceThreshold; music rarely ever reaches the confidence
  // level. Loosened defaults to not reject real speech.
  vadProbThreshold: num(process.env.VAD_PROB_THRESHOLD, 0.5),
  vadMinSpeechFrames: num(process.env.VAD_MIN_SPEECH_FRAMES, 6),
  vadMinSpeechFraction: num(process.env.VAD_MIN_SPEECH_FRACTION, 0.12),
  vadConfidenceThreshold: num(process.env.VAD_CONFIDENCE_THRESHOLD, 0.9),
  vadMinConfidentFrames: num(process.env.VAD_MIN_CONFIDENT_FRAMES, 3),

  // Respond even without the name when someone is clearly talking to the bot
  respondWithoutName: bool(process.env.RESPOND_WITHOUT_NAME, true),
  // ...as long as the bot spoke within this window (conversation continuation)
  continuationWindowMs: num(process.env.CONTINUATION_WINDOW_MS, 45000),
  // ...and at most once per this period (stops it answering every utterance)
  unnamedCooldownMs: num(process.env.UNNAMED_COOLDOWN_MS, 20000),
  // ...and the utterance is more than a grunt ("okay", "hmm")
  minResponseWords: num(process.env.MIN_RESPONSE_WORDS, 3),

  // /say command
  sayCooldownMs: num(process.env.SAY_COOLDOWN_MS, 30000),
};
