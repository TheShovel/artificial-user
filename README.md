# artificial-user

A Discord bot that hangs out in your voice channel. It hears what people say,
talks back like a friend, and remembers the conversation. Everything runs
locally on your machine.

## What it's like

Bobby is a regular person in the call. Casual, short replies, no
assistant-speak. It knows who's talking and won't call you by the wrong name.

**Wake words.** Say "Bobby" (or "robot", "ai", "clanker") and it will answer.
Calling it a "clanker" is an insult, so it claps back rudely. It also chimes
into the flow of conversation on its own, just not constantly.

**It remembers.** Facts people mention stick around, compressed as the
conversation grows. `/forget` wipes it clean.

**Multiple servers.** It can hang out in several servers at the same time.
Each server gets its own conversation memory, its own `/say` cooldown, and
its own voice connection. Work is spread across worker threads so one busy
server doesn't make it go quiet in another.

## Requirements

- Node.js 20 or newer
- `ffmpeg` on your PATH
- Ollama running with a model pulled: `ollama pull qwen3.5:2b`

## Setup

1. Create a bot at <https://discord.com/developers/applications> (New Application, then the Bot tab, reset the token).
2. No privileged intents needed. Invite it with the `bot` and `applications.commands` scopes, and the `Connect` + `Speak` permissions.
3. Copy the config and paste your token:

```sh
cp .env.example .env
```

4. Start it:

```sh
npm install
npm start
```

On the first run it downloads the speech models (a few hundred MB total) into
`./.cache` and `./models`. You'll see `[bot] logged in as Bobby#4089` when it's
ready.

## Commands

| Command | What it does |
|---|---|
| `/join` | Joins your voice channel and starts listening |
| `/leave` | Leaves and stops listening |
| `/say text` | Makes it say something out loud. It remembers having said it. Has a 30 second cooldown per server. |
| `/forget` | Clears its memory for this server |
| `/status` | Shows connection state, memory size, and which model it uses |

## How it works

```
you speak -> captured audio -> tiny Whisper checks if it's for Bobby (~0.4s)
                              -> bigger Whisper transcribes properly, but only
                                 when Bobby is actually going to answer
                              -> Ollama (qwen3.5:2b) writes a reply
                              -> eSpeak says it out loud
```

Each stage runs off the main thread so one server's load doesn't stall
another: transcription runs in dedicated Whisper workers, speech synthesis in
its own worker, and every server gets its own turn queue.

A few details worth knowing:

- **Hearing.** Silero VAD filters out music, clicks, and game audio before
  Whisper ever runs, so it doesn't hallucinate words from noise. Whisper-tiny
  does fast detection, and whisper-base only runs when a reply is coming.
- **Thinking.** Replies are capped to a couple of short sentences. The model
  can't loop on phrases (repeat penalty), won't repeat itself, gets nudged if
  it fixates on one topic, and never ends a reply on a question.
- **Memory.** The last 6 messages stay raw. Older ones get compressed into
  short fact lines and keywords. It only remembers what people said, not its
  own rambling. Saved to `data/memory.json`, so it survives restarts.
- **Talking.** eSpeak is a basic robotic voice on purpose: it costs almost
  nothing to run. The text is cleaned up first, so abbreviations like "idk"
  are said properly.

## Tuning

Everything is in `.env`, with comments in `.env.example`. The most useful
knobs:

- `BOT_NAME`, `BOT_ALIASES` (default `Bobby` plus `robot,ai,clanker`)
- `UNNAMED_COOLDOWN_MS` (how often it chimes in on its own, default 20s)
- `CONTINUATION_WINDOW_MS` (how long after speaking it stays in the flow)
- `LLM_MODEL` (swap to a bigger or smaller model anytime)
- `STT_MODEL` (transcription accuracy vs speed)
- `STT_MAIN_WORKERS` (how many quality transcription workers; raise it when
  the bot is in several servers at once)
- `SYSTEM_PROMPT` (rewrite its whole personality)

## Sanity check

```sh
npm run selftest
```

This tests the whole pipeline without Discord: speak with TTS, transcribe it
back, check the wake word, hit Ollama, verify memory. Good for checking
nothing broke after a change.

## Troubleshooting

- **`[llm] Ollama not reachable`** -> start Ollama and pull the model.
- **It doesn't respond** -> check the log. `not addressed` means it decided
  the message wasn't for it, `throttled` means the cooldown is active,
  `nothing intelligible` means it heard noise it couldn't make sense of.
- **It responds to noise** -> the VAD handles most of it. If game audio slips
  through, raise `VAD_MIN_CONFIDENT_FRAMES`.
- **Replies feel slow** -> the delay is mostly transcription. Lower
  `SILENCE_DURATION_MS` or switch `STT_MODEL` to `onnx-community/whisper-tiny.en`.

## Notes

- The voice is eSpeak via `mespeak`, which is GPL. Deliberately basic, and
  swappable in `src/tts.js` if you want something nicer.
- The models are small on purpose so it runs on a normal laptop. That means
  transcripts and replies are occasionally weird. Upgrading `LLM_MODEL` or
  `STT_MODEL` is the easiest way to get better results.
