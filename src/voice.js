import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { Readable } from 'node:stream';

const require = createRequire(import.meta.url);
const { OpusEncoder } = require('@discordjs/opus');
import {
  AudioPlayerStatus,
  EndBehaviorType,
  StreamType,
  VoiceConnectionStatus,
  createAudioPlayer,
  createAudioResource,
  entersState,
  joinVoiceChannel,
} from '@discordjs/voice';
import { config } from './config.js';
import { ask } from './llm.js';
import { estimateSpeech, transcribe, transcribeFast } from './stt.js';
import { botNameMentioned } from './text.js';
import { synthesize } from './tts.js';

/**
 * Resample the raw PCM stream that @discordjs/voice emits (48 kHz stereo,
 * s16le) down to 16 kHz mono s16le, which Whisper expects.
 */
function createResampler() {
  const ffmpeg = spawn('ffmpeg', [
    '-loglevel', 'error',
    '-f', 's16le', '-ar', '48000', '-ac', '2', '-i', 'pipe:0',
    '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1',
  ]);
  const chunks = [];
  ffmpeg.stdout.on('data', (chunk) => chunks.push(chunk));
  ffmpeg.stderr.on('data', (chunk) => console.error('[ffmpeg]', chunk.toString().trim()));
  return {
    stream: ffmpeg,
    async collect() {
      return new Promise((resolve) => {
        ffmpeg.on('close', () => resolve(Buffer.concat(chunks)));
        ffmpeg.stdin.on('error', () => {});
      });
    },
  };
}

export class VoiceBot {
  constructor(client) {
    this.client = client;
    /** @type {Map<string, object>} guildId -> per-guild state */
    this.guilds = new Map();
  }

  get(guildId) {
    return this.guilds.get(guildId);
  }

  async join(voiceChannel) {
    if (this.guilds.has(voiceChannel.guild.id)) return false;

    const connection = joinVoiceChannel({
      channelId: voiceChannel.id,
      guildId: voiceChannel.guild.id,
      adapterCreator: voiceChannel.guild.voiceAdapterCreator,
      selfDeaf: false,
      selfMute: false,
    });

    const player = createAudioPlayer();
    connection.subscribe(player);

    const state = {
      guildId: voiceChannel.guild.id,
      connection,
      player,
      voiceChannelId: voiceChannel.id,
      capturing: new Set(), // userIds currently being recorded
      turn: null, // Promise of the in-flight reply, if any
      lastSpokeAt: 0, // when the bot last spoke (conversation-continuation gate)
      lastWasQuestion: false, // did the bot's last reply ask a question?
      lastUnnamedResponseAt: 0, // throttle for unnamed responses
    };
    this.guilds.set(state.guildId, state);
    this.#setupListeners(state);
    return true;
  }

  #setupListeners(state) {
    // Record anyone who starts speaking; whether Bobby responds (or interrupts
    // his own reply) is decided after transcription, once we know if they
    // actually talked to him.
    state.connection.receiver.speaking.on('start', (userId) => {
      if (userId === this.client.user.id) return;
      if (state.capturing.has(userId)) return;
      this.#capture(state, userId);
    });

    // Clean up if the connection dies or is destroyed externally.
    state.connection.on('stateChange', (_oldState, newState) => {
      if (newState.status === VoiceConnectionStatus.Destroyed) {
        this.#cleanup(state);
      } else if (newState.status === VoiceConnectionStatus.Disconnected) {
        entersState(state.connection, VoiceConnectionStatus.Reconnecting, 5000).catch(() => {
          state.connection.destroy();
        });
      }
    });

    // Log playback errors instead of crashing.
    state.player.on('error', (error) => console.error('[player]', error.message));
  }

  /**
   * Does the transcript address another voice-channel member by name
   * ("I'll put you on, Hickey")? The current speaker and the bot's own names
   * are excluded, so we don't butt into side conversations.
   */
  #talksToOtherMember(state, speakerName, text) {
    const channel = this.client.channels.cache.get(state.voiceChannelId);
    const lower = text.toLowerCase();
    const speaker = speakerName.toLowerCase();
    const botNames = new Set([config.botName.toLowerCase(), ...config.botAliases.map((a) => a.toLowerCase())]);

    for (const member of channel?.members?.values() ?? []) {
      if (member.user.bot) continue;
      const display = (member.displayName ?? member.user.username ?? '').toLowerCase();
      if (!display || display === speaker || botNames.has(display)) continue;
      // Multi-word names match as substrings; single words with boundaries.
      const found = display.includes(' ')
        ? lower.includes(display)
        : new RegExp(`\\b${display.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`).test(lower);
      if (found) return true;
    }
    return false;
  }

  /**
   * Record one utterance from a user: capture their audio until ~700 ms of
   * silence, decode Opus -> PCM, resample it, transcribe it, then generate
   * and speak a reply.
   */
  #capture(state, userId) {
    state.capturing.add(userId);

    // `subscribe()` yields Opus packets, not PCM — decode them to 48 kHz
    // stereo s16le PCM before feeding ffmpeg. Uses the native libopus decoder
    // (robust against malformed packets); WASM decoders can crash here.
    const opusDecoder = new OpusEncoder(48000, 2, 'audio');
    const subscription = state.connection.receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: config.silenceDurationMs },
    });
    const resampler = createResampler();

    let stdinEnded = false;
    const endStdin = () => {
      if (!stdinEnded) {
        stdinEnded = true;
        resampler.stream.stdin.end();
      }
    };

    subscription.on('data', (opusPacket) => {
      try {
        const pcm = opusDecoder.decode(opusPacket);
        if (pcm?.length) resampler.stream.stdin.write(pcm);
      } catch {
        // Drop malformed packets instead of crashing.
      }
    });
    subscription.on('error', (error) => {
      console.error('[opus]', error.message);
      endStdin();
    });
    subscription.on('end', endStdin);
    subscription.on('close', endStdin);

    // Hard cap: never record one utterance for longer than this.
    const cap = setTimeout(() => subscription.destroy(), config.maxCaptureMs);
    subscription.on('close', () => clearTimeout(cap));

    resampler.collect().then(async (pcm) => {
      state.capturing.delete(userId);
      if (pcm.length < config.minPcmBytes) return; // too short to be speech

      // Reject clicks/noise before Whisper can hallucinate words from them.
      const speech = estimateSpeech(pcm);
      if (!speech.isSpeech) {
        console.log(
          `[vad] ignored non-speech (${speech.activeMs}ms active of ${speech.totalMs}ms, ` +
            `${(speech.fraction * 100).toFixed(0)}%)`,
        );
        return;
      }

      // Detection pass: whisper-tiny.en (~0.3 s) — fast, and hears the wake
      // word well. Returns the text plus whether the bot's name was said.
      let fast;
      try {
        fast = await transcribeFast(pcm);
      } catch (error) {
        console.error('[stt]', error);
        return;
      }
      const text = fast.text;
      if (!text) {
        console.log('[hear] (nothing intelligible)');
        return;
      }

      const user = this.client.users.cache.get(userId);
      const name = user?.displayName ?? user?.username ?? 'someone';
      console.log(`[hear] ${name}: ${text}`);

      // Respond when someone says the bot's name...
      const mentioned = fast.wake || botNameMentioned(text);
      const insulted = mentioned ? botNameMentioned(text).startsWith('clanker') : false;

      if (!mentioned) {
        // ...or when it's part of the live conversation: the bot spoke recently.
        // (A question-only gate was too strict — it made the bot nearly silent.)
        // Still skip single-word grunts like "okay" / "hmm".
        const inWindow = Date.now() - (state.lastSpokeAt ?? 0) < config.continuationWindowMs;
        const enoughWords = text.trim().split(/\s+/).length >= config.minResponseWords;
        if (!config.respondWithoutName || !inWindow || !enoughWords) {
          console.log(`[${config.botName.toLowerCase()}] not addressed (${name}: "${text}")`);
          return;
        }
        // Don't butt in when the speaker is clearly talking to someone else
        // ("I'll put you on, Hickey").
        if (this.#talksToOtherMember(state, name, text)) {
          console.log(`[${config.botName.toLowerCase()}] talking to someone else (${name}: "${text}")`);
          return;
        }
        // Throttle: at most one unnamed response per cooldown, so the bot
        // doesn't answer every single utterance (the wake word always works).
        if (Date.now() - state.lastUnnamedResponseAt < config.unnamedCooldownMs) {
          console.log(`[${config.botName.toLowerCase()}] throttled (${name}: "${text}")`);
          return;
        }
      }

      // The tiny pass found no wake word but we're responding anyway
      // (continuation) — grab the higher-quality transcript first.
      let replyText = text;
      if (!fast.wake) {
        try {
          replyText = (await transcribe(pcm)) || text;
        } catch {
          /* keep the tiny text */
        }
      }

      // The bot is addressed: if it is mid-reply, stop and cancel it.
      if (state.turn) {
        console.log(`[${config.botName.toLowerCase()}] interrupted current reply`);
        state.player.stop();
        state.interrupt?.abort();
      }

      if (!mentioned) state.lastUnnamedResponseAt = Date.now();
      await this.respond(state, replyText, name, insulted, mentioned);
    });
  }

  /**
   * Ask the LLM (with conversation memory) and speak the reply.
   * Turns are serialized per guild; if someone speaks while a reply is in
   * flight, the reply is stopped/cancelled so the new request takes over.
   */
  async respond(state, userText, speakerName, insulted = false, addressed = false) {
    const previousTurn = state.turn;
    let release;
    const turn = new Promise((resolve) => {
      release = resolve;
    });
    state.turn = turn;
    await previousTurn;

    const controller = new AbortController();
    state.interrupt = controller;
    try {
      const reply = await ask(state.guildId, userText, controller.signal, speakerName, insulted, addressed);
      if (controller.signal.aborted) return; // interrupted — nothing to say
      console.log(`[say] ${reply}`);
      state.lastWasQuestion = /[?？]\s*$/.test(reply);
      await this.playText(state, reply);
    } catch (error) {
      console.error('[llm]', error);
    } finally {
      if (state.interrupt === controller) state.interrupt = null;
      release();
      if (state.turn === turn) state.turn = null;
    }
  }

  /** Synthesize and play text through the voice connection. */
  async playText(state, text) {
    let wav;
    try {
      wav = synthesize(text);
    } catch (error) {
      console.error('[tts]', error);
      return;
    }
    if (!wav) return;

    const resource = createAudioResource(Readable.from([wav]), {
      inputType: StreamType.Arbitrary, // ffmpeg transcodes the WAV to Opus
    });
    state.player.play(resource);
    state.lastSpokeAt = Date.now(); // conversation is "live" from here on

    // Resolve when playback finishes.
    await new Promise((resolve) => {
      const onState = (_old, newState) => {
        if (newState.status === AudioPlayerStatus.Idle) {
          state.player.off('stateChange', onState);
          resolve();
        }
      };
      state.player.on('stateChange', onState);
      setTimeout(resolve, 120_000); // safety net
    });
  }

  #cleanup(state) {
    state.capturing.clear();
    this.guilds.delete(state.guildId);
  }

  async leave(guildId) {
    const state = this.guilds.get(guildId);
    if (!state) return false;
    this.#cleanup(state);
    state.player.stop();
    state.connection.destroy();
    return true;
  }
}
