import {
  Client,
  Events,
  GatewayIntentBits,
  InteractionType,
  MessageFlags,
  SlashCommandBuilder,
} from 'discord.js';
import { config } from './config.js';
import { checkOllama } from './llm.js';
import { memory } from './memory.js';
import { initStt } from './stt.js';
import { VoiceBot } from './voice.js';

// Memory writes are debounced; flush on shutdown so nothing is lost.
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    memory.flush();
    process.exit(0);
  });
}

const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
});

const voiceBot = new VoiceBot(client);

const commands = [
  new SlashCommandBuilder()
    .setName('join')
    .setDescription('Join your voice channel and start listening'),
  new SlashCommandBuilder()
    .setName('leave')
    .setDescription('Leave the voice channel and stop listening'),
  new SlashCommandBuilder()
    .setName('forget')
    .setDescription('Clear my memory of this channel\'s conversation'),
  new SlashCommandBuilder()
    .setName('say')
    .setDescription('Make the bot say something out loud')
    .addStringOption((option) =>
      option.setName('text').setDescription('What to say').setRequired(true),
    ),
  new SlashCommandBuilder()
    .setName('status')
    .setDescription('Show connection and memory status'),
];

client.once(Events.ClientReady, async () => {
  console.log(`[bot] logged in as ${client.user.tag}`);

  try {
    await client.application.commands.set(commands);
    console.log('[bot] slash commands registered');
  } catch (error) {
    console.error('[bot] failed to register commands:', error.message);
  }

  try {
    await checkOllama();
    console.log(`[llm] Ollama ready at ${config.ollamaUrl} (model: ${config.llmModel})`);
  } catch (error) {
    console.warn(`[llm] ${error.message} — replies will fail until this is fixed`);
  }
});

client.on('interactionCreate', async (interaction) => {
  if (interaction.type !== InteractionType.ApplicationCommand) return;
  const { commandName, guild } = interaction;
  if (!guild) {
    return interaction.reply({ content: 'This bot only works inside a server.', flags: MessageFlags.Ephemeral });
  }

  if (commandName === 'join') {
    const member = guild.members.cache.get(interaction.user.id);
    const channel = member?.voice?.channel;
    if (!channel) {
      return interaction.reply({ content: 'Join a voice channel first, then use /join.', flags: MessageFlags.Ephemeral });
    }
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });
    try {
      const joined = await voiceBot.join(channel);
      await interaction.editReply(
        joined
          ? `Joined **${channel.name}** and I'm listening. Say something!`
          : `I'm already in a voice channel here.`,
      );
    } catch (error) {
      console.error('[join]', error);
      await interaction.editReply(`Could not join: ${error.message}`);
    }
  } else if (commandName === 'leave') {
    const left = await voiceBot.leave(guild.id);
    await interaction.reply({ content: left ? 'Left the voice channel.' : "I'm not in a voice channel.", flags: MessageFlags.Ephemeral });
  } else if (commandName === 'forget') {
    memory.clear(interaction.guild.id);
    console.log(`[memory] cleared conversation for guild ${interaction.guild.id}`);
    await interaction.reply({ content: 'Memory cleared.', flags: MessageFlags.Ephemeral });
  } else if (commandName === 'say') {
    const state = voiceBot.get(guild.id);
    if (!state) {
      return interaction.reply({ content: 'I\'m not in a voice channel. Use /join first.', flags: MessageFlags.Ephemeral });
    }
    // Cooldown on /say, per server (one server can't block another).
    const remaining = config.sayCooldownMs - (Date.now() - state.lastSayAt);
    if (remaining > 0) {
      return interaction.reply({
        content: `/say is on cooldown — try again in ${Math.ceil(remaining / 1000)}s.`,
        flags: MessageFlags.Ephemeral,
      });
    }
    state.lastSayAt = Date.now();

    const text = interaction.options.getString('text', true);
    // Store it as the bot's own message so it "remembers" having said it.
    memory.add(guild.id, 'assistant', text);
    await interaction.reply({ content: `Saying: "${text}"`, flags: MessageFlags.Ephemeral });
    await voiceBot.playText(state, text);
  } else if (commandName === 'status') {
    const state = voiceBot.get(guild.id);
    const channelName = state
      ? client.channels.cache.get(state.voiceChannelId)?.name ?? 'unknown'
      : null;
    const historyCount = memory.get(interaction.guild.id).length;
    await interaction.reply({
      content: [
        `**Status**`,
        `- Voice: ${state ? `connected in "${channelName}"` : 'not connected'}`,
        `- Memory: ${historyCount} messages for this server`,
        `- Model: ${config.llmModel}`,
      ].join('\n'),
      flags: MessageFlags.Ephemeral,
    });
  }
});

// Auto-leave when everyone else leaves the channel.
client.on('voiceStateUpdate', (oldState, newState) => {
  if (!config.autoLeaveEmpty) return;
  const state = voiceBot.get(newState.guild.id);
  if (!state) return;

  // Only react when someone joined or left *our* voice channel.
  const affected =
    oldState.channelId === state.voiceChannelId || newState.channelId === state.voiceChannelId;
  if (!affected) return;

  const channel = client.channels.cache.get(state.voiceChannelId);
  const humans = channel?.members?.filter((member) => !member.user.bot).size ?? 0;
  if (humans > 0) return;

  console.log('[bot] voice channel is empty, leaving in 10s...');
  setTimeout(async () => {
    const current = voiceBot.get(newState.guild.id);
    if (!current) return;
    const nowChannel = client.channels.cache.get(current.voiceChannelId);
    if ((nowChannel?.members?.filter((member) => !member.user.bot).size ?? 0) === 0) {
      await voiceBot.leave(newState.guild.id);
    }
  }, 10_000);
});

client.on('error', (error) => console.error('[discord]', error));

if (!config.discordToken) {
  console.error('DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in.');
  process.exit(1);
}

// Load the speech-to-text model at startup (parallel with logging in) so the
// first person who speaks doesn't wait for a model download + load. Runs in a
// worker thread so inference never blocks the bot.
initStt().catch((error) => console.error('[stt] model load failed:', error.message));

client.login(config.discordToken);
