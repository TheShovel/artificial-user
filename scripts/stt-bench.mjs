#!/usr/bin/env node
/**
 * Benchmarks STT candidates: speed + accuracy on the fox phrase + wake words.
 * Usage: node scripts/stt-bench.mjs
 */
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
const require = createRequire(import.meta.url);
const OpusScript = require('opusscript');
const { OpusEncoder } = require('@discordjs/opus');
import { synthesize } from '../src/tts.js';
import { pipeline } from '@huggingface/transformers';

function wavToPcm16(wav) {
  return new Promise((resolve) => {
    const pcm = wav.subarray(44);
    const to48 = spawn('ffmpeg', ['-loglevel','error','-f','s16le','-ar','22050','-ac','1','-i','pipe:0','-f','s16le','-ar','48000','-ac','2','pipe:1']);
    const encoder = new OpusEncoder(48000, 2, 'audio');
    const opusFrames = [];
    to48.stdout.on('data', (c) => { for (let i = 0; i + 3840 <= c.length; i += 3840) opusFrames.push(encoder.encode(c.subarray(i, i + 3840))); });
    to48.on('close', () => {
      const opus = new OpusScript(48000, 2);
      const resampler = spawn('ffmpeg', ['-loglevel','error','-f','s16le','-ar','48000','-ac','2','-i','pipe:0','-f','s16le','-ar','16000','-ac','1','pipe:1']);
      const chunks = [];
      resampler.stdout.on('data', (c) => chunks.push(c));
      for (const f of opusFrames) { const d = opus.decode(f); if (d) resampler.stdin.write(d); }
      resampler.stdin.end();
      resampler.on('close', () => resolve(Buffer.concat(chunks)));
    });
    to48.stdin.on('error', () => {});
    to48.stdin.end(pcm);
  });
}

const phrases = {
  fox: 'the quick brown fox jumps over the lazy dog',
  wake1: 'hey bobby what do you think',
  wake2: 'bobby tell us a joke',
};

const models = process.argv[2]?.split(',') ?? [
  'onnx-community/whisper-small:q8',
  'onnx-community/whisper-small:q4',
  'onnx-community/whisper-base:q8',
  'onnx-community/whisper-tiny.en:q8',
];

const audio = {};
for (const [name, phrase] of Object.entries(phrases)) audio[name] = await wavToPcm16(synthesize(phrase));

for (const spec of models) {
  const [id, dtype] = spec.split(':');
  const t0 = Date.now();
  const transcriber = await pipeline('automatic-speech-recognition', id, { dtype });
  console.log(`\n=== ${id} (${dtype}) — load ${Date.now() - t0}ms ===`);
  for (const [name, pcm] of Object.entries(audio)) {
    const t = Date.now();
    const lang = id.includes('tiny.en') ? undefined : 'en';
    const out = await transcriber(audioToFloat32(pcm), { chunk_length_s: 30, stride_length_s: 5, ...(lang ? { language: lang } : {}) });
    console.log(`  ${name.padEnd(6)} ${String(Date.now() - t).padStart(5)}ms | ${JSON.stringify(out.text)}`);
  }
}

function audioToFloat32(buf) {
  const f = new Float32Array(buf.length / 2);
  for (let i = 0; i < f.length; i++) f[i] = buf.readInt16LE(i * 2) / 32768;
  return f;
}

process.exit(0);
