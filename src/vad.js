import fs from 'node:fs';
import path from 'node:path';

const MODEL_URL = 'https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx';
const MODEL_PATH = path.join(process.cwd(), 'models', 'silero-vad.onnx');

/** Silero VAD is a tiny (~2 MB) model; download once and cache it locally. */
export async function ensureVadModel() {
  if (fs.existsSync(MODEL_PATH) && fs.statSync(MODEL_PATH).size > 100_000) {
    return MODEL_PATH;
  }
  console.log('[stt-worker] downloading Silero VAD model (~2 MB)...');
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`Failed to download VAD model (HTTP ${response.status})`);
  fs.mkdirSync(path.dirname(MODEL_PATH), { recursive: true });
  const buffer = Buffer.from(await response.arrayBuffer());
  // Write to a temp file and rename so two workers starting at the same time
  // (one per STT pool) can never corrupt the model by writing over each other.
  const tmp = `${MODEL_PATH}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, buffer);
  try {
    fs.renameSync(tmp, MODEL_PATH);
  } catch {
    fs.rmSync(tmp, { force: true }); // another worker won the race; ours is identical
  }
  return MODEL_PATH;
}
