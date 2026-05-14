import * as ort from 'onnxruntime-web';
import { VehicleDetection } from '../types';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

let session: ort.InferenceSession | null = null;

const MODEL_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.35;
const IOU_THRESHOLD = 0.45;
const DETECT_INTERVAL_MS = 500;

// Must match: {0:'crane',1:'excavator',2:'tractor',3:'truck'}
const classes = ['crane', 'excavator', 'tractor', 'truck'];

async function ensureSession(): Promise<ort.InferenceSession> {
  if (!session) {
    session = await ort.InferenceSession.create('/best.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    console.log('[YOLO] Loaded /best.onnx', session.inputNames, session.outputNames);
  }
  return session;
}

function preprocessSource(source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement): ort.Tensor {
  const canvas = document.createElement('canvas');
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;

  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, MODEL_SIZE, MODEL_SIZE);

  const imageData = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const input = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);

  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
    input[i] = imageData.data[i * 4] / 255;
    input[i + MODEL_SIZE * MODEL_SIZE] = imageData.data[i * 4 + 1] / 255;
    input[i + 2 * MODEL_SIZE * MODEL_SIZE] = imageData.data[i * 4 + 2] / 255;
  }

  return new ort.Tensor('float32', input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function iou(a: number[], b: number[]): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union <= 0 ? 0 : inter / union;
}

function applyNMS(detections: VehicleDetection[]): VehicleDetection[] {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: VehicleDetection[] = [];

  for (const det of sorted) {
    if (!kept.some((k) => iou(det.bbox, k.bbox) > IOU_THRESHOLD)) {
      kept.push(det);
    }
  }

  return kept;
}

function parseOutput(output: ort.Tensor): VehicleDetection[] {
  const data = output.data as Float32Array;
  const dims = output.dims;

  const raw: VehicleDetection[] = [];

  const channelsFirst = dims.length === 3 && dims[1] <= 20;
  const numDetections = channelsFirst ? dims[2] : dims[1];
  const numValues = channelsFirst ? dims[1] : dims[2];

  for (let i = 0; i < numDetections; i++) {
    const get = (channel: number) => {
      return channelsFirst
        ? data[channel * numDetections + i]
        : data[i * numValues + channel];
    };

    let maxScore = 0;
    let classId = -1;

    for (let c = 0; c < classes.length; c++) {
      const score = get(4 + c);
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    if (maxScore < CONFIDENCE_THRESHOLD || classId === -1) continue;

    const cx = get(0);
    const cy = get(1);
    const w = get(2);
    const h = get(3);

    raw.push({
      class: classes[classId] as VehicleDetection['class'],
      confidence: maxScore,
      bbox: [
        ((cx - w / 2) / MODEL_SIZE) * 100,
        ((cy - h / 2) / MODEL_SIZE) * 100,
        (w / MODEL_SIZE) * 100,
        (h / MODEL_SIZE) * 100,
      ],
    });
  }

  return applyNMS(raw);
}

function drawDetections(
  canvas: HTMLCanvasElement,
  detections: VehicleDetection[],
  displayW: number,
  displayH: number
): void {
  canvas.width = displayW;
  canvas.height = displayH;

  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, displayW, displayH);

  for (const det of detections) {
    const x = (det.bbox[0] / 100) * displayW;
    const y = (det.bbox[1] / 100) * displayH;
    const w = (det.bbox[2] / 100) * displayW;
    const h = (det.bbox[3] / 100) * displayH;
    const label = `${det.class.toUpperCase()} ${Math.round(det.confidence * 100)}%`;

    ctx.strokeStyle = '#00E676';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.font = 'bold 14px monospace';
    const textW = ctx.measureText(label).width + 10;
    ctx.fillStyle = '#00E676';
    ctx.fillRect(x, Math.max(0, y - 22), textW, 22);

    ctx.fillStyle = '#000000';
    ctx.fillText(label, x + 5, Math.max(14, y - 6));
  }
}

export async function detectWithYOLO(
  base64Image: string,
  mimeType = 'image/jpeg'
): Promise<VehicleDetection[]> {
  try {
    if (mimeType.startsWith('video/')) {
      base64Image = await extractFrameFromVideo(base64Image, mimeType);
      mimeType = 'image/jpeg';
    }

    const sess = await ensureSession();

    const image = new Image();
    image.src = `data:${mimeType};base64,${base64Image}`;
    await new Promise((resolve) => {
      image.onload = resolve;
    });

    const tensor = preprocessSource(image);
    const outputs = await sess.run({ [sess.inputNames[0]]: tensor });
    return parseOutput(outputs[sess.outputNames[0]]);
  } catch (error) {
    console.error('[YOLO] Inference error:', error);
    return [];
  }
}

async function extractFrameFromVideo(base64: string, mimeType: string): Promise<string> {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.src = `data:${mimeType};base64,${base64}`;
    video.muted = true;
    video.currentTime = 1;

    video.addEventListener(
      'seeked',
      () => {
        const canvas = document.createElement('canvas');
        canvas.width = MODEL_SIZE;
        canvas.height = MODEL_SIZE;
        canvas.getContext('2d')!.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);
        resolve(canvas.toDataURL('image/jpeg').split(',')[1]);
      },
      { once: true }
    );

    video.load();
  });
}

export function startVideoDetection(
  videoEl: HTMLVideoElement,
  overlayCanvas: HTMLCanvasElement,
  onDetect?: (detections: VehicleDetection[]) => void
): () => void {
  let active = true;
  let rafId = 0;
  let lastInferTime = 0;
  let inferRunning = false;
  let lastDetections: VehicleDetection[] = [];

  async function loop(ts: number) {
    if (!active) return;

    const isReady = videoEl.readyState >= 2;
    const isPlaying = isReady && !videoEl.paused && !videoEl.ended;

    if (isPlaying && !inferRunning && ts - lastInferTime >= DETECT_INTERVAL_MS) {
      inferRunning = true;
      lastInferTime = ts;

      try {
        const sess = await ensureSession();
        const tensor = preprocessSource(videoEl);
        const outputs = await sess.run({ [sess.inputNames[0]]: tensor });
        lastDetections = parseOutput(outputs[sess.outputNames[0]]);
        onDetect?.(lastDetections);
      } catch (error) {
        console.error('[YOLO] Camera/video inference error:', error);
      } finally {
        inferRunning = false;
      }
    }

    if (videoEl.videoWidth && videoEl.videoHeight) {
      drawDetections(overlayCanvas, lastDetections, videoEl.videoWidth, videoEl.videoHeight);
    }

    rafId = requestAnimationFrame(loop);
  }

  rafId = requestAnimationFrame(loop);

  return () => {
    active = false;
    cancelAnimationFrame(rafId);
    const ctx = overlayCanvas.getContext('2d');
    ctx?.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  };
}
