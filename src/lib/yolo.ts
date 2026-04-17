import * as ort from 'onnxruntime-web';
import { VehicleDetection } from '../types';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

let session: ort.InferenceSession | null = null;
const MODEL_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.45;
const classes = ['excavator', 'tractor', 'truck', 'crane'];

// ─── SESSION LOADER ────────────────────────────────────────────────────────
async function ensureSession(): Promise<ort.InferenceSession> {
  if (!session) {
    session = await ort.InferenceSession.create('/best.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    console.log('[YOLO] Model loaded. Inputs:', session.inputNames, 'Outputs:', session.outputNames);
  }
  return session;
}

// ─── PREPROCESS ────────────────────────────────────────────────────────────
// Reads directly from a <video> or <canvas> element — no base64 round-trip
function preprocessSource(source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement): ort.Tensor {
  const canvas = document.createElement('canvas');
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, MODEL_SIZE, MODEL_SIZE);

  const imageData = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const input = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
    input[i]                               = imageData.data[i * 4]     / 255; // R
    input[i + MODEL_SIZE * MODEL_SIZE]     = imageData.data[i * 4 + 1] / 255; // G
    input[i + 2 * MODEL_SIZE * MODEL_SIZE] = imageData.data[i * 4 + 2] / 255; // B
  }
  return new ort.Tensor('float32', input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

// ─── NMS ──────────────────────────────────────────────────────────────────
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
    const suppressed = kept.some(k => iou(det.bbox, k.bbox) > IOU_THRESHOLD);
    if (!suppressed) kept.push(det);
  }
  return kept;
}

// ─── PARSE OUTPUT ──────────────────────────────────────────────────────────
function parseOutput(output: ort.Tensor): VehicleDetection[] {
  const data = output.data as Float32Array;
  const numDetections = output.dims[2]; // [1, 8, numDetections]
  const raw: VehicleDetection[] = [];

  for (let i = 0; i < numDetections; i++) {
    let maxScore = 0;
    let classId = -1;

    for (let c = 0; c < classes.length; c++) {
      const score = data[(4 + c) * numDetections + i];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    if (maxScore < CONFIDENCE_THRESHOLD || classId === -1) continue;

    const cx = data[0 * numDetections + i];
    const cy = data[1 * numDetections + i];
    const w  = data[2 * numDetections + i];
    const h  = data[3 * numDetections + i];

    raw.push({
      class: classes[classId] as VehicleDetection['class'],
      confidence: maxScore,
      bbox: [
        ((cx - w / 2) / MODEL_SIZE) * 100,
        ((cy - h / 2) / MODEL_SIZE) * 100,
        (w  / MODEL_SIZE) * 100,
        (h  / MODEL_SIZE) * 100,
      ],
    });
  }

  return applyNMS(raw);
}

// ─── DRAW OVERLAY ──────────────────────────────────────────────────────────
function drawDetections(
  canvas: HTMLCanvasElement,
  detections: VehicleDetection[],
  displayW: number,
  displayH: number
): void {
  canvas.width  = displayW;
  canvas.height = displayH;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, displayW, displayH);

  for (const det of detections) {
    const x = (det.bbox[0] / 100) * displayW;
    const y = (det.bbox[1] / 100) * displayH;
    const w = (det.bbox[2] / 100) * displayW;
    const h = (det.bbox[3] / 100) * displayH;
    const label = `${det.class.toUpperCase()} ${Math.round(det.confidence * 100)}%`;

    // Bounding box
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth   = 2;
    ctx.strokeRect(x, y, w, h);

    // Label background
    ctx.font = 'bold 14px monospace';
    const textW = ctx.measureText(label).width + 10;
    ctx.fillStyle = '#00FF00';
    ctx.fillRect(x, y - 22, textW, 22);

    // Label text
    ctx.fillStyle = '#000000';
    ctx.fillText(det.class.toUpperCase(), x + 5, y - 6);
  }
}

// ─── ONE-SHOT: STILL IMAGE (backward compatible) ───────────────────────────
export async function detectWithYOLO(
  base64Image: string,
  mimeType = 'image/jpeg'
): Promise<VehicleDetection[]> {
  try {
    // For video uploads: extract a single frame at t=1s (kept for compat)
    if (mimeType.startsWith('video/')) {
      base64Image = await extractFrameFromVideo(base64Image, mimeType);
      mimeType = 'image/jpeg';
    }

    const sess = await ensureSession();

    const image = new Image();
    image.src = `data:${mimeType};base64,${base64Image}`;
    await new Promise((resolve) => (image.onload = resolve));

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
    video.addEventListener('seeked', () => {
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 640;
      canvas.getContext('2d')!.drawImage(video, 0, 0, 640, 640);
      resolve(canvas.toDataURL('image/jpeg').split(',')[1]);
    }, { once: true });
    video.load();
  });
}

// ─── REAL-TIME VIDEO DETECTION LOOP ───────────────────────────────────────
//
// FIX 1 — Boxes don't follow vehicles:
//   The old code only ever read 1 frame. This loop reads a NEW frame every
//   150ms directly from the live <video> element.
//
// FIX 2 — Boxes freeze when video restarts/loops:
//   We guard readyState + paused + ended before every inference call, and
//   redraw cached detections on every requestAnimationFrame tick so the
//   overlay never goes stale.
//
// Usage in your component:
//
//   const stopRef = useRef<() => void>();
//
//   useEffect(() => {
//     stopRef.current = startVideoDetection(videoRef.current!, canvasRef.current!);
//     return () => stopRef.current?.();
//   }, []);
//
// Your JSX — overlay the canvas directly on top of the video:
//
//   <div style={{ position: 'relative' }}>
//     <video ref={videoRef} ... style={{ display: 'block' }} />
//     <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }} />
//   </div>

const DETECT_INTERVAL_MS = 10000; // Increased slightly for stability on laptops

export function startVideoDetection(
  videoEl: HTMLVideoElement,
  overlayCanvas: HTMLCanvasElement,
  onDetect?: (detections: VehicleDetection[]) => void
): () => void {
  let active = true;
  let rafId: number;
  let lastInferTime = 0;
  let inferRunning = false;
  let lastDetections: VehicleDetection[] = [];

  async function loop(ts: number) {
    if (!active) return;

    const isPlaying = videoEl.readyState >= 2 && !videoEl.paused && !videoEl.ended;
    const hasEnded = videoEl.ended;

    // 1. INFERENCE TRIGGER
    // We run if playing OR if it just ended (to get a final clean scan)
    if ((isPlaying || hasEnded) && !inferRunning && ts - lastInferTime >= DETECT_INTERVAL_MS) {
      
      // If video ended and we already have detections for the final frame, stop inferencing
      if (hasEnded && lastInferTime > 0) {
         // Stop the loop but keep drawing the last known boxes
         drawDetections(overlayCanvas, lastDetections, videoEl.videoWidth, videoEl.videoHeight); 
         return; 
      }

      inferRunning = true;
      lastInferTime = ts;

      try {
        const sess = await ensureSession();
        // Faster than Base64: Passing the video element directly to the tensor logic
        const tensor = preprocessSource(videoEl); 
        const outputs = await sess.run({ [sess.inputNames[0]]: tensor });
        lastDetections = parseOutput(outputs[sess.outputNames[0]]);
        if (onDetect) onDetect(lastDetections);
      } catch (e) {
        console.error('[YOLO] Inference error:', e);
      } finally {
        inferRunning = false;
      }
    }

    // 2. SMOOTH RENDERING
    drawDetections(
        overlayCanvas,
        lastDetections,
        videoEl.videoWidth,
        videoEl.videoHeight
      );

    // Only keep the animation loop running if the video is active
    if (!hasEnded) {
      rafId = requestAnimationFrame(loop);
    }
  }

  rafId = requestAnimationFrame(loop);

  return () => {
    active = false;
    cancelAnimationFrame(rafId);
  };
}