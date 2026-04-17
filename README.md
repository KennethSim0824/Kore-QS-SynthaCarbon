# AI Disclosure: This project was developed with the assistance of Google Gemini and Claude. AI was utilized to assist in structuring the YOLOv11 detection loop, debugging the Dialogflow CX integration, and optimizing the backend server architecture. All AI-assisted code has been manually reviewed, tested, and refined by the team.



<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/d8b50c5c-e6c0-46e7-973d-bad8bd69d529

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`


# SynthaCarbon AI

> Smart Mobility & Green Horizon Platform — real-time construction site vehicle monitoring and CO₂ emission auditing powered by on-device YOLO inference and Google AI.

![Phase 02](https://img.shields.io/badge/Phase-02-blue) ![Audit Active](https://img.shields.io/badge/Audit-Active-green) ![License](https://img.shields.io/badge/License-Private-red)

---

## Overview

SynthaCarbon AI is a browser-based infrastructure monitoring platform designed for construction sites. It ingests video or image footage, runs on-device vehicle detection using a custom YOLOv8/YOLOv11 ONNX model, computes real-time CO₂ emission estimates, and dispatches AI-generated intervention commands when heavy diesel traffic exceeds regulatory thresholds.

The system is built around three pillars:

- **Local ONNX inference** — vehicle detection runs entirely in the browser via ONNX Runtime Web (WebAssembly), with no data leaving the device
- **AI Dispatch Commander** — a Gemini-powered or Dialogflow CX-powered agent that generates authoritative situation reports citing Malaysian environmental legislation
- **Eco-Path Corridor Map** — a live visual of the R&R Skudai corridor that reacts to detected vehicle counts and triggers traffic signal changes

---

## Features

- Upload video (`.mp4`) or image (`.png`, `.webp`, `.jpg`) site footage
- Real-time bounding box overlay on detected vehicles with class label and confidence score
- Per-frame inference during video playback — boxes update as the video plays
- CO₂ emission estimation based on vehicle class emission factors
- Live emission analytics chart that grows with video duration
- Heavy diesel threshold monitoring — triggers GREEN LIGHT grid intervention protocol above 3 heavy units
- AI Dispatch Center with Gemini fallback if Dialogflow is unavailable
- Animated Eco-Path Corridor Map with tactical traffic lights
- Toggle between Local Model (ONNX) and Gemini Vision AI detection modes

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite 6 |
| Styling | Tailwind CSS v4 |
| Animation | Motion (Framer Motion) |
| Charts | Recharts |
| Icons | Lucide React |
| On-device inference | ONNX Runtime Web (WASM) |
| AI Vision fallback | Google Gemini (`gemini-3-flash-preview`) |
| AI Dispatch | Google Dialogflow CX |
| Backend | Express + tsx |
| Model training | YOLOv8/YOLOv11 → exported to ONNX |

---

## Project Structure

```
synthacarbon-ai/
├── public/
│   └── best.onnx              # Trained ONNX model (place here for Vite to serve)
├── src/
│   ├── App.tsx                # Main UI — upload, detection, metrics, map
│   ├── main.tsx               # React entry point
│   ├── index.css              # Global styles
│   ├── types.ts               # Shared types, emission factors, vehicle constants
│   └── lib/
│       ├── yolo.ts            # ONNX inference, NMS, video detection loop
│       ├── gemini.ts          # Gemini Vision + Commander response
│       └── utils.ts           # Tailwind cn() utility
├── server.ts                  # Express server + Dialogflow CX dispatch endpoint
├── best.pt                    # PyTorch model weights (for re-export only)
├── .env                       # Environment variables (never commit)
├── .env.example               # Example env file
├── .gitignore
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## Prerequisites

- Node.js 18+
- A Google Cloud project with Dialogflow CX enabled (optional — Gemini fallback is used if unavailable)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com)
- Python 3.8+ with `ultralytics` installed (only needed to re-export the ONNX model)

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd synthacarbon-ai
npm install
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# Required — Gemini API key for Vision AI and Commander fallback
GEMINI_API_KEY=your_gemini_api_key_here

# Optional — Dialogflow CX (if not set, Gemini Commander is used)
DIALOGFLOW_PROJECT_ID=your-gcp-project-id
DIALOGFLOW_LOCATION=global
DIALOGFLOW_AGENT_ID=your-dialogflow-agent-id
```

### 3. Add the ONNX model

The ONNX model must be placed at `public/best.onnx` so Vite can serve it at `/best.onnx`.

If you need to re-export from the included `best.pt` weights:

```bash
pip install ultralytics
python3 -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=640, opset=12, simplify=True)
"
mv best.onnx public/best.onnx
```

### 4. (Optional) Dialogflow CX service account

If using Dialogflow CX, place your Google Cloud service account JSON at the project root:

```
synthacarbon-ai/service-account-key.json
```

> ⚠️ This file is listed in `.gitignore` and must never be committed to version control.

### 5. Run the development server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

---

## Usage

1. Open the app in your browser
2. Select detection mode — **Local Model (best.onnx)** runs entirely in-browser; **Gemini Vision AI** sends frames to the Gemini API
3. Click the upload zone and select a video or image file from a construction site
4. The system will:
   - Display the footage in the Perception Preview panel
   - Overlay bounding boxes with class labels and confidence scores
   - Update the Emission Analytics chart in real time as the video plays
   - Count heavy diesel units and trigger the GREEN LIGHT intervention protocol if the count exceeds 3
   - Generate an AI Dispatch Commander report in the right panel
   - Animate the Eco-Path Corridor Map and traffic signals based on the intervention status

---

## Vehicle Classes & Emission Factors

The model detects four vehicle classes. Emission factors are used to estimate CO₂ intensity:

| Class | Emission Factor |
|---|---|
| Truck | 15,000 g/hr |
| Excavator | 12,000 g/hr |
| Tractor | 9,500 g/hr |
| Crane | 5,500 g/hr |

All four classes are treated as heavy diesel for the intervention threshold calculation.

---

## Detection Pipeline

```
Upload / Video frame
        │
        ▼
  extractFrameFromVideo()        ← for video uploads: seeks to t=1s
        │
        ▼
  preprocessSource()             ← resize to 640×640, RGB planar Float32
        │
        ▼
  ONNX Runtime (WASM)            ← runs best.onnx in-browser
        │
        ▼
  parseOutput()                  ← decode [1, 8, N] tensor → raw detections
        │
        ▼
  applyNMS()                     ← Non-Maximum Suppression (IoU threshold 0.45)
        │
        ▼
  VehicleDetection[]             ← bbox as [left%, top%, width%, height%]
```

For live video, `startVideoDetection()` runs the full pipeline on a `requestAnimationFrame` loop, triggering inference every 10 seconds and rendering bounding boxes on an overlay `<canvas>` at every frame.

---

## API Endpoint

The Express backend exposes one route used by the AI Dispatch Center:

### `POST /api/dispatch`

Sends a status text to Dialogflow CX and returns the agent's response.

**Request body:**
```json
{
  "text": "Status Update: 4 heavy units detected. Grid Intervention required.",
  "sessionId": "user-session"
}
```

**Response:**
```json
{
  "message": "COMMAND ACKNOWLEDGED. Grid Intervention is active..."
}
```

If Dialogflow CX is unavailable or returns an error, the frontend automatically falls back to the Gemini Commander (`getCommanderResponse()` in `gemini.ts`).

---

## Legislation References

The AI Dispatch Commander cites the following Malaysian regulations in its situation reports:

- **Section 48, Road Transport Act 1987** — prohibition on grid obstruction
- **Section 22, Environmental Quality Act 1974** — mandatory emission control

The GREEN LIGHT protocol is justified by the **Restart Penalty** — stationary heavy diesel engines generate up to 400% higher CO₂ emissions when restarting from a full stop compared to continuous low-speed flow.

---

## Scripts

| Script | Description |
|---|---|
| `npm run dev` | Start development server (Express + Vite middleware) |
| `npm run build` | Build frontend for production |
| `npm run preview` | Preview production build |
| `npm run lint` | TypeScript type check |
| `npm run clean` | Remove `dist/` folder |

---

## Environment Notes

- **Google AI Studio** — HMR WebSocket errors (`WebSocket closed without opened`) are expected and harmless. The app disables HMR automatically via the `DISABLE_HMR` environment variable.
- **ONNX model path** — `best.onnx` must be inside `public/` for Vite to serve it correctly. Placing it in the project root will result in a 404 error at runtime.
- **Gemini quota** — the free tier has daily request limits. If you hit the quota, either wait for the daily reset, enable billing on your Google Cloud project, or switch to Local Model mode.

---

## Security

- Never commit `.env`, `service-account-key.json`, or any API keys
- The `.gitignore` excludes both files by default
- The Gemini API key is injected at build time via Vite's `define` — it will be visible in the browser bundle, so use a restricted key scoped to your domain in production

---

## License

Private — all rights reserved.
