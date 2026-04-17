# ─────────────────────────────────────────────
# Stage 1: Build the Vite/React frontend
# ─────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies first (layer-cached until package files change)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build


# ─────────────────────────────────────────────
# Stage 2: Production runtime
# ─────────────────────────────────────────────
FROM node:22-alpine AS runtime

WORKDIR /app

ENV NODE_ENV=production

# Install only production dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy compiled frontend from the builder stage
COPY --from=builder /app/dist ./dist

# Copy server source (tsx compiles it at start-up)
COPY server.ts ./

# Copy ONNX model (served as a static asset)
COPY public/ ./public/

# service-account-key.json is sensitive — mount it at runtime via a
# Docker secret or volume instead of baking it into the image:
#   docker run -v /host/path/service-account-key.json:/app/service-account-key.json ...
# Un-comment the next line only for local testing (not for production):
# COPY service-account-key.json ./

EXPOSE 3000

# Pass env vars at runtime:
#   docker run --env-file .env ...
# Required variables:
#   GEMINI_API_KEY
#   DIALOGFLOW_PROJECT_ID
#   DIALOGFLOW_LOCATION
#   DIALOGFLOW_AGENT_ID
#   APP_URL

CMD ["npx", "tsx", "server.ts"]
