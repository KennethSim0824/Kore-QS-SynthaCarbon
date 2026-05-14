import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { SessionsClient } from "@google-cloud/dialogflow-cx";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 8080;  // ← changed

app.use(express.json());

// --- 1. CONFIGURATION ---
const projectId = process.env.DIALOGFLOW_PROJECT_ID || "kore-qs-hackathon";
const location = process.env.DIALOGFLOW_LOCATION || "global";
const agentId = process.env.DIALOGFLOW_AGENT_ID || "caed5dd0-759b-4eea-9a7f-7f4fc9ce5eab";

// --- 2. INITIALIZE CLIENT ---
const client = new SessionsClient(
  process.env.NODE_ENV === "production"
    ? {}  // ← Cloud Run uses built-in IAM, no key file needed
    : { keyFilename: "./service-account-key.json" }
);

// --- 3. API ROUTES ---
app.post("/api/dispatch", async (req, res) => {
  const { text, sessionId } = req.body;
  if (!text) return res.status(400).json({ error: "Text is required" });

  try {
    const sessionPath = client.projectLocationAgentSessionPath(
      projectId.trim(),
      location.trim(),
      agentId.trim(),
      sessionId || "default-session"
    );

    const request = {
      session: sessionPath,
      queryInput: {
        text: { text },
        languageCode: "en",
      },
    };

    const [response] = await client.detectIntent(request);
    
    const responseMessages = response.queryResult?.responseMessages;
    const resultText = responseMessages?.[0]?.text?.text?.[0];
    
    if (resultText) {
      console.log("✅ Dispatch Response:", resultText);
      return res.json({ message: resultText });
    }
    
    throw new Error("Empty Dialogflow response");

  } catch (error) {
    console.error("❌ Dialogflow dispatch failed:", (error as Error).message);
    res.status(503).json({ 
      error: "Dialogflow unavailable", 
      details: (error as Error).message 
    });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`\n🚀 GRID COMMANDER ACTIVE`);
    console.log(`🔗 Project: ${projectId}`);
    console.log(`📍 Endpoint: http://localhost:${PORT}\n`);
  });
}

startServer();