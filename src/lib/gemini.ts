import { GoogleGenAI, Type } from "@google/genai";
import { VehicleClass, VehicleDetection } from "@/src/types";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

export async function analyzeFootage(base64Image: string, mimeType: string) {
  const model = "gemini-3-flash-preview";
  
  const prompt = `Analyze this image from a construction site. 
Identify all heavy vehicles: excavators, tractors, trucks, and cranes. 
Return the result as a JSON array of objects, each with 'class' (one of: excavator, tractor, truck, crane) and 'confidence' (0-1).`;

  const response = await ai.models.generateContent({
    model,
    contents: [
      {
        parts: [
          { inlineData: { data: base64Image, mimeType } },
          { text: prompt }
        ]
      }
    ],
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            class: { type: Type.STRING, enum: ['excavator', 'tractor', 'truck', 'crane'] },
            confidence: { type: Type.NUMBER }
          },
          required: ['class', 'confidence']
        }
      }
    }
  });

  return JSON.parse(response.text || '[]') as VehicleDetection[];
}

export async function getCommanderResponse(
  vehicleCount: number, 
  isInterventionRequired: boolean
) {
  const model = "gemini-3-flash-preview";
  
  const prompt = `URGENT INFRASTRUCTURE COMMAND UPDATE.
Scan complete. Result: ${vehicleCount} heavy diesel units detected.
Mode: ${isInterventionRequired ? 'GREEN LIGHT PROTOCOL INITIATED' : 'STANDARD MONITORING'}.

Instructions for Agent:
1. Speak as a Senior Site Commander. Confirm the grid intervention status.
2. Cite Section 48 of the Road Transport Act 1987 (Obstruction) and Section 22 of the Environmental Quality Act 1974 (Emission Control).
3. Explain that commanding a GREEN LIGHT prevents the 'Restart Penalty' carbon spike (up to 400% higher emissions when starting from a stop).
4. Direct standard vehicles to R&R Skudai via the Eco-Path to clear the lane for heavy units if intervention is active.
5. Keep the response concise and authoritative.`;

  const response = await ai.models.generateContent({
    model,
    contents: prompt
  });

  return response.text || "Connection to command center lost.";
}
