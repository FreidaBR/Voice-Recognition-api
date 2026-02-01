from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import os
import numpy as np
import librosa

# --------------------
# App setup
# --------------------
app = FastAPI()

# CORS (important for hackathon tester)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# API Key (can be overridden by env variable)
# --------------------
API_KEY = os.getenv("API_KEY", "hackathon123")

# --------------------
# Request Schema (matches tester UI)
# --------------------
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str

# --------------------
# Health check endpoint (optional but useful)
# --------------------
@app.get("/detect")
def detect_probe():
    return {
        "status": "ok",
        "message": "Voice detection API is reachable"
    }

# --------------------
# Main detection endpoint
# --------------------
@app.post("/detect")
def detect_audio(
    data: AudioRequest,
    x_api_key: str = Header(None, alias="x-api-key")
):
    # 🔐 API key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 🔊 Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=None)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted audio")

    # 🎧 Feature extraction
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    energy_variance = float(np.var(y))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # 🧠 Simple & explainable decision logic
    score = 0
    if zcr < 0.08:
        score += 1
    if energy_variance < 0.01:
        score += 1
    if spectral_flatness < 0.2:
        score += 1

    if score >= 2:
        classification = "AI"
        confidence = min(0.9, 0.7 + 0.05 * score)
        explanation = (
            "Low pitch variation and uniform spectral characteristics "
            "suggest patterns commonly found in AI-generated speech."
        )
    else:
        classification = "Human"
        confidence = 0.6 + 0.05 * score
        explanation = (
            "Natural fluctuations in pitch and energy indicate human speech."
        )

    # ✅ Final response (EXACTLY what tester expects)
    return {
        "classification": classification,
        "confidence": round(confidence, 2),
        "explanation": explanation
    }
