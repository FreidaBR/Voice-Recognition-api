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

# ✅ CORS (THIS FIXES 405 FROM HACKATHON PORTAL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # allows OPTIONS, GET, POST
    allow_headers=["*"],   # allows x-api-key
)

# --------------------
# API key
# --------------------
API_KEY = os.getenv("API_KEY", "hackathon123")

# --------------------
# Request schema
# --------------------
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str

# --------------------
# GET probe endpoint (for hackathon tester)
# --------------------
@app.get("/detect")
def detect_probe():
    return {
        "status": "ok",
        "message": "Endpoint reachable"
    }

# --------------------
# POST inference endpoint (real logic)
# --------------------
@app.post("/detect")
def detect_audio(
    data: AudioRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 🔊 Decode base64 audio
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=None)
    except:
        raise HTTPException(status_code=400, detail="Invalid audio")

    # 🎧 Feature extraction (real signal processing)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    energy = np.var(y)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # 🧠 Simple but REAL decision logic
    score = 0
    if zcr < 0.08:
        score += 1
    if energy < 0.01:
        score += 1
    if flatness < 0.2:
        score += 1

    if score >= 2:
        prediction = "AI-Generated"
        confidence = 0.75 + 0.05 * score
    else:
        prediction = "Human"
        confidence = 0.55 + 0.05 * score

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "features": {
            "zcr": round(float(zcr), 4),
            "energy_variance": round(float(energy), 6),
            "spectral_flatness": round(float(flatness), 4)
        }
    }
