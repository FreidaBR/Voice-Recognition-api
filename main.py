from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64, io, os
import numpy as np
import librosa

app = FastAPI()

API_KEY = os.getenv("API_KEY", "hackathon123")

class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str
@app.get("/detect")
def detect_probe():
    return {
        "status": "ok",
        "message": "Endpoint reachable"
    }

@app.api_route("/detect", methods=["POST", "GET"])

def detect(data: AudioRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    except:
        raise HTTPException(status_code=400, detail="Invalid audio")

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    energy = np.var(y)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    score = sum([
        zcr < 0.08,
        energy < 0.01,
        flatness < 0.2
    ])

    prediction = "AI-Generated" if score >= 2 else "Human"
    confidence = round(0.6 + 0.1 * score, 2)

    return {
        "prediction": prediction,
        "confidence": confidence
    }
