from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from .schemas import HealthResponse, TTSRequestBody


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    if peak > 1e-8:
        wav = wav / peak * 0.9

    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes(pcm16.tobytes())
    return bio.getvalue()


def create_app(*, tts, model_path: str) -> FastAPI:
    app = FastAPI(
        title="Voxia Open API",
        description="HTTP API for Voxia Open",
        version="0.1.0",
    )

    app.state.tts = tts
    app.state.model_path = model_path

    @app.get("/health", response_model=HealthResponse)
    def health():
        sample_rate: Optional[int] = None
        model_loaded = getattr(app.state, "tts", None) is not None
        if model_loaded:
            try:
                sample_rate = int(getattr(app.state.tts.runtime.adapter, "sample_rate", 0)) or None
            except Exception:
                sample_rate = None
        return HealthResponse(
            status="ok",
            model_loaded=model_loaded,
            sample_rate=sample_rate,
        )

    @app.get("/")
    def root():
        return JSONResponse(
            {
                "name": "Voxia Open API",
                "version": "0.1.0",
                "health": "/health",
                "tts": "/tts",
                "model": app.state.model_path,
            }
        )

    @app.post("/tts")
    def tts_synthesize(req: TTSRequestBody):
        wav, sr = app.state.tts.speak(
            req.text,
            speaker=req.speaker,
            style=req.style,
            style_weight=req.style_weight,
            speed=req.speed,
            seed=req.seed,
        )
        payload = _wav_bytes(wav, sr)
        return Response(
            content=payload,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="output.wav"',
                "X-Sample-Rate": str(sr),
            },
        )

    return app