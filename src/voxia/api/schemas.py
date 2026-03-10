from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TTSRequestBody(BaseModel):
    text: str = Field(..., description="text to synthesize")
    speaker: str = Field(default="default")
    style: str = Field(default="Neutral")
    style_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    speed: float = Field(default=1.0, gt=0.0)
    seed: Optional[int] = Field(default=None)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    sample_rate: Optional[int] = None