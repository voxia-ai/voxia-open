from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceRequest:
    text: str
    speaker: str = "default"
    style: str = "Neutral"
    style_weight: float = 1.0
    speed: float = 1.0
    seed: Optional[int] = None
    normalize: bool = True
    split: bool = True
    chunk_chars: int = 160
    frame_sec: Optional[float] = None
    first_chunk_chars: int = 40