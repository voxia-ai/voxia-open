from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSRequest:

    text: str
    speaker: str = "default"
    style: str = "Neutral"
    style_weight: float = 1.0
    speed: float = 1.0
    seed: Optional[int] = None