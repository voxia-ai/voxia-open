from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class ModelAdapter(ABC):
    sample_rate: int

    @abstractmethod
    def infer_text(
        self,
        text: str,
        *,
        speaker: str,
        style: str,
        style_weight: float,
        speed: float,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        raise NotImplementedError

    @abstractmethod
    def resolve_speaker_id(self, speaker: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def resolve_style_vec(self, style: str, style_weight: float):
        raise NotImplementedError