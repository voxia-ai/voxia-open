from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


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


@dataclass
class PreparedChunk:
    index: int
    text: str
    speaker: str
    style: str
    style_weight: float
    is_first: bool = False
    seed: Optional[int] = None


@dataclass
class FeatureBatch:
    phones: torch.Tensor
    phone_lens: torch.Tensor
    bert: torch.Tensor
    lang_ids: Optional[torch.Tensor] = None
    tone_ids: Optional[torch.Tensor] = None
    style_vec: Optional[torch.Tensor] = None
    spk_id: Optional[torch.Tensor] = None


@dataclass
class AudioChunk:
    index: int
    wav: np.ndarray
    sample_rate: int
    is_first: bool = False
    is_last: bool = False