from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from voxia.adapters import SBV2Adapter
from voxia.runtime import InferenceRequest, VoxiaRuntime


class TTS:
    def __init__(self, runtime: VoxiaRuntime):
        self.runtime = runtime

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        format: str = "sbv2",
        device: str = "cpu",
        strict: bool = False,
    ) -> "TTS":
        if format != "sbv2":
            raise ValueError("Voxia Open v0.1 は format='sbv2' のみ対応です。")

        adapter = SBV2Adapter.load(path, device=device, strict=strict)
        runtime = VoxiaRuntime(adapter)
        return cls(runtime)

    def speak(
        self,
        text: str,
        *,
        speaker: str = "default",
        style: str = "Neutral",
        style_weight: float = 1.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
        normalize: bool = True,
        split: bool = True,
        chunk_chars: int = 160,
    ) -> Tuple[np.ndarray, int]:
        req = InferenceRequest(
            text=text,
            speaker=speaker,
            style=style,
            style_weight=style_weight,
            speed=speed,
            seed=seed,
            normalize=normalize,
            split=split,
            chunk_chars=chunk_chars,
            frame_sec=None,
        )
        return self.runtime.speak(req)

    def stream(
        self,
        text: str,
        *,
        speaker: str = "default",
        style: str = "Neutral",
        style_weight: float = 1.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
        normalize: bool = True,
        split: bool = True,
        chunk_chars: int = 160,
        frame_sec: float = 0.25,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        req = InferenceRequest(
            text=text,
            speaker=speaker,
            style=style,
            style_weight=style_weight,
            speed=speed,
            seed=seed,
            normalize=normalize,
            split=split,
            chunk_chars=chunk_chars,
            frame_sec=frame_sec,
        )
        return self.runtime.stream(req)