from __future__ import annotations

from typing import Iterator

import numpy as np

from .types import InferenceRequest


class PreparedChunk:
    def __init__(self, index, text, speaker, style, style_weight, is_first=False, seed=None):
        self.index = index
        self.text = text
        self.speaker = speaker
        self.style = style
        self.style_weight = style_weight
        self.is_first = is_first
        self.seed = seed


class ChunkScheduler:
    def split_text(self, text: str, *, first_chunk_chars: int = 40, chunk_chars: int = 160) -> list[str]:
        text = (text or "").strip()
        if not text:
            return [""]

        first_chunk_chars = max(10, int(first_chunk_chars))
        chunk_chars = max(first_chunk_chars, int(chunk_chars))

        if len(text) <= first_chunk_chars:
            return [text]

        first = text[:first_chunk_chars].strip()
        rest = text[first_chunk_chars:].strip()

        out = [first] if first else []
        while rest:
            out.append(rest[:chunk_chars].strip())
            rest = rest[chunk_chars:].strip()

        return [x for x in out if x]

    def prepare_chunks(self, req: InferenceRequest) -> list[PreparedChunk]:
        if req.split:
            texts = self.split_text(
                req.text,
                first_chunk_chars=req.first_chunk_chars,
                chunk_chars=req.chunk_chars,
            )
        else:
            texts = [req.text]

        chunks: list[PreparedChunk] = []
        for i, text in enumerate(texts):
            seed = (req.seed + i) if req.seed is not None else None
            chunks.append(
                PreparedChunk(
                    index=i,
                    text=text,
                    speaker=req.speaker,
                    style=req.style,
                    style_weight=float(req.style_weight),
                    is_first=(i == 0),
                    seed=seed,
                )
            )
        return chunks


class StreamScheduler:
    def iter_wav_frames(self, wav: np.ndarray, sr: int, frame_sec: float = 0.25) -> Iterator[np.ndarray]:
        hop = max(1, int(sr * frame_sec))
        n = int(wav.shape[0])
        i = 0
        while i < n:
            yield wav[i : i + hop]
            i += hop