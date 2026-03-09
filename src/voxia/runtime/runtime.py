from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from voxia.adapters.base import ModelAdapter
from voxia.runtime.pipeline import VoxiaPipeline
from voxia.runtime.scheduler import ChunkScheduler, StreamScheduler
from voxia.runtime.types import InferenceRequest


class VoxiaRuntime:
    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        pipeline: VoxiaPipeline | None = None,
        chunk_scheduler: ChunkScheduler | None = None,
        stream_scheduler: StreamScheduler | None = None,
    ):
        self.adapter = adapter
        self.pipeline = pipeline or VoxiaPipeline()
        self.chunk_scheduler = chunk_scheduler or ChunkScheduler()
        self.stream_scheduler = stream_scheduler or StreamScheduler()

    def speak(self, req: InferenceRequest) -> Tuple[np.ndarray, int]:
        req = self.pipeline.prepare_request(req)
        chunks = self.chunk_scheduler.prepare_chunks(req)

        wavs: list[np.ndarray] = []
        sr_out = int(self.adapter.sample_rate)

        for chunk in chunks:
            wav, sr = self.adapter.infer_text(
                chunk.text,
                speaker=chunk.speaker,
                style=chunk.style,
                style_weight=chunk.style_weight,
                speed=req.speed,
                seed=chunk.seed,
            )
            wavs.append(np.asarray(wav, dtype=np.float32))
            sr_out = sr

        if not wavs:
            return np.zeros((0,), dtype=np.float32), sr_out
        return np.concatenate(wavs, axis=0).astype(np.float32), sr_out

    def stream(self, req: InferenceRequest) -> Iterator[Tuple[np.ndarray, int]]:
        req = self.pipeline.prepare_request(req)
        chunks = self.chunk_scheduler.prepare_chunks(req)

        for chunk in chunks:
            wav, sr = self.adapter.infer_text(
                chunk.text,
                speaker=chunk.speaker,
                style=chunk.style,
                style_weight=chunk.style_weight,
                speed=req.speed,
                seed=chunk.seed,
            )

            if req.frame_sec and req.frame_sec > 0:
                for frame in self.stream_scheduler.iter_wav_frames(wav, sr, req.frame_sec):
                    yield frame.astype(np.float32), sr
            else:
                yield np.asarray(wav, dtype=np.float32), sr