from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class JPFeatures:
    phones: torch.Tensor
    phone_lens: torch.Tensor
    bert: torch.Tensor
    lang_ids: Optional[torch.Tensor] = None
    tone_ids: Optional[torch.Tensor] = None


class JapanesePipeline:
    def __init__(self, device: str = "cpu", bert_dim: int = 1024):
        self.device = str(device)
        self.bert_dim = int(bert_dim)

    def text_to_features(self, text: str) -> JPFeatures:
        # v0.1 最小版:
        # 文字単位で簡易 phones を作る
        chars = list(text.strip()) or [" "]
        phones = [min(111, max(0, ord(c) % 112)) for c in chars]

        T = len(phones)
        phones_t = torch.tensor([phones], dtype=torch.long, device=self.device)
        lens_t = torch.tensor([T], dtype=torch.long, device=self.device)

        # bert は最初はゼロ特徴で開始
        bert_t = torch.zeros((1, self.bert_dim, T), dtype=torch.float32, device=self.device)

        lang_ids = torch.zeros((1, T), dtype=torch.long, device=self.device)
        tone_ids = torch.zeros((1, T), dtype=torch.long, device=self.device)

        return JPFeatures(
            phones=phones_t,
            phone_lens=lens_t,
            bert=bert_t,
            lang_ids=lang_ids,
            tone_ids=tone_ids,
        )