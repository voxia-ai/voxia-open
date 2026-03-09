from __future__ import annotations

import re
import unicodedata

from .types import InferenceRequest


class VoxiaPipeline:
    def normalize(self, text: str) -> str:
        t = unicodedata.normalize("NFKC", text).strip()
        for ch in ["\r", "\t"]:
            t = t.replace(ch, " ")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def prepare_request(self, req: InferenceRequest) -> InferenceRequest:
        if req.normalize:
            req.text = self.normalize(req.text)
        return req