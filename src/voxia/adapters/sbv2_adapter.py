from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from voxia.adapters.base import ModelAdapter
from voxia.formats.sbv2_loader import load_sbv2
from voxia.model.sbv2_native import SBV2Native
from voxia.nlp import JapanesePipeline


class SBV2Adapter(ModelAdapter):
    def __init__(self, bundle, model, nlp, device: str = "cpu"):
        self.bundle = bundle
        self.model = model
        self.nlp = nlp
        self.device = str(device)
        self.sample_rate = int(getattr(model, "cfg", {}).get("data", {}).get("sampling_rate", bundle.sample_rate))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        device: str = "cpu",
        strict: bool = False,
    ) -> "SBV2Adapter":
        bundle = load_sbv2(str(path), strict=strict)

        sd = dict(bundle.synth_state_dict)
        sd.pop("iteration", None)

        model = SBV2Native.build_from_state_dict(bundle.hparams, sd, device=device).eval()
        model.load_state_dict(sd, strict=False)
        model.to(device)

        nlp = JapanesePipeline(device=device)
        return cls(bundle=bundle, model=model, nlp=nlp, device=device)

    def resolve_speaker_id(self, speaker: str) -> int:
        spk2id = getattr(self.bundle, "speakers", {}) or {}
        if not spk2id:
            return 0
        if speaker in spk2id:
            return int(spk2id[speaker])

        low = speaker.lower()
        for k, v in spk2id.items():
            if str(k).lower() == low:
                return int(v)

        return int(next(iter(spk2id.values())))

    def resolve_style_vec(self, style: str, style_weight: float):
        assets = getattr(self.bundle, "conditioner_assets", {}) or {}
        vecs = assets.get("style_vectors", None)
        style2id = assets.get("style2id", None) or getattr(self.bundle, "styles", {}) or {}

        if vecs is None:
            return None

        vecs = np.asarray(vecs, dtype=np.float32)

        def find_id(name: str):
            if name in style2id:
                return int(style2id[name])
            low = name.lower()
            for k, v in style2id.items():
                if str(k).lower() == low:
                    return int(v)
            return None

        sid = find_id(style) if style else None
        neutral_id = find_id("Neutral") or find_id("neutral") or 0
        if sid is None:
            sid = neutral_id

        v = vecs[int(sid)]
        vn = vecs[int(neutral_id)]
        w = float(style_weight)
        w = 0.0 if w < 0 else 1.0 if w > 1 else w
        return (vn + w * (v - vn)).astype(np.float32)

    @torch.inference_mode()
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
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) & 0xFFFFFFFF)

        feats = self.nlp.text_to_features(text)
        spk_id = self.resolve_speaker_id(speaker)
        style_vec_np = self.resolve_style_vec(style, style_weight)

        style_vec = None
        if style_vec_np is not None:
            style_vec = torch.from_numpy(style_vec_np).unsqueeze(0).to(self.device)

        length_scale = 1.0 / max(float(speed), 1e-6)

        wav_t = self.model.infer_from_features(
            phones=feats.phones,
            phone_lens=feats.phone_lens,
            bert=feats.bert,
            style_vec=style_vec,
            spk_id=torch.tensor([spk_id], dtype=torch.long, device=self.device),
            lang_ids=feats.lang_ids,
            tone_ids=feats.tone_ids,
            length_scale=length_scale,
        )

        wav = wav_t.detach().cpu().float().numpy().reshape(-1)
        return wav, int(self.bundle.sample_rate)