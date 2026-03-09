from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from .schema import ModelBundle


def _load_weights(ckpt: Path):
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(ckpt))
    return torch.load(str(ckpt), map_location="cpu")


def _find_checkpoint(root: Path) -> Optional[Path]:
    patterns = ["*.safetensors", "*.pth", "*.pt"]
    for pat in patterns:
        cands = sorted(root.glob(pat))
        if cands:
            return cands[0]
    return None


def _find_config(root: Path):
    for name in ["config.yaml", "config.yml", "config.json"]:
        p = root / name
        if p.exists():
            if p.suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return None


def _find_style_vectors(root: Path):
    p = root / "style_vectors.npy"
    if p.exists():
        return np.load(str(p)).astype(np.float32)
    return None


def _find_speakers(root: Path):
    p = root / "speakers.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _find_styles(root: Path):
    p = root / "styles.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_sbv2(path: str, search_root=None, strict: bool = False) -> ModelBundle:
    warnings: list[str] = []

    path = Path(path)

    if path.is_dir():
        root = path
        ckpt = _find_checkpoint(root)
    else:
        ckpt = path
        root = Path(search_root) if search_root else path.parent

    if ckpt is None or not Path(ckpt).exists():
        raise FileNotFoundError("Checkpoint not found")

    state_dict = _load_weights(Path(ckpt))

    config = _find_config(root)
    if config is None:
        if strict:
            raise RuntimeError("Config not found")
        warnings.append("Config not found. Using minimal defaults.")
        config = {"sample_rate": 24000, "model": {}, "data": {}}

    data = config.get("data", {}) if isinstance(config, dict) else {}
    sample_rate = int(
        data.get(
            "sampling_rate",
            config.get("sample_rate", 24000) if isinstance(config, dict) else 24000,
        )
    )

    speakers = _find_speakers(root)
    if speakers is None:
        speakers = data.get("spk2id", {"default": 0}) if isinstance(data, dict) else {"default": 0}

    styles = _find_styles(root)
    if styles is None:
        styles = data.get("style2id", {"Neutral": 0}) if isinstance(data, dict) else {"Neutral": 0}

    style_vectors = _find_style_vectors(root)

    conditioner_assets = {}
    if style_vectors is not None:
        conditioner_assets["style_vectors"] = style_vectors
    if isinstance(styles, dict):
        conditioner_assets["style2id"] = styles

    return ModelBundle(
        sample_rate=sample_rate,
        synth_state_dict=state_dict,
        hparams=config,
        speakers=speakers or {"default": 0},
        styles=styles or {"Neutral": 0},
        textproc_assets={},
        conditioner_assets=conditioner_assets,
        warnings=warnings,
    )