from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ModelBundle:
    sample_rate: int
    synth_state_dict: Dict[str, Any]
    hparams: Dict[str, Any]

    speakers: Dict[str, int] = field(default_factory=dict)
    styles: Dict[str, int] = field(default_factory=dict)

    textproc_assets: Dict[str, Any] = field(default_factory=dict)
    conditioner_assets: Dict[str, Any] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)