# src/voxia/model/__init__.py
"""
Voxia model package.

このパッケージは以下を提供します:
- SBV2互換 Synth（段階的置換で実装を進める）
- HiFi-GAN系 Decoder（vocoder）
- state_dict から器を作る StateTree
"""

from .sbv2_native import SBV2Native

__all__ = ["SBV2Native"]