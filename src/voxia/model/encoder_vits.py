# src/voxia/model/encoder_vits.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_vits import MultiHeadAttentionRel


class LayerNormBetaGamma(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(int(channels)))
        self.beta = nn.Parameter(torch.zeros(int(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)


class FFNConv(nn.Module):
    def __init__(self, channels: int, filter_channels: int, kernel_size: int, p_dropout: float):
        super().__init__()
        C = int(channels)
        Fch = int(filter_channels)
        k = int(kernel_size)
        pad = (k - 1) // 2
        self.conv_1 = nn.Conv1d(C, Fch, k, padding=pad)
        self.conv_2 = nn.Conv1d(Fch, C, k, padding=pad)
        self.drop = nn.Dropout(float(p_dropout))

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
        y = self.conv_1(x)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.conv_2(y)
        if x_mask is not None:
            y = y * x_mask
        return y


class Encoder(nn.Module):
    """
    SBV2 encoder 互換
    + enc_p.encoder.spk_emb_linear.* を実装（話者埋め込みを足す）
    """
    def __init__(
        self,
        *,
        channels: int,
        n_heads: int,
        n_layers: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        max_rel_dist: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        C = int(channels)
        L = int(n_layers)

        self.attn_layers = nn.ModuleList(
            [
                MultiHeadAttentionRel(
                    channels=C,
                    n_heads=int(n_heads),
                    p_dropout=float(p_dropout),
                    max_rel_dist=int(max_rel_dist),
                )
                for _ in range(L)
            ]
        )
        self.ffn_layers = nn.ModuleList(
            [
                FFNConv(
                    channels=C,
                    filter_channels=int(filter_channels),
                    kernel_size=int(kernel_size),
                    p_dropout=float(p_dropout),
                )
                for _ in range(L)
            ]
        )
        self.norm_layers_1 = nn.ModuleList([LayerNormBetaGamma(C) for _ in range(L)])
        self.norm_layers_2 = nn.ModuleList([LayerNormBetaGamma(C) for _ in range(L)])
        self.drop = nn.Dropout(float(p_dropout))

        # ★SBV2: enc_p.encoder.spk_emb_linear.*
        self.gin_channels = int(gin_channels)
        if self.gin_channels > 0:
            self.spk_emb_linear = nn.Linear(self.gin_channels, C)
        else:
            self.spk_emb_linear = None

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor], g: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,C,T), g: (B,gin) or (B,gin,T)
        if self.spk_emb_linear is not None and g is not None:
            if g.dim() == 3:
                g0 = g.mean(dim=2)  # (B,gin)
            else:
                g0 = g
            x = x + self.spk_emb_linear(g0).unsqueeze(2)

        for i in range(len(self.attn_layers)):
            y = self.attn_layers[i](x, x_mask)
            x = x + self.drop(y)
            x = self.norm_layers_1[i](x)
            if x_mask is not None:
                x = x * x_mask

            y = self.ffn_layers[i](x, x_mask)
            x = x + self.drop(y)
            x = self.norm_layers_2[i](x)
            if x_mask is not None:
                x = x * x_mask

        return x