# src/voxia/model/attn_vits.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionRel(nn.Module):
    """
    SBV2互換の相対位置付きMulti-Head Attention（軽量実装）
    - conv_q/k/v/o を持つ（state_dict互換）
    - emb_rel_k / emb_rel_v は「ヘッド共有」: (1, 2*max_rel+1, head_dim)
      ※あなたの ckpt が (1, 9, 96) なのでこれに合わせる

    state_dict key 互換:
      encoder.attn_layers.{i}.conv_q.*, conv_k.*, conv_v.*, conv_o.*
      encoder.attn_layers.{i}.emb_rel_k
      encoder.attn_layers.{i}.emb_rel_v
    """

    def __init__(
        self,
        *,
        channels: int,
        n_heads: int,
        p_dropout: float,
        max_rel_dist: int,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_heads = int(n_heads)
        self.p_dropout = float(p_dropout)
        self.max_rel = int(max_rel_dist)

        assert self.channels % self.n_heads == 0, "channels must be divisible by n_heads"
        self.head_dim = self.channels // self.n_heads

        self.conv_q = nn.Conv1d(self.channels, self.channels, 1)
        self.conv_k = nn.Conv1d(self.channels, self.channels, 1)
        self.conv_v = nn.Conv1d(self.channels, self.channels, 1)
        self.conv_o = nn.Conv1d(self.channels, self.channels, 1)

        # ★ヘッド共有：先頭次元は 1 固定
        rel_size = 2 * self.max_rel + 1
        self.emb_rel_k = nn.Parameter(torch.randn(1, rel_size, self.head_dim) * 0.01)
        self.emb_rel_v = nn.Parameter(torch.randn(1, rel_size, self.head_dim) * 0.01)

        self.drop = nn.Dropout(self.p_dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, H, T, D)
        B, C, T = x.shape
        x = x.view(B, self.n_heads, self.head_dim, T).permute(0, 1, 3, 2)
        return x

    def _rel_indices(self, T: int, device) -> torch.Tensor:
        # index in [0, 2*max_rel]
        pos = torch.arange(T, device=device)
        rel = pos[None, :] - pos[:, None]   # (T,T)
        rel = rel.clamp(-self.max_rel, self.max_rel) + self.max_rel
        return rel  # (T,T)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B,C,T), x_mask: (B,1,T) or None
        B, C, T = x.shape

        q = self._shape(self.conv_q(x))
        k = self._shape(self.conv_k(x))
        v = self._shape(self.conv_v(x))

        # content scores
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B,H,T,T)

        # relative position bias: shared across heads
        rel_idx = self._rel_indices(T, x.device)  # (T,T)
        # emb_rel_k: (1, 2K+1, D) -> (2K+1, D)
        rel_k = self.emb_rel_k[0]  # (2K+1, D)
        # gather -> (T,T,D)
        rel_k_tt = rel_k.index_select(0, rel_idx.reshape(-1)).view(T, T, self.head_dim)
        # q: (B,H,T,D), rel_k_tt: (T,T,D)
        # -> (B,H,T,T)
        attn = attn + torch.einsum("bhtd,ttd->bhtt", q, rel_k_tt)

        if x_mask is not None:
            # mask: (B,1,T) -> key mask (B,1,1,T)
            key_mask = (x_mask <= 0).view(B, 1, 1, T)
            attn = attn.masked_fill(key_mask, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # weighted sum content
        y = torch.matmul(attn, v)  # (B,H,T,D)

        # relative value contribution
        rel_v = self.emb_rel_v[0]  # (2K+1, D)
        rel_v_tt = rel_v.index_select(0, rel_idx.reshape(-1)).view(T, T, self.head_dim)  # (T,T,D)
        y = y + torch.einsum("bhtt,ttd->bhtd", attn, rel_v_tt)

        # (B,H,T,D) -> (B,C,T)
        y = y.permute(0, 1, 3, 2).contiguous().view(B, C, T)
        y = self.conv_o(y)

        if x_mask is not None:
            y = y * x_mask
        return y