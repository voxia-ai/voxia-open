from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionRel(nn.Module):
    """
    SBV2互換の相対位置付きMulti-Head Attention（軽量版）
    - conv_q / conv_k / conv_v / conv_o
    - emb_rel_k / emb_rel_v はヘッド共有: (1, 2K+1, head_dim)
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
        # (T, T) in [0, 2*max_rel]
        pos = torch.arange(T, device=device)
        rel = pos[None, :] - pos[:, None]
        rel = rel.clamp(-self.max_rel, self.max_rel) + self.max_rel
        return rel

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, C, T), x_mask: (B, 1, T) or None
        B, C, T = x.shape

        q = self._shape(self.conv_q(x))  # (B,H,T,D)
        k = self._shape(self.conv_k(x))  # (B,H,T,D)
        v = self._shape(self.conv_v(x))  # (B,H,T,D)

        # content attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B,H,T,T)

        # relative key bias
        rel_idx = self._rel_indices(T, x.device)  # (T,T)
        rel_k = self.emb_rel_k[0]  # (2K+1, D)
        rel_k_tt = rel_k.index_select(0, rel_idx.reshape(-1)).view(T, T, self.head_dim)  # (T,T,D)

        # q: (B,H,T,D), rel_k_tt: (T,T,D) -> (B,H,T,T)
        attn = attn + torch.einsum("bhid,ijd->bhij", q, rel_k_tt)

        if x_mask is not None:
            key_mask = (x_mask <= 0).view(B, 1, 1, T)
            attn = attn.masked_fill(key_mask, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # content value aggregation
        y = torch.matmul(attn, v)  # (B,H,T,D)

        # relative value contribution
        rel_v = self.emb_rel_v[0]  # (2K+1, D)
        rel_v_tt = rel_v.index_select(0, rel_idx.reshape(-1)).view(T, T, self.head_dim)  # (T,T,D)

        # attn: (B,H,T,T), rel_v_tt: (T,T,D) -> (B,H,T,D)
        y = y + torch.einsum("bhij,ijd->bhid", attn, rel_v_tt)

        # (B,H,T,D) -> (B,C,T)
        y = y.permute(0, 1, 3, 2).contiguous().view(B, C, T)
        y = self.conv_o(y)

        if x_mask is not None:
            y = y * x_mask

        return y