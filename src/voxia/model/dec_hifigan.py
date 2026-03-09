# src/voxia/model/dec_hifigan.py
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _wn(m: nn.Module) -> nn.Module:
    """weight_norm を付ける（state_dict が weight_g/weight_v を持つ層用）"""
    return torch.nn.utils.weight_norm(m)


class ResBlock1(nn.Module):
    """
    VITS系 HiFi-GAN ResBlock (resblock=1)

    対応キー:
      dec.resblocks.N.convs1.i.(weight_g/weight_v/bias)
      dec.resblocks.N.convs2.i.(weight_g/weight_v/bias)
    """

    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        assert len(dilations) == 3, "resblock=1 は dilation が3本の想定です"

        k = int(kernel_size)

        self.convs1 = nn.ModuleList([
            _wn(
                nn.Conv1d(
                    channels,
                    channels,
                    k,
                    1,
                    padding=((k - 1) // 2) * d,
                    dilation=d,
                )
            )
            for d in dilations
        ])

        self.convs2 = nn.ModuleList([
            _wn(
                nn.Conv1d(
                    channels,
                    channels,
                    k,
                    1,
                    padding=(k - 1) // 2,
                    dilation=1,
                )
            )
            for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for m in list(self.convs1) + list(self.convs2):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except Exception:
                pass


class Generator(nn.Module):
    """
    VITS系 HiFi-GAN Generator（SBV2互換）

    ✔ checkpointキー差分に合わせた方針
      - conv_pre / conv_post : weight_norm なし（dec.conv_pre.weight / dec.conv_post.weight）
      - ups / resblocks      : weight_norm あり（weight_g/weight_v）

    対応キー例:
      dec.conv_pre.(weight|bias)
      dec.ups.i.(weight_g|weight_v|bias)
      dec.resblocks.N.convs{1,2}.i.(weight_g|weight_v|bias)
      dec.conv_post.weight
      dec.cond.(weight|bias)  # gin_channels>0 の時
    """

    def __init__(
        self,
        *,
        in_channels: int,                 # inter_channels (例: 192)
        upsample_initial_channel: int,    # 例: 512
        upsample_rates: List[int],        # 例: [8,8,2,2,2]
        upsample_kernel_sizes: List[int], # 例: [16,16,8,2,2]
        resblock_kernel_sizes: List[int], # 例: [3,7,11]
        resblock_dilation_sizes: List[List[int]],  # 例: [[1,3,5],...]
        gin_channels: int = 0,            # 例: 512
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.upsample_initial_channel = int(upsample_initial_channel)
        self.upsample_rates = [int(x) for x in upsample_rates]
        self.upsample_kernel_sizes = [int(x) for x in upsample_kernel_sizes]
        self.num_upsamples = len(self.upsample_rates)

        self.num_kernels = len(resblock_kernel_sizes)
        self.gin_channels = int(gin_channels)

        # conv_pre / conv_post は weight_norm なし（checkpointに weight がある）
        self.conv_pre = nn.Conv1d(self.in_channels, self.upsample_initial_channel, 7, 1, padding=3)

        # 条件付け（gin_channelsがある場合）
        # checkpointに dec.cond.weight/bias があるので名前は "cond"
        if self.gin_channels > 0:
            self.cond = nn.Conv1d(self.gin_channels, self.upsample_initial_channel, 1)
        else:
            self.cond = None

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = self.upsample_initial_channel
        for (u, k) in zip(self.upsample_rates, self.upsample_kernel_sizes):
            # VITS系定番: padding=(k-u)//2
            self.ups.append(
                _wn(nn.ConvTranspose1d(ch, ch // 2, int(k), int(u), padding=(int(k) - int(u)) // 2))
            )
            ch = ch // 2

            # 1 upsample段につき resblock を num_kernels 個追加（合計 num_upsamples*num_kernels）
            for (rk, rd) in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, int(rk), [int(x) for x in rd]))

        # conv_post は bias=False が多い（あなたのキーに conv_post.bias が無い）
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, inter_channels, T)
        g: (B, gin_channels, T)  ※condするなら同時間長
        """
        x = self.conv_pre(x)
        if self.cond is not None and g is not None:
            x = x + self.cond(g)

        rb = 0
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = 0.0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[rb + j](x)
            x = xs / self.num_kernels
            rb += self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        # conv_pre/conv_post は weight_norm なしなので何もしない
        for u in self.ups:
            try:
                torch.nn.utils.remove_weight_norm(u)
            except Exception:
                pass
        for r in self.resblocks:
            r.remove_weight_norm()