# src/voxia/model/sbv2_native.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_vits import Encoder
from voxia.model.encoder_vits import LayerNormBetaGamma


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)[None, :]
    return ids < lengths[:, None]


def length_regulator(x: torch.Tensor, durations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, T = x.shape
    outs = []
    out_lens = []
    for b in range(B):
        reps = durations[b].clamp_min(0).tolist()
        chunks = []
        for t, r in enumerate(reps):
            if r <= 0:
                continue
            chunks.append(x[b : b + 1, :, t : t + 1].repeat(1, 1, r))
        y = torch.cat(chunks, dim=2) if chunks else x.new_zeros((1, C, 1))
        outs.append(y)
        out_lens.append(y.shape[2])

    max_len = max(out_lens) if out_lens else 1
    padded = []
    for y in outs:
        if y.shape[2] < max_len:
            pad = y.new_zeros((1, C, max_len - y.shape[2]))
            y = torch.cat([y, pad], dim=2)
        padded.append(y)
    y = torch.cat(padded, dim=0)
    return y, torch.tensor(out_lens, device=x.device, dtype=torch.long)


class DurationPredictor(nn.Module):
    def __init__(self, *, in_channels: int, channels: int, p_dropout: float = 0.1):
        super().__init__()
        Cin = int(in_channels)
        C = int(channels)
        self.conv1 = nn.Conv1d(Cin, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.proj = nn.Conv1d(C, 1, kernel_size=1)
        self.drop = float(p_dropout)
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x * x_mask)
        y = F.relu(y)
        y = y.transpose(1, 2)
        y = self.ln1(y)
        y = y.transpose(1, 2)
        y = F.dropout(y, p=self.drop, training=self.training)

        y = self.conv2(y * x_mask)
        y = F.relu(y)
        y = y.transpose(1, 2)
        y = self.ln2(y)
        y = y.transpose(1, 2)
        y = F.dropout(y, p=self.drop, training=self.training)

        out = self.proj(y * x_mask) * x_mask
        return out


# sbv2_native.py の上部（DurationPredictor の代わりに追加）


class DurationPredictorSBV2(nn.Module):
    """
    SBV2 dp のキー互換:
      dp.cond.*
      dp.conv_1.*
      dp.conv_2.*
      dp.norm_1.beta/gamma
      dp.norm_2.beta/gamma
      dp.proj.*
    """
    def __init__(self, *, hidden_channels: int, gin_channels: int, channels: int, p_dropout: float):
        super().__init__()
        H = int(hidden_channels)
        G = int(gin_channels)
        C = int(channels)
        self.p_dropout = float(p_dropout)

        # conditioning (g)
        self.cond = nn.Conv1d(G, H, 1) if G > 0 else None

        self.conv_1 = nn.Conv1d(H, C, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.norm_1 = LayerNormBetaGamma(C)
        self.norm_2 = LayerNormBetaGamma(C)
        self.proj = nn.Conv1d(C, 1, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,H,T), g: (B,G,T)
        if self.cond is not None and g is not None:
            x = x + self.cond(g)

        y = self.conv_1(x * x_mask)
        y = F.relu(y)
        y = self.norm_1(y)
        y = F.dropout(y, p=self.p_dropout, training=self.training)
        y = y * x_mask

        y = self.conv_2(y)
        y = F.relu(y)
        y = self.norm_2(y)
        y = F.dropout(y, p=self.p_dropout, training=self.training)
        y = y * x_mask

        out = self.proj(y) * x_mask
        return out


class IdentityFlow(nn.Module):
    def forward(self, z: torch.Tensor, g: Optional[torch.Tensor] = None, reverse: bool = False) -> torch.Tensor:
        return z


class TextEncoderSBV2(nn.Module):
    """
    ✅ enc_p のキー互換を最大化する実装

    互換キー:
      - enc_p.emb.*
      - enc_p.bert_proj.*
      - enc_p.style_proj.*
      - enc_p.language_emb.weight
      - enc_p.tone_emb.weight
      - enc_p.encoder.attn_layers.{i}.*
      - enc_p.encoder.ffn_layers.{i}.*
      - enc_p.encoder.norm_layers_1.{i}.beta/gamma
      - enc_p.encoder.norm_layers_2.{i}.beta/gamma
    """
    def __init__(
        self,
        gin_channels: int = 0 ,
        *,
        n_vocab: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        bert_dim: int,
        style_dim: int = 256,
        n_languages: int = 1,
        n_tones: int = 1,
        max_rel_dist: int = 192,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)

        self.emb = nn.Embedding(int(n_vocab), self.hidden_channels)
        self.bert_proj = nn.Conv1d(int(bert_dim), self.hidden_channels, 1)
        self.style_proj = nn.Linear(int(style_dim), self.hidden_channels)

        # 追加（ckptにある）
        self.language_emb = nn.Embedding(int(n_languages), self.hidden_channels)
        self.tone_emb = nn.Embedding(int(n_tones), self.hidden_channels)

        self.encoder = Encoder(
            channels=self.hidden_channels,
            gin_channels=int(gin_channels),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            filter_channels=int(filter_channels),
            kernel_size=int(kernel_size),
            p_dropout=float(p_dropout),
            max_rel_dist=int(max_rel_dist),
        )

        self.proj = nn.Conv1d(self.hidden_channels, self.hidden_channels * 2, 1)

    def forward(
        self,
        phones: torch.Tensor,         # (B,T)
        phone_lens: torch.Tensor,     # (B,)
        bert: torch.Tensor,           # (B,bert_dim,T)
        g: Optional[torch.Tensor] = None,
        style_vec: Optional[torch.Tensor] = None,   # (B,256)
        lang_ids: Optional[torch.Tensor] = None,    # (B,T) or (B,)
        tone_ids: Optional[torch.Tensor] = None,    # (B,T) or (B,)
    ):
        B, T = phones.shape
        x = self.encoder(x, x_mask, g=g)

        # bert
        x = x + self.bert_proj(bert)

        # language / tone（SBV2は phoneごとに入ることが多いので (B,T) を許容）
        if lang_ids is None:
            lang_ids = phones.new_zeros((B, T))
        if tone_ids is None:
            tone_ids = phones.new_zeros((B, T))

        if lang_ids.dim() == 1:
            lang_ids = lang_ids[:, None].expand(B, T)
        if tone_ids.dim() == 1:
            tone_ids = tone_ids[:, None].expand(B, T)

        x = x + self.language_emb(lang_ids).transpose(1, 2)
        x = x + self.tone_emb(tone_ids).transpose(1, 2)

        # style
        if style_vec is not None:
            x = x + self.style_proj(style_vec).unsqueeze(2)

        x_mask = sequence_mask(phone_lens, T).to(x.dtype).unsqueeze(1)
        x = x * x_mask

        x = self.encoder(x, x_mask)
        x = x * x_mask

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.hidden_channels, dim=1)
        return x, m, logs


class SBV2Native(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        model = cfg.get("model", {})
        data = cfg.get("data", {})

        self.n_vocab = int(model.get("n_vocab", 512))
        self.hidden_channels = int(model.get("hidden_channels", model.get("inter_channels", 192)))
        self.inter_channels = int(model.get("inter_channels", self.hidden_channels))
        self.filter_channels = int(model.get("filter_channels", 768))
        self.n_heads = int(model.get("n_heads", 2))
        self.n_layers = int(model.get("n_layers", 6))
        self.kernel_size = int(model.get("kernel_size", 3))
        self.p_dropout = float(model.get("p_dropout", 0.1))

        self.bert_dim = int(model.get("bert_dim", 768))
        self.gin_channels = int(model.get("gin_channels", 0))
        self.dp_channels = int(model.get("dp_channels", 256))
        self.sdp_post_proj_out = int(model.get("sdp_post_proj_out", 1))
        self.n_speakers = int(data.get("n_speakers", 1))

        self.n_languages = int(model.get("n_languages", 1))
        self.n_tones = int(model.get("n_tones", 1))

        self.enc_p = TextEncoderSBV2(
            n_vocab=self.n_vocab,
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            bert_dim=self.bert_dim,
            style_dim=256,
            n_languages=self.n_languages,
            n_tones=self.n_tones,
            max_rel_dist=int(model.get("max_rel_dist", 192)),
        )

        self.dp = DurationPredictorSBV2(
            hidden_channels=self.hidden_channels,
            gin_channels=self.gin_channels,
            channels=self.dp_channels,
            p_dropout=self.p_dropout,
        )
        
        # placeholder（次で本実装）
        in_g = self.gin_channels if self.gin_channels > 0 else self.hidden_channels
        self.sdp = nn.ModuleDict(
            {
                "cond": nn.Conv1d(in_g, self.hidden_channels, 1),
                "post_proj": nn.Conv1d(self.hidden_channels, self.sdp_post_proj_out, 1),
            }
        )
        self.flow = IdentityFlow()

        if self.gin_channels > 0:
            self.emb_g = nn.Embedding(self.n_speakers, self.gin_channels)
        else:
            self.emb_g = None

        from voxia.model.dec_hifigan import Generator
        import inspect

        sig = inspect.signature(Generator.__init__).parameters
        cand = dict(
            in_channels=int(self.inter_channels),
            upsample_initial_channel=int(model.get("upsample_initial_channel", 512)),
            initial_channel=int(model.get("upsample_initial_channel", 512)),
            resblock=int(model.get("resblock", 1)),
            resblock_kernel_sizes=[int(x) for x in model.get("resblock_kernel_sizes", [3, 7, 11])],
            resblock_dilation_sizes=[[int(v) for v in row] for row in model.get("resblock_dilation_sizes", [[1, 3, 5]] * 3)],
            upsample_rates=[int(x) for x in model.get("upsample_rates", [8, 8, 2, 2, 2])],
            upsample_kernel_sizes=[int(x) for x in model.get("upsample_kernel_sizes", [16, 16, 8, 2, 2])],
            gin_channels=int(self.gin_channels),
        )
        kwargs = {k: v for k, v in cand.items() if k in sig}
        self.dec = Generator(**kwargs)

    @classmethod
    def build_from_state_dict(cls, cfg: dict, state_dict: dict, *, device: str = "cpu") -> "SBV2Native":
        model_cfg = dict(cfg.get("model", {}))
        data_cfg = dict(cfg.get("data", {}))

        emb = state_dict["enc_p.emb.weight"]  # (n_vocab, hidden)
        n_vocab = int(emb.shape[0])
        hidden = int(emb.shape[1])

        bert_proj = state_dict["enc_p.bert_proj.weight"]  # (hidden, bert_dim, 1)
        bert_dim = int(bert_proj.shape[1])

        sdp_cond = state_dict["sdp.cond.weight"]  # (hidden, gin, 1)
        gin_channels = int(sdp_cond.shape[1])

        dp_proj = state_dict.get("dp.proj.weight", None)  # (1, dp_channels, 1)
        dp_channels = int(dp_proj.shape[1]) if dp_proj is not None else hidden

        sdp_post = state_dict.get("sdp.post_proj.weight", None)  # (out, hidden, 1)
        sdp_post_out = int(sdp_post.shape[0]) if sdp_post is not None else 1

        # language/tone
        n_lang = int(state_dict.get("enc_p.language_emb.weight", torch.empty(1, hidden)).shape[0])
        n_tone = int(state_dict.get("enc_p.tone_emb.weight", torch.empty(1, hidden)).shape[0])

        model_cfg["n_vocab"] = n_vocab
        model_cfg["hidden_channels"] = hidden
        model_cfg["inter_channels"] = int(model_cfg.get("inter_channels", hidden))
        model_cfg["bert_dim"] = bert_dim
        model_cfg["gin_channels"] = gin_channels
        model_cfg["dp_channels"] = dp_channels
        model_cfg["sdp_post_proj_out"] = sdp_post_out
        model_cfg["n_layers"] = int(model_cfg.get("n_layers", 6))
        model_cfg["n_heads"] = int(model_cfg.get("n_heads", 2))
        model_cfg["n_languages"] = n_lang
        model_cfg["n_tones"] = n_tone

        # --- 追加: max_rel_dist を ckpt から推定 ---
        # enc_p.encoder.attn_layers.0.emb_rel_k : (1, 2K+1, head_dim)
        k0 = state_dict.get("enc_p.encoder.attn_layers.0.emb_rel_k", None)
        if k0 is not None and k0.dim() == 3:
            rel_size = int(k0.shape[1])
            max_rel = (rel_size - 1) // 2
            model_cfg["max_rel_dist"] = int(max_rel)
        else:
            model_cfg["max_rel_dist"] = int(model_cfg.get("max_rel_dist", 4))

        cfg2 = dict(cfg)
        cfg2["model"] = model_cfg
        cfg2["data"] = data_cfg

        return cls(cfg=cfg2).to(device)

    def make_g_from_speaker(self, spk_id: torch.Tensor, T: int) -> Optional[torch.Tensor]:
        if self.emb_g is None:
            return None
        g = self.emb_g(spk_id).unsqueeze(2)
        return g.expand(-1, -1, int(T))

    @torch.inference_mode()
    def infer_from_features(
        self,
        *,
        phones: torch.Tensor,
        phone_lens: torch.Tensor,
        bert: torch.Tensor,
        style_vec: Optional[torch.Tensor] = None,
        spk_id: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
        tone_ids: Optional[torch.Tensor] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
    ) -> torch.Tensor:
        B, T_text = phones.shape
        device = phones.device
        if spk_id is None:
            spk_id = torch.zeros((B,), dtype=torch.long, device=device)

        g_text = self.make_g_from_speaker(spk_id, T_text)  # (B,G,T)

        h, m_p, logs_p = self.enc_p(
            phones, phone_lens, bert,
            g=g_text,
            style_vec=style_vec,
            lang_ids=lang_ids,
            tone_ids=tone_ids,
        )

        g = self.make_g_from_speaker(spk_id, T_text)  # (B,G,T)  ※T_textはphone長
        x_mask = sequence_mask(phone_lens, T_text).to(h.dtype).unsqueeze(1)
        logw = self.dp(h, x_mask, g=g)  # ← g を渡す
        w = torch.exp(logw).squeeze(1) * float(noise_scale_w)
        w = w * float(length_scale)

        dur = torch.ceil(w).to(torch.long).clamp_min(1) * sequence_mask(phone_lens, T_text).to(torch.long)

        h_lr, _ = length_regulator(h, dur)
        m_lr, _ = length_regulator(m_p, dur)
        logs_lr, _ = length_regulator(logs_p, dur)

        T_frame = int(h_lr.shape[2])
        g = self.make_g_from_speaker(spk_id, T_frame)

        eps = torch.randn_like(m_lr) * float(noise_scale)
        z_p = m_lr + eps * torch.exp(logs_lr)

        z = self.flow(z_p, g=g, reverse=True)

        try:
            wav = self.dec(z, g=g)
        except TypeError:
            wav = self.dec(z, g)
        return wav