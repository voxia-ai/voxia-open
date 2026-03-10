#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

from voxia import TTS


def main() -> int:
    parser = argparse.ArgumentParser(description="Voxia Open streaming demo")
    parser.add_argument("--model", required=True, help="モデルディレクトリ")
    parser.add_argument("--text", default="こんにちは。Voxia Open のストリーミングデモです。", help="読み上げテキスト")
    parser.add_argument("--device", default="cpu", help="推論デバイス")
    parser.add_argument("--speaker", default="default", help="話者名")
    parser.add_argument("--style", default="Neutral", help="スタイル名")
    parser.add_argument("--style-weight", type=float, default=1.0, help="スタイル強度")
    parser.add_argument("--speed", type=float, default=1.0, help="速度")
    parser.add_argument("--frame-sec", type=float, default=0.25, help="ストリーム時のフレーム長（秒）")
    parser.add_argument("--no-play", action="store_true", help="再生せずログだけ表示")
    args = parser.parse_args()

    tts = TTS.load(args.model, format="sbv2", device=args.device, strict=False)

    print("=== Voxia Open Streaming Demo ===")
    print("text:", args.text)
    print("device:", args.device)
    print("speaker:", args.speaker)
    print("style:", args.style)
    print("frame_sec:", args.frame_sec)
    print("")

    total_samples = 0
    sr_out = None

    for i, (chunk, sr) in enumerate(
        tts.stream(
            args.text,
            speaker=args.speaker,
            style=args.style,
            style_weight=args.style_weight,
            speed=args.speed,
            frame_sec=args.frame_sec,
        )
    ):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        total_samples += len(chunk)
        sr_out = sr

        print(f"[chunk {i:03d}] samples={len(chunk)} sec={len(chunk)/sr:.3f}")

        if not args.no_play and sd is not None and len(chunk) > 0:
            sd.play(chunk, sr)
            sd.wait()

    if sr_out is not None:
        print("")
        print(f"total_audio_sec={total_samples / sr_out:.3f}")
    else:
        print("音声チャンクが生成されませんでした。")

    if not args.no_play and sd is None:
        print("")
        print("sounddevice が未インストールのため再生をスキップしました。")
        print("再生するには: pip install sounddevice")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())