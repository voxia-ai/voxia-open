#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _project_root() -> Path:
    # src/examples/demo.py -> voxia-open/
    return Path(__file__).resolve().parents[2]


def _ensure_src_on_path() -> None:
    root = _project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _format_sec(x: float) -> str:
    if x < 1.0:
        return f"{x * 1000:.0f} ms"
    return f"{x:.3f} s"


def _save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    """
    依存を増やさずに標準ライブラリだけで wav 保存
    保存前に軽く normalize する
    """
    import wave

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    # normalize
    peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    if peak > 1e-8:
        wav = wav / peak * 0.9

    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sr))
        f.writeframes(pcm16.tobytes())
        

def main() -> int:
    parser = argparse.ArgumentParser(description="Voxia Open demo")
    parser.add_argument("--model", default="path/to/model", help="モデルディレクトリまたは ckpt へのパス")
    parser.add_argument("--text", default="こんにちは。Voxia Open のデモです。", help="読み上げるテキスト")
    parser.add_argument("--speaker", default="default", help="話者名")
    parser.add_argument("--style", default="Neutral", help="スタイル名")
    parser.add_argument("--style-weight", type=float, default=1.0, help="スタイル強度")
    parser.add_argument("--speed", type=float, default=1.0, help="読み上げ速度")
    parser.add_argument("--device", default="cpu", help="推論デバイス")
    parser.add_argument("--out", default="demo_out.wav", help="出力 wav ファイル名")
    parser.add_argument("--no-save", action="store_true", help="wav を保存しない")
    args = parser.parse_args()

    _ensure_src_on_path()

    print("=" * 52)
    print("Voxia Open Demo")
    print("=" * 52)
    print(f"Model   : {args.model}")
    print(f"Device  : {args.device}")
    print(f"Speaker : {args.speaker}")
    print(f"Style   : {args.style} (weight={args.style_weight})")
    print(f"Speed   : {args.speed}")
    print(f"Text    : {args.text}")
    print("")

    try:
        from voxia import TTS
    except Exception as e:
        print("[ERROR] voxia の import に失敗しました。")
        print("        PYTHONPATH や src 配置を確認してください。")
        print("")
        print("detail:", repr(e))
        return 1

    try:
        print("[1/3] Loading runtime...")
        t0 = time.perf_counter()
        tts = TTS.load(
            args.model,
            format="sbv2",
            device=args.device,
            strict=False,
        )
        t1 = time.perf_counter()
        print(f"      OK  ({_format_sec(t1 - t0)})")
        print("")

        print("[2/3] Generating speech...")
        t2 = time.perf_counter()
        wav, sr = tts.speak(
            args.text,
            speaker=args.speaker,
            style=args.style,
            style_weight=args.style_weight,
            speed=args.speed,
        )
        t3 = time.perf_counter()
        print(f"      OK  ({_format_sec(t3 - t2)})")
        print("")

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio_sec = (len(wav) / sr) if sr > 0 else 0.0
        rtf = ((t3 - t2) / audio_sec) if audio_sec > 0 else float("inf")

        print("[3/3] Result")
        print(f"      Sample rate : {sr}")
        print(f"      Samples     : {len(wav)}")
        print(f"      Audio length: {audio_sec:.3f} s")
        print(f"      Gen time    : {_format_sec(t3 - t2)}")
        print(f"      RTF         : {rtf:.3f}")
        print("")

        if not args.no_save:
            out_path = Path(args.out)
            _save_wav(out_path, wav, sr)
            print(f"Saved audio to: {out_path}")
        else:
            print("Skip saving audio (--no-save)")
        print("")
        print("Done.")
        return 0

    except FileNotFoundError as e:
        print("[ERROR] モデルファイルが見つかりません。")
        print("        --model に正しいモデルディレクトリを指定してください。")
        print("")
        print("detail:", str(e))
        return 2

    except Exception as e:
        import traceback
        print("[ERROR] デモ実行中に例外が発生しました。")
        print("")
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    raise SystemExit(main())