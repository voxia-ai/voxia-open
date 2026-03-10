#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    import wave

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

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


def _load_tts(model: str, device: str = "cpu"):
    from voxia import TTS
    return TTS.load(model, format="sbv2", device=device, strict=False)


def cmd_speak(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    tts = _load_tts(args.model, device=args.device)
    t1 = time.perf_counter()

    wav, sr = tts.speak(
        args.text,
        speaker=args.speaker,
        style=args.style,
        style_weight=args.style_weight,
        speed=args.speed,
    )
    t2 = time.perf_counter()

    _save_wav(Path(args.out), wav, sr)

    audio_sec = len(wav) / sr if sr > 0 else 0.0
    gen_sec = t2 - t1
    rtf = gen_sec / audio_sec if audio_sec > 0 else float("inf")

    print("Voxia CLI - speak")
    print(f"model       : {args.model}")
    print(f"device      : {args.device}")
    print(f"sample_rate : {sr}")
    print(f"samples     : {len(wav)}")
    print(f"audio_sec   : {audio_sec:.3f}")
    print(f"load_sec    : {t1 - t0:.3f}")
    print(f"gen_sec     : {gen_sec:.3f}")
    print(f"rtf         : {rtf:.3f}")
    print(f"saved       : {args.out}")
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    tts = _load_tts(args.model, device=args.device)

    total = 0
    sr_out: Optional[int] = None

    print("Voxia CLI - stream")
    print(f"model  : {args.model}")
    print(f"device : {args.device}")
    print("")

    t0 = time.perf_counter()
    first_chunk_at: Optional[float] = None

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
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()

        total += len(chunk)
        sr_out = sr
        print(f"[chunk {i:03d}] samples={len(chunk)} sec={len(chunk)/sr:.3f}")

    t1 = time.perf_counter()

    if sr_out is None:
        print("no audio chunk generated")
        return 1

    total_sec = total / sr_out
    ttfb = (first_chunk_at - t0) if first_chunk_at is not None else float("nan")

    print("")
    print(f"ttfb_sec    : {ttfb:.3f}")
    print(f"audio_sec   : {total_sec:.3f}")
    print(f"stream_sec  : {t1 - t0:.3f}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    tts = _load_tts(args.model, device=args.device)

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

    print("[1/3] Loading runtime...")
    print("      OK")
    print("")

    print("[2/3] Generating speech...")
    t0 = time.perf_counter()
    wav, sr = tts.speak(
        args.text,
        speaker=args.speaker,
        style=args.style,
        style_weight=args.style_weight,
        speed=args.speed,
    )
    t1 = time.perf_counter()
    print(f"      OK  ({t1 - t0:.3f} s)")
    print("")

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    audio_sec = len(wav) / sr if sr > 0 else 0.0
    rtf = (t1 - t0) / audio_sec if audio_sec > 0 else float("inf")

    print("[3/3] Result")
    print(f"      Sample rate : {sr}")
    print(f"      Samples     : {len(wav)}")
    print(f"      Audio length: {audio_sec:.3f} s")
    print(f"      Gen time    : {t1 - t0:.3f} s")
    print(f"      RTF         : {rtf:.3f}")
    print("")

    if args.out:
        _save_wav(Path(args.out), wav, sr)
        print(f"Saved audio to: {args.out}")
        print("")

    print("Done.")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from voxia import TTS

    texts = [
        ("short", "こんにちは。今日はいい天気ですね。"),
        ("medium", "こんにちは。Voxia のベンチマークを実行しています。ストリーミングと生成速度を確認します。"),
        ("long", "本日は音声合成エンジン Voxia のベンチマークを実行します。CPU上での推論速度やストリーミング挙動を測定します。"),
    ]

    tts = TTS.load(args.model, format="sbv2", device=args.device, strict=False)

    print("Voxia CLI - benchmark")
    print(f"model   : {args.model}")
    print(f"device  : {args.device}")
    print(f"runs    : {args.runs}")
    print("")

    for name, text in texts:
        rtfs = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            wav, sr = tts.speak(text)
            t1 = time.perf_counter()

            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            audio_sec = len(wav) / sr if sr > 0 else 0.0
            gen_sec = t1 - t0
            rtf = gen_sec / audio_sec if audio_sec > 0 else float("inf")
            rtfs.append(rtf)

        print(f"{name:>6} | mean_rtf={sum(rtfs)/len(rtfs):.3f} | runs={args.runs}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voxia", description="Voxia Open CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser):
        sp.add_argument("--model", required=True, help="モデルディレクトリ")
        sp.add_argument("--device", default="cpu", help="推論デバイス")
        sp.add_argument("--speaker", default="default", help="話者名")
        sp.add_argument("--style", default="Neutral", help="スタイル名")
        sp.add_argument("--style-weight", type=float, default=1.0, help="スタイル強度")
        sp.add_argument("--speed", type=float, default=1.0, help="速度")

    sp_speak = sub.add_parser("speak", help="テキストから音声を生成")
    add_common(sp_speak)
    sp_speak.add_argument("--text", required=True, help="読み上げテキスト")
    sp_speak.add_argument("--out", default="output.wav", help="出力 wav")
    sp_speak.set_defaults(func=cmd_speak)

    sp_stream = sub.add_parser("stream", help="ストリーミング生成")
    add_common(sp_stream)
    sp_stream.add_argument("--text", required=True, help="読み上げテキスト")
    sp_stream.add_argument("--frame-sec", type=float, default=0.25, help="フレーム秒")
    sp_stream.set_defaults(func=cmd_stream)

    sp_demo = sub.add_parser("demo", help="README向けデモ")
    add_common(sp_demo)
    sp_demo.add_argument("--text", default="こんにちは。これは Voxia Open のデモです。", help="読み上げテキスト")
    sp_demo.add_argument("--out", default="demo_out.wav", help="出力 wav")
    sp_demo.set_defaults(func=cmd_demo)

    sp_bench = sub.add_parser("benchmark", help="簡易ベンチマーク")
    sp_bench.add_argument("--model", required=True, help="モデルディレクトリ")
    sp_bench.add_argument("--device", default="cpu", help="推論デバイス")
    sp_bench.add_argument("--runs", type=int, default=3, help="繰り返し回数")
    sp_bench.set_defaults(func=cmd_benchmark)

    sp_serve = sub.add_parser("serve", help="HTTP API server")
    sp_serve.add_argument("--model", required=True, help="モデルディレクトリ")
    sp_serve.add_argument("--device", default="cpu", help="推論デバイス")
    sp_serve.add_argument("--host", default="127.0.0.1", help="bind host")
    sp_serve.add_argument("--port", type=int, default=8000, help="bind port")
    sp_serve.set_defaults(func=cmd_serve)

    return p

def cmd_serve(args: argparse.Namespace) -> int:
    from voxia.server import run_server

    run_server(
        model=args.model,
        device=args.device,
        host=args.host,
        port=args.port,
    )
    return 0

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())