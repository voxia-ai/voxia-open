# Voxia Open

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Stars](https://img.shields.io/github/stars/voxia-ai/voxia-open)

[English](README.md) | 日本語

**Voxia Open** は、リアルタイム音声AIアプリケーションのための  
高性能な音声合成ランタイムです。

<p align="center">
  <img src="docs/demo.gif" width="900">
</p>

Voxia Open は **SBV2互換モデルに対応したローカル音声合成ランタイム**を提供します。  
本プロジェクトは、Voxia エコシステムのオープンソース基盤として開発されています。

Voxia の長期的な目標は **Voice AI Operating System（音声AI OS）** を構築することです。

---
<br>

# 概要

現代の音声AIアプリケーションでは、以下のような要件が求められます。

- 低レイテンシ音声合成
- ストリーミング音声生成
- 柔軟なモデルバックエンド
- スケーラブルなランタイム設計

Voxia Open は **アプリケーションと音声モデルの間に Runtime 層**を設けることで  
柔軟な音声AIアーキテクチャを実現します。

---
<br>

# 特徴

- ローカル音声合成ランタイム
- SBV2互換モデル対応
- ストリーミングTTS
- Python API
- ベンチマークツール
- Runtime / Adapter アーキテクチャ
- Voice AIアプリケーション向け設計

---
<br>

# クイックスタート

リポジトリをクローンしてインストールします。

```bash
git clone https://github.com/voxia-ai/voxia-open
cd voxia-open

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

---
<br>

# 最小サンプル

```bash
from voxia import TTS

tts = TTS.load("/path/to/model_dir")

wav, sr = tts.speak("こんにちは。Voxia Open のテストです。")
```

---
<br>

# ストリーミング音声合成

```bash
from voxia import TTS

tts = TTS.load("/path/to/model_dir")

for chunk, sr in tts.stream("こんにちは。ストリーミングデモです。"):
    ...
```

---
<br>

# デモ
デモスクリプトを実行します。
```bash
python src/examples/demo.py \
  --model /path/to/model_dir \
  --text "こんにちは。Voxia Open のデモです。" \
  --out demo_out.wav
```
生成された音声は
```bash
demo_out.wav
```
に保存されます。

---
<br>

# モデル
Voxia Open には 事前学習済みモデルは含まれていません。

互換性のある SBV2 モデルを準備し、以下のように指定してください。
```bash
--model /path/to/model_dir
```

---
<br>

# ベンチマーク
ベンチマークツールを実行できます。
```bash
PYTHONPATH=./src python3 examples/benchmark.py \
  --model ./path/to/model_dir \
  --device cpu \
  --runs 5 \
  --warmup 1 \
  --threads 4 \
  --json-out voxia_bench.json
```
指標

**RTF (Real Time Factor)**
<br>RTF < 1.0 の場合、リアルタイムより高速に生成されます。

**TTFB (Time To First Byte)**
<br>最初の音声チャンクが生成されるまでの時間。

---
<br>

# アーキテクチャ
Voxia は アプリケーションと音声モデルの間に Runtime 層を設けています。
```bash
アプリケーション
      ↓
   Voxia API
      ↓
 Voxia Runtime
      ↓
 Model Adapter
      ↓
  Voice Model
```

この設計により、将来的に以下を同じAPIで扱えるようになります。

- SBV2互換モデル

- Voxia独自モデル

- Cloud実行

- Edge実行


---
<br>

# プロジェクト構成
```bash
voxia-open/
├ src/
│  └ voxia/
│     ├ __init__.py
│     ├ tts.py
│     ├ runtime/
│     ├ adapters/
│     ├ formats/
│     ├ nlp/
│     ├ model/
│     └ utils/
│
├ examples/
│  ├ benchmark.py
│  └ demo.py
│
├ tests/
├ docs/
│  └ demo.gif
│
├ README.md
├ README_ja.md
├ LICENSE
└ pyproject.toml
```

---
<br>

# Voxia エコシステム
```bash
Voxia
├ Voxia Open     (オープンソースランタイム)
├ Voxia Cloud    (商用API)
├ Voxia Studio   (開発ツール)
├ Voxia Edge     (軽量ランタイム)
└ Voxia Core     (独自モデル)
```

---
<br>

# ロードマップ
Phase 1

- SBV2互換ランタイム

- ストリーミング音声合成

- ベンチマークツール

Phase 2

- Voxia Runtime エンジン

- Cloud API 統合

- 日本語音声パイプライン強化

Phase 3

- Voxia 独自モデル

- 音声AIエージェント

- Edge Runtime

---
<br>

# コントリビューション
Issue や Pull Request を歓迎します。

特に以下の分野の貢献を歓迎します。

- Runtime設計

- ストリーミング改善

- 日本語前処理

- ドキュメント

- ベンチマーク改善

---
<br>

# ライセンス
Apache License 2.0

---
<br>

# ビジョン

Voxia は次のようなアプリケーションの基盤になることを目指しています。

- リアルタイム音声アプリ

- AIアシスタント

- 音声エージェント

- ゲーム

- ロボティクス

- エッジAIデバイス

**Voxia = Voice AI Operating System**
