# Voxia Open

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Stars](https://img.shields.io/github/stars/voxia-ai/voxia-open)

[English](README.md) | 日本語

**リアルタイム音声AIアプリケーションを構築するための軽量オープンソースランタイム**

Voxia Open は、次世代の音声AIシステムのための  
**実験的オープンソースランタイム**です。

ローカル音声合成、ストリーミングAPI、CLIツール、HTTPサーバーを提供し、  
開発者がリアルタイム音声アプリケーションを構築できる環境を目指しています。

---

## 特徴
- SBV2互換モデル対応
- ローカル推論
- ストリーミング音声合成
- Python API
- CLIツール
- HTTP APIサーバー
- ベンチマークツール
- モジュール型ランタイム設計
---
## インストール

```bash
git clone https://github.com/voxia-ai/voxia-open
cd voxia-open

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

---

## 必要環境
Python 3.9 以上
<br>PyTorch 2.0 以上

---

## クイックスタート
```bash
from voxia import TTS

tts = TTS.load("/path/to/model_dir")

wav, sr = tts.speak("こんにちは。Voxia Open のテストです。")
```

---


## CLI

音声生成
```bash
voxia speak \
  --model /path/to/model_dir \
  --text "こんにちは"
```

ストリーミング生成
```bash
voxia stream \
  --model /path/to/model_dir \
  --text "ストリーミングデモ"
```

ベンチマーク
```bash
voxia benchmark --model /path/to/model_dir
```

---

## HTTP API
Voxia Open は HTTP API サーバーも提供しています。

サーバー起動
```bash
voxia serve --model /path/to/model_dir
```

ヘルスチェック
```bash
curl http://127.0.0.1:8000/health
```

音声生成
```bash
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"こんにちは"}' \
  --output output.wav
```


---


## モデル
Voxia Open には 事前学習済みモデルは含まれていません。

SBV2互換モデルを使用できます。

モデルディレクトリ例
```bash
/path/to/model_dir
 ├ config.json
 ├ model.safetensors
 └ style_vectors.npy
```

---

## デモ
<p align="center"> <img src="docs/demo.gif" width="800"> </p>

---

## ストリーミング

リアルタイム音声生成デモ

<p align="center"> <img src="docs/streaming_demo.gif" width="800"> </p>

---

## アーキテクチャ

<p align="center"> <img src="docs/architecture.png" width="300"> </p>

Voxia は音声モデルとランタイムを分離した設計になっています。
```bash
Application
     ↓
 Voxia API
     ↓
 Voxia Runtime
     ↓
 Model Adapter
     ↓
 Voice Model
```
この構造により、将来的に複数の音声モデルに対応できます。

---

## 現状

Voxia Open は現在 **開発中のプロジェクト**です。

主な目的は以下です。

- ローカル音声推論ランタイム
- ストリーミング音声パイプライン
- モデル互換レイヤー
- 開発者ツール（CLI / HTTP API / ベンチマーク）

現在の実装は、将来の音声AIシステムのための  
**ランタイムアーキテクチャと開発基盤の構築**に重点を置いています。

ネイティブ推論パイプラインはまだ実験段階のため、  
現時点では明瞭な音声を生成しない場合があります。

---

## Voxia エコシステム
```bash
Voxia
├ Voxia Open   (オープンソースランタイム)
├ Voxia Cloud  (マネージドAPIプラットフォーム)
├ Voxia Studio (開発ツール)
├ Voxia Edge   (軽量ランタイム)
└ Voxia Core   (独自音声モデル)
```

---

## ライセンス

Apache License 2.0
---

## ビジョン
Voxia は次のようなアプリケーションの基盤を目指します。

- 音声AIアシスタント

- 音声AIエージェント



- ゲーム

- ロボティクス

- エッジデバイス

**Voxia = Voice AI Operating System**
