# Voxia Open

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Stars](https://img.shields.io/github/stars/voxia-ai/voxia-open)

English | [日本語](README_ja.md)

**High-performance speech synthesis runtime for Voice AI applications**

<p align="center">
  <img src="docs/demo.gif" width="900">
</p>

Voxia Open is an experimental **speech synthesis runtime** designed for real-time Voice AI applications.

It provides a **local inference runtime compatible with SBV2-style models** and serves as the open foundation of the Voxia ecosystem.

The long-term goal of Voxia is to become a **Voice AI Operating System**.

---
<br>


# Overview

Modern Voice AI applications require:

- low-latency speech synthesis
- streaming audio generation
- flexible model backends
- scalable runtime architecture

Voxia Open provides a **runtime layer between applications and voice models**, enabling flexible integration of speech models and future cloud services.

---
<br>


# Features

- Local speech synthesis runtime
- SBV2-compatible model support
- Streaming TTS
- Python API
- Benchmark tools
- Runtime / Adapter architecture
- Designed for Voice AI applications

---
<br>


# Quick Start

Clone the repository and install the package.

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

# Minimal Example

```bash
from voxia import TTS

tts = TTS.load("/path/to/model_dir")

wav, sr = tts.speak("Hello. This is Voxia Open.")
```

---
<br>

# Streaming Example
```bash
from voxia import TTS

tts = TTS.load("/path/to/model_dir")

for chunk, sr in tts.stream("Hello. This is a streaming demo."):
    ...
```

---
<br>

# Demo
Run the demo script.
```bash
python src/examples/demo.py \
  --model /path/to/model_dir \
  --text "Hello. This is the Voxia Open demo." \
  --out demo_out.wav
```
The generated audio will be saved to:
```bash
demo_out.wav
```

---
<br>

# Model
Voxia Open does not include pretrained models.
Please prepare a compatible SBV2 model and specify it using:
```bash
--model /path/to/model_dir
```

---
<br>

# Benchmark
Run the benchmark tool:
```bash
PYTHONPATH=./src python3 examples/benchmark.py \
  --model ./path/to/model_dir \
  --device cpu \
  --runs 5 \
  --warmup 1 \
  --threads 4 \
  --json-out voxia_bench.json
```
Metrics:

RTF (Real Time Factor)
<br>If RTF < 1.0, synthesis is faster than real-time.

TTFB (Time To First Byte)
<br>Time until the first audio chunk is returned.

---
<br>

# Architecture
Voxia introduces a runtime layer between applications and voice models.
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
This architecture allows Voxia to support:
- SBV2-compatible models

- Voxia-native models

- cloud execution

- edge execution

- with the same API.

---
<br>

# Project Structure
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
├ LICENSE
└ pyproject.toml
```

---
<br>

# Voxia Ecosystem
```bash
Voxia
├ Voxia Open     (open source runtime)
├ Voxia Cloud    (commercial API)
├ Voxia Studio   (development tools)
├ Voxia Edge     (lightweight runtime)
└ Voxia Core     (proprietary models)
```

---
<br>

# Roadmap

Phase 1

- SBV2-compatible runtime

- Streaming speech synthesis

- Benchmark tools

Phase 2

- Voxia runtime engine

- Cloud API integration

- improved Japanese speech pipeline

Phase 3

- Voxia native models

- Voice AI agents

- Edge runtime

---
<br>

# Current Status
Voxia Open is currently under active development.

The main goals are:

- local speech inference runtime

- runtime / adapter architecture

- foundation for Voxia Cloud and Voxia Core


---
<br>

# Contributing
Issues and pull requests are welcome.

Areas where contributions are especially helpful:

- runtime architecture

- streaming improvements

- Japanese NLP pipeline

- documentation

- benchmarking

---
<br>

# License
Apache License 2.0

---
<br>

# Vision

Voxia aims to become the foundation for:

- real-time voice applications

- AI assistants

- voice agents

- robotics

- games

- edge AI devices

**Voxia = Voice AI Operating System**