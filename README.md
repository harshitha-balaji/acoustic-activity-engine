# 🎙️ Acoustic Activity Engine (AAE)

> A rule-based audio analysis pipeline for speaker activity tracking using acoustic feature matching and configurable activity detection.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![librosa](https://img.shields.io/badge/Audio-librosa-FF6B35?style=flat-square)](https://librosa.org/)
[![NumPy](https://img.shields.io/badge/Numerical-NumPy-013243?style=flat-square)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/Assignment-SciPy-8CAAE6?style=flat-square)](https://scipy.org/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Overview

AAE detects and tracks speaker activity within an audio recording by combining acoustic feature extraction, similarity-based identity matching, and configurable activity confirmation. Instead of treating each speech segment independently, it associates observations over time to produce consistent speaker activity events.

The project was built to explore lightweight, signal-processing approaches to speaker activity tracking without relying on pretrained machine learning models.

---

## Features

- Acoustic feature extraction using MFCCs and pitch
- Similarity-based speaker association
- Hungarian assignment for observation matching
- Configurable acoustic profiles
- Fully configuration-driven architecture

---

## Pipeline

```text
Audio Input
     │
     ▼
Feature Extraction
     │
     ▼
Speaker Association
     │
     ▼
Activity Resolution
     │
     ▼
Activity Report
```

---

## Tech Stack

- Python
- librosa
- NumPy
- SciPy
- JSON Configuration

---

## Project Structure

```text
acoustic_activity_engine/
├── acoustic_activity_engine.py
├── acoustic_profiles.json
├── engine_settings.json
└── requirements.txt
```

---

## Design

AAE intentionally follows a deterministic signal-processing pipeline built from acoustic features rather than learned speaker embeddings. Separating acoustic profiles from the core implementation allows the same tracking pipeline to be adapted to different recording environments without modifying the underlying logic.

---

## Limitations

- Performance depends on recording quality and background noise.
- Overlapping speech can reduce tracking accuracy.
- Speaker identities are session-specific and do not persist across recordings.

---

## Future Improvements

- Neural pitch estimation
- Real-time microphone streaming
- Transcript alignment
- Overlapping speaker detection
- Speaker embedding integration

---

## Sample Audio Attribution

Sample conversation recording used for testing was sourced from:

https://www.kaggle.com/datasets/mozillaorg/common-voice/data

Credit belongs to the original author and dataset providers.

---

*Built as an exploration of rule-based speaker activity tracking using acoustic signal processing.*
