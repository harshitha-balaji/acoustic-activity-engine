# 🎙️ Acoustic Activity Engine (AAE)

> **Config-driven speaker detection and activity tracking engine** — extracts voiced observations from audio, maintains speaker identity across frames via MFCC-pitch cosine assignment, and emits only confirmed, durable activity events. No ML models. Pure acoustic signal processing.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![librosa](https://img.shields.io/badge/Audio-librosa-FF6B35?style=flat-square)](https://librosa.org/)
[![NumPy](https://img.shields.io/badge/DSP-NumPy-013243?style=flat-square)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/Assignment-SciPy-8CAAE6?style=flat-square)](https://scipy.org/)
![Config](https://img.shields.io/badge/architecture-config--driven-yellow?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## What is this?

Single-threshold energy detectors flag noise as speech. AAE solves this with a three-stage pipeline — voiced observations are extracted, matched to confirmed identities via a combined MFCC-pitch similarity matrix, and promoted to activity events only after sustaining across multiple frames.

Give it an audio file and an acoustic profile, and it tracks every distinct speaker across time: assigning identities, persisting them through silence gaps, and emitting start and stop events only for confirmed, durable activity.

```
Available Acoustic Profiles:
> ['meeting', 'wildlife']

Enter profile:
> meeting

Audio file:
> sample_audio.wav

[INFO] Profile: MEETING
[INFO] Audio: sample_audio.wav

0.60s : Entity_0 STARTED
3.84s : Entity_1 STARTED
11.52s : Entity_0 STOPPED
...

==============================
ACTIVITY SUMMARY
==============================
Profile: meeting
Total Events: 6
==============================
```

---

## Features

- **Energy-gated observation extraction** — silences and sub-threshold noise frames are discarded before any identity logic runs
- **MFCC-pitch combined similarity** — vocal tract shape via cosine MFCC distance fused with log-normalised fundamental frequency for cross-speaker discrimination
- **Hungarian assignment matching** — globally optimal observation-to-entity assignment via linear sum on the full similarity cost matrix
- **Two-stage confirmation gate** — observations enter a pending buffer and are only promoted to confirmed entities after sustained cross-frame evidence
- **Exponential moving average signature update** — confirmed entity signatures adapt gradually over time rather than snapping to each new observation
- **Multi-domain acoustic profiles** — switch between meeting, wildlife, or any custom environment via config without touching engine logic

---

## How It Works

Three sequential stages process each audio file:

### 1. Acoustic Extraction
Six-stage pipeline that isolates voiced observations from raw audio:
- **Frame segmentation** — audio is chunked into fixed-duration hop windows at a configured stride
- **Energy gating** — frames below the minimum energy threshold are discarded immediately
- **Context window assembly** — a wider multi-hop context window is assembled around each active frame for stable spectral averaging
- **MFCC signature computation** — mel-frequency cepstral coefficients extracted across the context window and collapsed to a normalised mean vector; optional delta-MFCC appended for dynamic texture
- **Pitch estimation** — autocorrelation-based F0 detection run across sub-frames within the context window; voiced sub-frames averaged and log-normalised to a `[0, 1]` scale spanning the human voice range
- **Observation assembly** — timestamp, energy scalar, MFCC vector, and log-F0 value packaged into a typed `AcousticObservation` dataclass

### 2. Identity Tracking
Continuous speaker association across sequential observations:
- **Similarity matrix assembly** — combined MFCC cosine and pitch proximity scores computed between every active entity and every incoming observation
- **Hungarian optimal assignment** — linear sum assignment minimises the global cost matrix, preventing greedy nearest-neighbour mismatches
- **Pending promotion pipeline** — unmatched observations enter a staging buffer; entities are only confirmed after meeting the minimum confirmation frame count
- **EMA signature evolution** — matched entity signatures and pitch values updated via exponential moving average; unvoiced frames excluded from pitch blending to prevent spurious drift
- **Memory decay** — unmatched confirmed entities and stale pending candidates age out independently after exceeding their memory limits

### 3. Activity Resolution
Event emission logic that suppresses transient noise from the output:
- **Duration gate** — newly confirmed entities are only announced if their span from first to last observation meets the minimum activity duration threshold
- **STARTED emission** — qualifying entities emit a timestamped start event anchored to first observation
- **STOPPED emission** — entities removed from the tracker emit a stop event offset by one window duration from their last confirmed frame
- **Event log assembly** — all emitted events collected and returned as a typed `ActivityEvent` list for downstream use

---

## Project Structure

```
acoustic_activity_engine/
├── acoustic_activity_engine.py   # Core audio pipeline + interactive runner
├── acoustic_profiles.json        # Environment-specific thresholds and window parameters
├── engine_settings.json          # Tracker memory, smoothing, assignment, and confirmation settings
└── requirements.txt
```

`acoustic_activity_engine.py` contains only logic — never hardcoded environment values. All acoustic parameters live in config.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Audio loading and MFCC extraction | `librosa` |
| Numerical computation and similarity matrices | `NumPy` |
| Optimal Hungarian assignment | `SciPy` (`linear_sum_assignment`) |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the engine

```bash
python acoustic_activity_engine.py
```

### 3. Follow the prompts

```
Available: ['meeting', 'wildlife']

Profile: meeting

Audio file: sample_audio.wav
```

---

## Configuration

### `engine_settings.json`

| Parameter | What it controls |
|-----------|-----------------|
| `memory_limit` | Maximum frames a confirmed entity or pending candidate persists without a match before deregistration |
| `signature_smoothing_alpha` | EMA weight on the existing signature during identity updates; higher values resist drift |
| `min_confirmation_frames` | Minimum cross-frame matches required before a pending observation is promoted to a confirmed entity |
| `mfcc_alpha` | Weight given to MFCC cosine similarity in the combined score; `1 − mfcc_alpha` goes to pitch proximity |

### `acoustic_profiles.json`

Each acoustic profile contains:
- `window_duration` — hop window length in seconds used during frame segmentation
- `minimum_energy` — RMS energy floor below which frames are discarded
- `n_mfcc` — number of mel-frequency cepstral coefficients extracted per frame
- `use_delta_mfcc` — whether delta-MFCC dynamic features are appended to the signature vector
- `context_window_multiplier` — number of hops assembled into the wider context window for stable spectral averaging
- `f0_frame_duration` — sub-frame length in seconds used during pitch estimation
- `similarity_threshold` — minimum combined score required for an observation to be associated with an existing entity
- `minimum_activity_duration` — minimum elapsed time in seconds before a new entity emits a STARTED event

To add a new environment (e.g. `podcast`, `call_centre`), add a top-level key with the same structure and load it through the engine runner.

---

## Design Decisions

**Why no ML model?**
AAE was built signal-first deliberately — MFCC extraction and autocorrelation pitch estimation instead of learned speaker embeddings. The engine stays lightweight, interpretable, and runnable on CPU without pretrained weights or GPU dependencies. The ceiling of this approach (overlapping speech, unseen acoustic environments) directly motivates why speaker embedding models like `pyannote` exist.

**Why Hungarian assignment instead of greedy nearest-neighbour?**
Greedy matching picks the best local pair at each step, which can steal an observation from a better-fitting entity further down the cost matrix. Linear sum assignment minimises the global cost across all simultaneous pairings in a single pass, producing consistently lower total mismatch — especially important when two speakers have similar MFCC profiles.

**Why config-driven?**
Hardcoded acoustic thresholds make adaptation impossible across recording environments. Externalising profiles means the same tracking engine handles a meeting and a serene wildlife recording without touching a line of engine logic.

---

## Roadmap

- [ ] Swap pitch estimator for `CREPE` neural F0 model for higher accuracy on degraded audio
- [ ] Real-time streaming mode using `sounddevice` for live microphone input
- [ ] Per-entity transcript alignment via Whisper timestamps
- [ ] Overlap detection for simultaneous multi-speaker frames
- [ ] Export to RTTM diarisation format for benchmarking against standard datasets

---

## Limitations

- MFCC signatures are sensitive to background noise and recording channel characteristics — profiles may need retuning across environments
- Overlapping speech from two speakers in the same frame produces a blended observation that neither entity claims cleanly
- The pitch estimator uses autocorrelation heuristics and may misclassify some voiced frames as unvoiced in noisy conditions
- Speaker identity is relative within a session — entity IDs are not persistent across separate audio files
- Performance depends on appropriate profile-specific energy and similarity thresholds for the target acoustic environment

---
## Sample Audio Attribution

Sample conversation recording used for testing was sourced from:
https://www.kaggle.com/datasets/mozillaorg/common-voice/data

Credit belongs to the original author and dataset providers. Refer to the Kaggle page for licensing and usage terms.

---


*Built signal-first. The ceiling of this approach is the reason speaker embedding models exist.*
