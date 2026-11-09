# 🎧 VibeTuner AI — Deep Learning Audio Emotion Classifier & Acoustic DSP Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Librosa](https://img.shields.io/badge/Librosa-1.0.0-9cf.svg)](https://librosa.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Real-time audio signal processing and deep neural network inference pipeline for automated musical mood extraction, timbral profiling, and acoustic intelligence.**

---

## 🚀 Key Highlights & Architectural Features

- **26 Psychoacoustic Features**: Computes 20 Mel-Frequency Cepstral Coefficients (MFCCs), Spectral Centroid (timbral brightness), Spectral Rolloff (85% energy distribution), Zero Crossing Rate (ZCR / noisiness), RMS energy (loudness), and rhythmic onset tempo (BPM).
- **Calibrated PyTorch Neural Network**: 3-Layer Dense Deep Neural Network (`12,228` parameters) utilizing `BatchNorm1D`, `Dropout` regularization, and `ReLU` non-linearities for robust multi-class probability outputs.
- **Dark Studio Glassmorphism UI**: High-contrast, accessibility-checked interface powered by dynamic linear/radial gradient meshes, backdrop-blur cards, and custom typography (`Plus Jakarta Sans` & `JetBrains Mono`).
- **Interactive Multi-Panel Visualizations**:
  - **Time-Domain Amplitude Envelope** & **STFT Power Spectrogram** (dB)
  - **Interactive Plotly 20-band MFCC Timbre Heatmap**
  - **12-Tone Chromagram (Pitch Class Profile)** for musical key & tonality detection
  - **Acoustic Trajectory Time-Series** (Centroid & Rolloff frequency tracking)
  - **Psychological Vibe Radar** (Arousal, Valence, Danceability, Acoustic Warmth)
- **Built-in Synthetic Audio Presets**: Includes 4 in-memory multi-track audio synthesizers (Synthwave, Meditation Drone, Major Funk, Melancholic Lo-Fi) so reviewers and recruiters can test the complete DSP pipeline with zero external files needed.
- **Exportable Machine Intelligence Dossier**: Instant downloadable JSON report complete with timestamped audio fingerprints (MD5), acoustic parameters, and confidence distributions.

---

## 🏛️ Neural Network Architecture

```
  [Input Audio Stream]
          │
          ▼
  [DSP Feature Extraction: Librosa] ──► 26-Dimensional Feature Vector
          │
          ▼
  [Tensor Normalization Layer] (Mean-Centering & Std Scaling)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ PyTorch AudioClassifierDNN Architecture                │
│                                                        │
│  Layer 1: Linear(26 -> 128) -> ReLU -> BN1D -> Drop0.3 │
│  Layer 2: Linear(128 -> 64) -> ReLU -> BN1D -> Drop0.2 │
│  Layer 3: Linear(64 -> 4)   -> Logits Projection       │
└────────────────────────────────────────────────────────┘
          │
          ▼
  [Softmax Classification Layer]
          │
  ┌───────┼───────────────────┬───────────────────┐
  ▼       ▼                   ▼                   ▼
🌿 Calm  🔥 Energetic / Hype ✨ Happy / Upbeat  🌧️ Melancholic / Sad
```

---

## 📊 Extracted Acoustic Feature Matrix

| Feature Category | Dimension | Psychological / Musical Correlation |
| :--- | :--- | :--- |
| **MFCCs (1–20)** | 20 | Spectral shape, vocal/instrumental timbre envelope |
| **Spectral Centroid** | 1 | Perceived frequency center of mass ("brightness" vs "warmth") |
| **Spectral Rolloff** | 1 | 85% energy frequency threshold (high-frequency presence) |
| **Zero Crossing Rate** | 1 | Percussiveness, drum transients, noisiness |
| **RMS Energy** | 1 | Overall signal power, acoustic loudness |
| **Tempo (BPM)** | 1 | Rhythmic pace, pulse rate, danceability |
| **MFCC Variance** | 1 | Timbral stability vs textural dynamism |

---

## ⚡ Quickstart Guide

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/vibetuner-audio-dl.git
cd vibetuner-audio-dl
pip install -r requirements.txt
```

### 2. Launch the Streamlit Studio Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
vibetuner-audio-dl/
├── app.py              # Complete single-page Streamlit application & DSP engine
├── requirements.txt    # Production dependencies
└── README.md           # Documentation, architectural schematics & usage guide
```

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
