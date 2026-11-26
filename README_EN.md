# waterfall-extractor

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

[中文文档](README.md) | **English**

**`waterfall-extractor`** is a lightweight, integration-oriented scientific Python package for **3D spectrum waterfall and speed-tracked cascade feature extraction** from high-frequency vibration signals in rotating and hydraulic machinery (e.g., axial piston pumps, bearings, and electro-hydraulic equipment).

## Key Features

- **Dual Analysis Modes**:
  - **Waterfall Mode (Time–Frequency)**: STFT-based spectral evolution for stationary or tachometer-free vibration recordings.
  - **Cascade Mode (RPM–Frequency)**: Tachometer-driven instantaneous RPM tracking and speed-binned cascade matrix construction for variable-speed run-up/run-down diagnostics.
- **Array-First Scientific Interface**: Pure `numpy` / `scipy` array inputs decoupled from proprietary file formats, enabling reproducible integration into PHM pipelines and ML benchmarks.
- **Physics-Interpretable Feature Suite**: Extracts dominant frequency, spectral entropy, spectral kurtosis, spectral centroid, harmonic energy ratio, RMS, crest factor, and Hilbert envelope spectrum peaks as both summary descriptors and frame-wise time series.
- **Structured Exception Hierarchy**: Explicit validation via `ConfigError`, `SignalInputError`, and `ComputationError`.

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Time–Frequency Waterfall Mode

```python
import numpy as np
from waterfall_extractor import WaterfallConfig, WaterfallExtractor

fs = 20000.0
t = np.arange(0, 2.0, 1.0 / fs)
signal = np.sin(2 * np.pi * 60.0 * t) + 0.5 * np.sin(2 * np.pi * 180.0 * t)

config = WaterfallConfig(sampling_rate=fs, window_size=1024, overlap=0.5, nfft=4096)
extractor = WaterfallExtractor(config)

output = extractor.analyze_waterfall(signal)
summary = output["features"]["summary"]
print(summary["main_freq_mean"], summary["spectral_centroid_mean"], summary["rms_mean"])
```

### 2. Variable-Speed Cascade Mode

```python
result = extractor.analyze_cascade(signal, tacho_signal)
print(result["matrix_result"]["rpm_axis"], result["features"]["summary"])
```

## Extracted Diagnostic Features

| Feature Key | Physical Interpretation |
|---|---|
| `main_freq_mean` | Mean dominant frequency across frames (Hz) |
| `entropy_mean` | Mean normalized spectral entropy (tone vs. broadband noise) |
| `kurtosis_mean` / `kurtosis_peak_freq` | Spectral kurtosis and peak-impulsiveness frequency (Hz) |
| `rms_mean` / `crest_factor_mean` | Vibration energy level and impulsive peak-to-RMS ratio |
| `spectral_centroid_mean` | Center of mass of the power spectrum (Hz) |
| `harmonic_ratio_mean` | Energy concentration in rotational harmonics |
| `envelope_peak_mean` | Demodulated Hilbert envelope peak indicator for cyclic impacts |

## Running Tests

```bash
pytest -q
```

## Author & Maintainer

- **Hao Wei** — School of Mechanical Engineering, Yanshan University (Hydraulic & Rotating Machinery PHM)
- License: [MIT](LICENSE)
