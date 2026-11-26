"""Unit tests for waterfall_extractor scientific signal processing algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from waterfall_extractor import (
    ConfigError,
    SignalInputError,
    WaterfallConfig,
    WaterfallExtractor,
    convert_to_dB,
    get_version_info,
)


def _make_tone_signal(fs: float = 20000.0, duration: float = 2.0, f0: float = 120.0) -> np.ndarray:
    t = np.arange(0.0, duration, 1.0 / fs)
    rng = np.random.default_rng(42)
    return np.sin(2.0 * np.pi * f0 * t) + 0.35 * np.sin(2.0 * np.pi * (3.0 * f0) * t) + 0.05 * rng.standard_normal(len(t))


def test_waterfall_dominant_frequency_accuracy() -> None:
    fs = 20000.0
    f0 = 120.0
    signal = _make_tone_signal(fs=fs, duration=2.0, f0=f0)
    config = WaterfallConfig(sampling_rate=fs, window_size=1024, overlap=0.5, nfft=4096)
    extractor = WaterfallExtractor(config)

    output = extractor.analyze_waterfall(signal)
    matrix_result = output["matrix_result"]
    summary = output["features"]["summary"]

    assert matrix_result["mode"] == "waterfall"
    assert matrix_result["matrix"].ndim == 2
    assert matrix_result["matrix"].shape == (len(matrix_result["freq_axis"]), len(matrix_result["time_axis"]))
    assert abs(summary["main_freq_mean"] - f0) < 6.0
    assert summary["rms_mean"] > 0.0
    assert np.isfinite(summary["spectral_centroid_mean"])


def test_spectral_entropy_distinguishes_tone_from_white_noise() -> None:
    fs = 20000.0
    rng = np.random.default_rng(123)
    t = np.arange(0.0, 2.0, 1.0 / fs)
    pure_tone = np.sin(2.0 * np.pi * 250.0 * t)
    white_noise = rng.standard_normal(len(t))

    extractor = WaterfallExtractor(WaterfallConfig(sampling_rate=fs, window_size=1024, overlap=0.5, nfft=2048))
    tone_summary = extractor.analyze_waterfall(pure_tone)["features"]["summary"]
    noise_summary = extractor.analyze_waterfall(white_noise)["features"]["summary"]

    assert tone_summary["entropy_mean"] < noise_summary["entropy_mean"]


def test_cascade_variable_speed_tracking() -> None:
    fs = 20000.0
    duration = 2.0
    t = np.arange(0.0, duration, 1.0 / fs)
    rpm_series = np.linspace(1000.0, 2000.0, len(t))
    freq_series = rpm_series / 60.0
    phase = np.cumsum(freq_series) / fs

    tacho = (np.sin(2.0 * np.pi * phase) > 0.0).astype(float)
    rng = np.random.default_rng(7)
    signal = np.sin(2.0 * np.pi * phase) + 0.3 * np.sin(4.0 * np.pi * phase) + 0.05 * rng.standard_normal(len(t))

    config = WaterfallConfig(
        sampling_rate=fs,
        window_size=1024,
        cascade_rpm_step=100.0,
        cascade_rpm_range=(900.0, 2200.0),
    )
    extractor = WaterfallExtractor(config)
    output = extractor.analyze_cascade(signal, tacho)

    matrix_result = output["matrix_result"]
    summary = output["features"]["summary"]
    assert matrix_result["mode"] == "cascade"
    assert len(matrix_result["rpm_axis"]) >= 5
    assert sum(matrix_result["frame_quality"]) > 0
    assert summary["mode"] == "cascade"
    assert np.isfinite(summary["main_freq_mean"])


def test_invalid_inputs_raise_structured_errors() -> None:
    with pytest.raises((ConfigError, ValueError)):
        WaterfallExtractor(WaterfallConfig(sampling_rate=-100.0))

    extractor = WaterfallExtractor(WaterfallConfig(sampling_rate=20000.0, window_size=1024))
    with pytest.raises(SignalInputError):
        extractor.build_waterfall(np.array([1.0, 2.0, 3.0]))


def test_convert_to_db_and_version_metadata() -> None:
    arr = np.array([[1.0, 10.0], [0.0, 100.0]])
    db = convert_to_dB(arr)
    assert db.shape == arr.shape
    assert np.all(np.isfinite(db))

    info = get_version_info()
    assert "package_version" in info
    assert "algorithm_id" in info
