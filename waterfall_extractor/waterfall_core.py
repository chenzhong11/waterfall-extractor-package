"""频谱瀑布图/级联图核心算法模块。

合并自：
- signal_processor.py（信号处理基础）
- waterfall_core.py（瀑布图/级联图矩阵构建）
- feature_extractor.py（特征提取）

Author: 位豪
Date: 2026-06-29
Version: V1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import (
    butter,
    filtfilt,
    hilbert,
    spectrogram,
    stft as scipy_stft,
    windows,
)
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

from .version import (
    ALGORITHM_ID,
    ALGORITHM_VERSION,
    CREATOR,
    FEATURE_SCHEMA_VERSION,
    LAST_EDIT_DATE,
    PACKAGE_VERSION,
    get_version_info,
)


# ============================================================
# 第一部分：异常定义
# ============================================================

class WaterfallExtractorError(Exception):
    """封装包异常基类，便于上层程序统一捕获本包抛出的错误。"""


class ConfigError(ValueError, WaterfallExtractorError):
    """配置参数非法时抛出。"""


class SignalInputError(ValueError, WaterfallExtractorError):
    """输入信号非法时抛出。"""


class ComputationError(RuntimeError, WaterfallExtractorError):
    """计算过程异常时抛出。"""


# ============================================================
# 第二部分：配置类
# ============================================================

@dataclass(frozen=True)
class WaterfallConfig:
    """频谱瀑布图/级联图特征提取配置。

    Attributes:
        sampling_rate: 采样率 Hz
        window_size: FFT 窗长（采样点数）
        overlap: 重叠率 0~1
        nfft: FFT 点数（零填充提高频率分辨率）
        window_type: 窗函数类型
        cascade_rpm_step: 级联图转速步长 rpm
        cascade_rpm_range: 级联图转速范围 (min, max) rpm
        rpm_spread_threshold: 窗口内转速波动阈值 rpm
        device_type: 设备类型（pump/universal）
        has_tachometer: 是否有转速信号
        rotational_freq_hz: 转频 Hz
        key_orders: 关键阶次列表
        harmonic_count: 谐波数量
        order_bandwidth_hz: 阶次搜索带宽 Hz
        envelope_bandpass: 包络分析带通范围 Hz
    """

    sampling_rate: float = 20000.0
    window_size: int = 1024
    overlap: float = 0.5
    nfft: int = 4096
    window_type: str = "hann"
    cascade_rpm_step: float = 25.0
    cascade_rpm_range: Tuple[float, float] = (500.0, 5000.0)
    rpm_spread_threshold: float = 20.0
    device_type: str = "universal"
    has_tachometer: bool = False
    rotational_freq_hz: float = 24.67
    key_orders: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.57, 5.43, 7.0, 14.0])
    harmonic_count: int = 5
    order_bandwidth_hz: float = 2.0
    envelope_bandpass: Tuple[float, float] = (500.0, 5000.0)

    def validate(self) -> None:
        """校验配置参数。"""
        if self.sampling_rate <= 0:
            raise ConfigError("sampling_rate must be greater than 0.")
        if self.window_size <= 0:
            raise ConfigError("window_size must be greater than 0.")
        if not 0 <= self.overlap < 1:
            raise ConfigError("overlap must be in [0, 1).")
        if self.nfft <= 0:
            raise ConfigError("nfft must be greater than 0.")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于传递给内部函数。"""
        return {
            "sampling_rate": self.sampling_rate,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "nfft": self.nfft,
            "window_type": self.window_type,
            "cascade_rpm_step": self.cascade_rpm_step,
            "cascade_rpm_range": self.cascade_rpm_range,
            "rpm_spread_threshold": self.rpm_spread_threshold,
            "device_type": self.device_type,
            "has_tachometer": self.has_tachometer,
            "rotational_freq_hz": self.rotational_freq_hz,
            "key_orders": self.key_orders,
            "harmonic_count": self.harmonic_count,
            "order_bandwidth_hz": self.order_bandwidth_hz,
            "envelope_bandpass": self.envelope_bandpass,
        }


# ============================================================
# 第三部分：信号处理基础函数
# ============================================================

def compute_fft_spectrum(signal_data: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """计算单边FFT频谱。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-02
    输入参数:
        signal_data，时域振动信号；
        sampling_rate，采样率 Hz。
    输出参数:
        freqs，频率轴 Hz；
        magnitudes，幅值谱（单边，已归一化）。
    """
    n = len(signal_data)
    fft_result = np.fft.rfft(signal_data)
    freqs = np.fft.rfftfreq(n, 1.0 / sampling_rate)
    magnitudes = np.abs(fft_result) / n * 2
    return freqs, magnitudes


def compute_stft(
    signal_data: np.ndarray,
    sampling_rate: float,
    window_size: int = 1024,
    overlap_ratio: float = 0.5,
    nfft: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算短时傅里叶变换（STFT），得到时频谱。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-02
    输入参数:
        signal_data，时域振动信号；
        sampling_rate，采样率 Hz；
        window_size，窗长（采样点数）；
        overlap_ratio，重叠率 0~1；
        nfft，FFT点数。
    输出参数:
        freqs，频率轴 Hz；
        times，时间轴 s；
        stft_magnitude，STFT幅值矩阵 (freq_bins x time_frames)。
    """
    noverlap = int(window_size * overlap_ratio)
    freqs, times, Zxx = scipy_stft(
        signal_data, fs=sampling_rate,
        window="hann", nperseg=window_size,
        noverlap=noverlap, nfft=nfft,
    )
    stft_magnitude = np.abs(Zxx)
    return freqs, times, stft_magnitude


def bandpass_filter(
    signal_data: np.ndarray,
    sampling_rate: float,
    low_freq: float,
    high_freq: float,
    order: int = 4,
) -> np.ndarray:
    """带通滤波。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-02
    输入参数:
        signal_data，时域振动信号；
        sampling_rate，采样率 Hz；
        low_freq，低截止频率 Hz；
        high_freq，高截止频率 Hz；
        order，滤波器阶数。
    输出参数: 滤波后的信号。
    """
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal_data)


def compute_envelope(
    signal_data: np.ndarray,
    sampling_rate: float,
    bandpass_range: Tuple[float, float] = (500.0, 5000.0),
) -> np.ndarray:
    """计算包络信号：带通滤波 + Hilbert变换。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-02
    输入参数:
        signal_data，时域振动信号；
        sampling_rate，采样率 Hz；
        bandpass_range，带通滤波频率范围 (low, high) Hz。
    输出参数: 包络信号。
    """
    filtered = bandpass_filter(signal_data, sampling_rate, bandpass_range[0], bandpass_range[1])
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)
    envelope = envelope - np.mean(envelope)
    return envelope


def compute_envelope_spectrum(
    signal_data: np.ndarray,
    sampling_rate: float,
    bandpass_range: Tuple[float, float] = (500.0, 5000.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """计算包络谱：对包络信号做FFT。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-02
    输入参数:
        signal_data，时域振动信号；
        sampling_rate，采样率 Hz；
        bandpass_range，带通滤波频率范围 Hz。
    输出参数:
        env_freqs，频率轴；
        env_magnitudes，包络谱幅值。
    """
    envelope = compute_envelope(signal_data, sampling_rate, bandpass_range)
    env_freqs, env_magnitudes = compute_fft_spectrum(envelope, sampling_rate)
    return env_freqs, env_magnitudes


# ============================================================
# 第四部分：瀑布图/级联图核心函数
# ============================================================

def tacho_to_rpm(
    tacho_signal: np.ndarray,
    fs: float,
    threshold: Optional[float] = None,
    ppr: int = 1,
    smooth_window: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """将转速脉冲信号转换为RPM时间序列。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        tacho_signal，转速脉冲信号；
        fs，采样率 Hz；
        threshold，脉冲检测阈值（None则自动取中位数）；
        ppr，每转脉冲数；
        smooth_window，平滑滤波窗口大小。
    输出参数:
        time_axis，时间轴 s；
        rpm_timeseries，与tacho_signal等长的RPM序列。
    """
    if threshold is None:
        threshold = (np.max(tacho_signal) + np.min(tacho_signal)) / 2

    above = tacho_signal > threshold
    rising_edges = np.where(np.diff(above.astype(int)) == 1)[0]

    if len(rising_edges) < 2:
        time_axis = np.arange(len(tacho_signal)) / fs
        return time_axis, np.full(len(tacho_signal), 1500.0)

    pulse_times = rising_edges / fs
    intervals = np.diff(pulse_times)
    rpm_raw = 60.0 / (intervals * ppr)

    median_rpm = np.median(rpm_raw)
    valid = (rpm_raw > median_rpm * 0.3) & (rpm_raw < median_rpm * 3.0)
    rpm_raw = rpm_raw[valid]
    pulse_times_valid = pulse_times[1:][valid]

    if len(rpm_raw) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        rpm_smooth = np.convolve(rpm_raw, kernel, mode="same")
    else:
        rpm_smooth = rpm_raw

    time_axis = np.arange(len(tacho_signal)) / fs
    rpm_timeseries = np.interp(time_axis, pulse_times_valid, rpm_smooth)

    return time_axis, rpm_timeseries


def build_waterfall_matrix(
    signal_data: np.ndarray,
    fs: float,
    window_size: int = 1024,
    overlap: float = 0.5,
    nfft: int = 4096,
    window_type: str = "hann",
) -> Dict[str, Any]:
    """构建瀑布图矩阵（跟踪时间模式）。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        signal_data，时域振动信号 (1D)；
        fs，采样率 Hz；
        window_size，FFT窗长（采样点数）；
        overlap，重叠率 0~1；
        nfft，FFT点数；
        window_type，窗函数类型。
    输出参数: 包含 matrix、freq_axis、time_axis、mode、metadata 的字典。
    """
    arr = np.asarray(signal_data, dtype=float)
    if arr.ndim != 1:
        raise SignalInputError(f"Expected 1D vibration signal, got shape {arr.shape}")
    if arr.size < window_size:
        raise SignalInputError(
            f"Signal length ({arr.size}) must be >= window_size ({window_size})"
        )
    if not np.all(np.isfinite(arr)):
        raise SignalInputError("Input signal contains NaN or infinite values")

    noverlap = int(window_size * overlap)

    freqs, times, Sxx = spectrogram(
        arr, fs=fs,
        window=window_type,
        nperseg=window_size,
        noverlap=noverlap,
        nfft=nfft,
        scaling="spectrum",
    )

    magnitude = np.sqrt(Sxx)

    metadata = {
        "sampling_rate": fs,
        "signal_length": len(signal_data),
        "duration_s": len(signal_data) / fs,
        "window_size": window_size,
        "window_type": window_type,
        "overlap_ratio": overlap,
        "nfft": nfft,
        "freq_resolution": fs / nfft,
        "num_frames": len(times),
        "num_freq_bins": len(freqs),
    }

    return {
        "matrix": magnitude,
        "freq_axis": freqs,
        "time_axis": times,
        "mode": "waterfall",
        "metadata": metadata,
    }


def build_cascade_matrix(
    signal_data: np.ndarray,
    rpm_timeseries: np.ndarray,
    fs: float,
    rpm_step: float = 25.0,
    rpm_range: Tuple[float, float] = (500.0, 5000.0),
    window_size: int = 1024,
    window_type: str = "hann",
    rpm_spread_threshold: float = 20.0,
) -> Dict[str, Any]:
    """构建级联图矩阵（跟踪转速模式）。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        signal_data，时域振动信号 (1D)；
        rpm_timeseries，与signal_data等长的转速序列 RPM；
        fs，采样率 Hz；
        rpm_step，转速步长 rpm；
        rpm_range，转速范围 (min, max) rpm；
        window_size，每帧FFT窗长；
        window_type，窗函数类型；
        rpm_spread_threshold，窗口内转速波动阈值 rpm。
    输出参数: 包含 matrix、rpm_axis、freq_axis、mode、frame_quality、metadata 的字典。
    """
    rpm_min, rpm_max = rpm_range
    actual_min = max(rpm_min, int(np.min(rpm_timeseries)))
    actual_max = min(rpm_max, int(np.max(rpm_timeseries)))
    rpm_bins = np.arange(actual_min, actual_max, rpm_step)

    freqs = np.fft.rfftfreq(window_size, 1.0 / fs)
    freq_bins = len(freqs)

    cascade = []
    frame_quality = []
    window = windows.get_window(window_type, window_size)

    for rpm_target in rpm_bins:
        center_idx = np.argmin(np.abs(rpm_timeseries - rpm_target))
        start = max(0, center_idx - window_size // 2)
        end = min(len(signal_data), start + window_size)

        segment = signal_data[start:end]
        if len(segment) < window_size:
            segment = np.pad(segment, (0, window_size - len(segment)))

        segment_rpm = rpm_timeseries[start:end]
        rpm_spread = np.max(segment_rpm) - np.min(segment_rpm)
        rpm_median = np.median(segment_rpm)
        relative_spread = rpm_spread / (rpm_median + 1e-10)
        quality_ok = relative_spread < 0.15
        frame_quality.append(quality_ok)

        segment = segment - np.mean(segment)
        windowed = segment * window
        spectrum = np.abs(np.fft.rfft(windowed))
        cascade.append(spectrum[:freq_bins])

    cascade_matrix = np.array(cascade)

    good_frames = sum(frame_quality)
    total_frames = len(frame_quality)
    metadata = {
        "sampling_rate": fs,
        "signal_length": len(signal_data),
        "duration_s": len(signal_data) / fs,
        "window_size": window_size,
        "window_type": window_type,
        "rpm_step": rpm_step,
        "rpm_range": (actual_min, actual_max),
        "rpm_spread_threshold": rpm_spread_threshold,
        "num_frames": total_frames,
        "good_frames": good_frames,
        "bad_frames": total_frames - good_frames,
        "frame_quality_ratio": good_frames / total_frames if total_frames > 0 else 0,
        "num_freq_bins": freq_bins,
        "freq_resolution": fs / window_size,
    }

    return {
        "matrix": cascade_matrix,
        "rpm_axis": rpm_bins,
        "freq_axis": freqs,
        "mode": "cascade",
        "frame_quality": frame_quality,
        "metadata": metadata,
    }


def convert_to_dB(matrix: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """将幅值矩阵转换为dB尺度。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        matrix，幅值矩阵；
        floor_db，动态范围下限 dB。
    输出参数: dB幅值矩阵。
    """
    max_val = np.max(matrix)
    if max_val == 0:
        return np.full_like(matrix, floor_db)
    matrix_dB = 20 * np.log10(matrix / max_val + 1e-10)
    matrix_dB = np.maximum(matrix_dB, floor_db)
    return matrix_dB


# ============================================================
# 第五部分：特征提取函数
# ============================================================

def _get_frames(matrix: np.ndarray, mode: str) -> List[np.ndarray]:
    """统一获取帧列表。级联图: 行=帧；瀑布图: 列=帧。"""
    if mode == "cascade":
        return [matrix[i, :] for i in range(matrix.shape[0])]
    else:
        return [matrix[:, i] for i in range(matrix.shape[1])]


def calc_main_frequency(matrix: np.ndarray, freq_axis: np.ndarray, mode: str) -> Dict[str, Any]:
    """每帧能量最大频率。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        matrix，幅值谱矩阵；
        freq_axis，频率轴；
        mode，模式（waterfall/cascade）。
    输出参数: 包含 main_freq_series、main_freq_mean、main_freq_std 的字典。
    """
    frames = _get_frames(matrix, mode)
    peak_indices = np.argmax(frames, axis=1)
    main_freqs = freq_axis[peak_indices]
    return {
        "main_freq_series": main_freqs,
        "main_freq_mean": float(np.mean(main_freqs)),
        "main_freq_std": float(np.std(main_freqs)),
    }


def calc_spectral_entropy(matrix: np.ndarray, freq_axis: np.ndarray, mode: str) -> Dict[str, Any]:
    """Shannon 熵在功率谱上的应用。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        matrix，幅值谱矩阵；
        freq_axis，频率轴；
        mode，模式（waterfall/cascade）。
    输出参数: 包含 entropy_series、entropy_mean、entropy_std 的字典。
    """
    frames = _get_frames(matrix, mode)
    entropy_per_frame = []
    for frame in frames:
        power = frame ** 2
        total = np.sum(power)
        if total > 0:
            p = power / total
            p = p[p > 0]
            entropy = -np.sum(p * np.log2(p))
        else:
            entropy = 0.0
        entropy_per_frame.append(entropy)
    entropy_arr = np.array(entropy_per_frame)
    return {
        "entropy_series": entropy_arr,
        "entropy_mean": float(np.mean(entropy_arr)),
        "entropy_std": float(np.std(entropy_arr)),
    }


def calc_spectral_kurtosis_from_matrix(
    matrix: np.ndarray, freq_axis: np.ndarray, mode: str
) -> Dict[str, Any]:
    """谱峭度：对每个频率 bin，计算其在所有帧上的幅值峭度。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        matrix，幅值谱矩阵；
        freq_axis，频率轴；
        mode，模式（waterfall/cascade）。
    输出参数: 包含 kurtosis_series、kurtosis_map、kurtosis_mean、kurtosis_std、kurtosis_peak_freq 的字典。
    """
    frames = _get_frames(matrix, mode)

    kurtosis_per_frame = []
    for frame in frames:
        if np.std(frame) > 0:
            k = float(np.mean((frame - np.mean(frame)) ** 4) / (np.std(frame) ** 4))
        else:
            k = 0.0
        kurtosis_per_frame.append(k)
    kurtosis_arr = np.array(kurtosis_per_frame)

    if mode == "cascade":
        kurtosis_map = np.zeros(matrix.shape[1])
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            if np.std(col) > 0:
                kurtosis_map[j] = float(np.mean((col - np.mean(col)) ** 4) / (np.std(col) ** 4))
    else:
        kurtosis_map = np.zeros(matrix.shape[0])
        for j in range(matrix.shape[0]):
            row = matrix[j, :]
            if np.std(row) > 0:
                kurtosis_map[j] = float(np.mean((row - np.mean(row)) ** 4) / (np.std(row) ** 4))

    peak_idx = np.argmax(kurtosis_map)
    peak_freq = float(freq_axis[peak_idx])

    return {
        "kurtosis_series": kurtosis_arr,
        "kurtosis_map": kurtosis_map,
        "kurtosis_mean": float(np.mean(kurtosis_arr)),
        "kurtosis_std": float(np.std(kurtosis_arr)),
        "kurtosis_peak_freq": peak_freq,
    }


def calc_time_domain_features(matrix: np.ndarray, mode: str) -> Dict[str, Any]:
    """时域统计特征：RMS/峰值/峰峰值/波峰因子/形状因子/脉冲因子/偏度/峭度因子。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        matrix，幅值谱矩阵；
        mode，模式（waterfall/cascade）。
    输出参数: 包含各特征时序序列和统计量的字典。
    """
    frames = _get_frames(matrix, mode)

    rms_series, peak_series, pp_series = [], [], []
    cf_series, sf_series, impf_series = [], [], []
    skew_series, kurt_series = [], []

    for frame in frames:
        abs_frame = np.abs(frame)
        mean_abs = np.mean(abs_frame)
        rms = np.sqrt(np.mean(frame ** 2))
        peak = np.max(abs_frame)
        pp = np.max(frame) - np.min(frame)

        cf = peak / rms if rms > 0 else 0.0
        sf = rms / mean_abs if mean_abs > 0 else 0.0
        impf = peak / mean_abs if mean_abs > 0 else 0.0

        if np.std(frame) > 0:
            skew = float(scipy_skew(frame))
            kurt = float(scipy_kurtosis(frame, fisher=False))
        else:
            skew, kurt = 0.0, 0.0

        rms_series.append(rms)
        peak_series.append(peak)
        pp_series.append(pp)
        cf_series.append(cf)
        sf_series.append(sf)
        impf_series.append(impf)
        skew_series.append(skew)
        kurt_series.append(kurt)

    rms_arr = np.array(rms_series)
    peak_arr = np.array(peak_series)
    pp_arr = np.array(pp_series)
    cf_arr = np.array(cf_series)
    sf_arr = np.array(sf_series)
    impf_arr = np.array(impf_series)
    skew_arr = np.array(skew_series)
    kurt_arr = np.array(kurt_series)

    return {
        "rms_series": rms_arr,
        "rms_mean": float(np.mean(rms_arr)),
        "rms_std": float(np.std(rms_arr)),
        "rms_max": float(np.max(rms_arr)),
        "peak_series": peak_arr,
        "peak_mean": float(np.mean(peak_arr)),
        "peak_std": float(np.std(peak_arr)),
        "pp_series": pp_arr,
        "pp_mean": float(np.mean(pp_arr)),
        "crest_factor_series": cf_arr,
        "crest_factor_mean": float(np.mean(cf_arr)),
        "crest_factor_std": float(np.std(cf_arr)),
        "crest_factor_max": float(np.max(cf_arr)),
        "shape_factor_series": sf_arr,
        "shape_factor_mean": float(np.mean(sf_arr)),
        "impulse_factor_series": impf_arr,
        "impulse_factor_mean": float(np.mean(impf_arr)),
        "impulse_factor_max": float(np.max(impf_arr)),
        "skewness_series": skew_arr,
        "skewness_mean": float(np.mean(skew_arr)),
        "skewness_std": float(np.std(skew_arr)),
        "kurtosis_factor_series": kurt_arr,
        "kurtosis_factor_mean": float(np.mean(kurt_arr)),
        "kurtosis_factor_std": float(np.std(kurt_arr)),
        "kurtosis_factor_max": float(np.max(kurt_arr)),
    }


def calc_frequency_domain_features(
    matrix: np.ndarray, freq_axis: np.ndarray, config: Dict[str, Any], mode: str
) -> Dict[str, Any]:
    """频域增强特征：谐波能量比/亚同步能量比/频带能量/频谱重心/频谱展宽/频谱滚降点。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        matrix，幅值谱矩阵；
        freq_axis，频率轴；
        config，配置字典；
        mode，模式（waterfall/cascade）。
    输出参数: 包含各特征时序序列和统计量的字典。
    """
    frames = _get_frames(matrix, mode)

    device_type = config.get("device_type", "universal")
    has_tachometer = config.get("has_tachometer", False)
    key_orders = config.get("key_orders", [1.0, 2.0, 3.57, 5.43, 7.0, 14.0])
    harmonic_count = config.get("harmonic_count", 5)
    bandwidth_hz = config.get("order_bandwidth_hz", 2.0)

    centroid_series, spread_series, rolloff_series = [], [], []
    band_low_series, band_mid_series, band_high_series = [], [], []
    harmonic_ratio_series, sub_sync_ratio_series = [], []

    for frame in frames:
        power = frame ** 2
        total_power = np.sum(power)

        if total_power > 0:
            p_norm = power / total_power

            centroid = np.sum(freq_axis * p_norm)
            centroid_series.append(centroid)

            spread = np.sqrt(np.sum(((freq_axis - centroid) ** 2) * p_norm))
            spread_series.append(spread)

            cumsum = np.cumsum(p_norm)
            rolloff_idx = min(np.searchsorted(cumsum, 0.85), len(freq_axis) - 1)
            rolloff_series.append(freq_axis[rolloff_idx])

            low_mask = freq_axis < 500
            mid_mask = (freq_axis >= 500) & (freq_axis < 2000)
            high_mask = freq_axis >= 2000

            band_low_series.append(np.sum(power[low_mask]) / total_power if np.any(low_mask) else 0)
            band_mid_series.append(np.sum(power[mid_mask]) / total_power if np.any(mid_mask) else 0)
            band_high_series.append(np.sum(power[high_mask]) / total_power if np.any(high_mask) else 0)

            if device_type == "pump" and not has_tachometer:
                base_freq = config.get("rotational_freq_hz", 24.67)
            else:
                base_freq = freq_axis[np.argmax(frame)]

            harmonic_power = 0
            for order_idx in range(1, min(harmonic_count + 1, 6)):
                target_freq = base_freq * order_idx
                mask = np.abs(freq_axis - target_freq) <= bandwidth_hz
                if np.any(mask):
                    harmonic_power += np.sum(power[mask])
            harmonic_ratio_series.append(min(harmonic_power / total_power, 1.0))

            if base_freq > 0:
                sub_sync_mask = (freq_axis > 0) & (freq_axis < base_freq)
                sub_sync_ratio = np.sum(power[sub_sync_mask]) / total_power if np.any(sub_sync_mask) else 0.0
            else:
                sub_sync_ratio = 0.0
            sub_sync_ratio_series.append(sub_sync_ratio)
        else:
            centroid_series.append(0)
            spread_series.append(0)
            rolloff_series.append(0)
            band_low_series.append(0)
            band_mid_series.append(0)
            band_high_series.append(0)
            harmonic_ratio_series.append(0)
            sub_sync_ratio_series.append(0)

    centroid_arr = np.array(centroid_series)
    spread_arr = np.array(spread_series)
    rolloff_arr = np.array(rolloff_series)
    band_low_arr = np.array(band_low_series)
    band_mid_arr = np.array(band_mid_series)
    band_high_arr = np.array(band_high_series)
    harmonic_arr = np.array(harmonic_ratio_series)
    sub_sync_arr = np.array(sub_sync_ratio_series)

    return {
        "spectral_centroid_series": centroid_arr,
        "spectral_centroid_mean": float(np.mean(centroid_arr)),
        "spectral_centroid_std": float(np.std(centroid_arr)),
        "spectral_spread_series": spread_arr,
        "spectral_spread_mean": float(np.mean(spread_arr)),
        "spectral_rolloff_series": rolloff_arr,
        "spectral_rolloff_mean": float(np.mean(rolloff_arr)),
        "band_energy_low_series": band_low_arr,
        "band_energy_low_mean": float(np.mean(band_low_arr)),
        "band_energy_mid_series": band_mid_arr,
        "band_energy_mid_mean": float(np.mean(band_mid_arr)),
        "band_energy_high_series": band_high_arr,
        "band_energy_high_mean": float(np.mean(band_high_arr)),
        "harmonic_ratio_series": harmonic_arr,
        "harmonic_ratio_mean": float(np.mean(harmonic_arr)),
        "harmonic_ratio_std": float(np.std(harmonic_arr)),
        "sub_sync_ratio_series": sub_sync_arr,
        "sub_sync_ratio_mean": float(np.mean(sub_sync_arr)),
        "sub_sync_ratio_std": float(np.std(sub_sync_arr)),
    }


def calc_envelope_features(
    matrix: np.ndarray, freq_axis: np.ndarray, mode: str
) -> Dict[str, Any]:
    """包络谱特征：包络谱峰值/包络谱主频。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        matrix，幅值谱矩阵；
        freq_axis，频率轴；
        mode，模式（waterfall/cascade）。
    输出参数: 包含各特征时序序列和统计量的字典。
    """
    frames = _get_frames(matrix, mode)

    env_peak_series, env_main_freq_series = [], []

    for frame in frames:
        peak_idx = np.argmax(frame)
        env_peak_series.append(float(frame[peak_idx]))
        env_main_freq_series.append(float(freq_axis[peak_idx]))

    env_peak_arr = np.array(env_peak_series)
    env_freq_arr = np.array(env_main_freq_series)

    return {
        "envelope_peak_series": env_peak_arr,
        "envelope_peak_mean": float(np.mean(env_peak_arr)),
        "envelope_peak_std": float(np.std(env_peak_arr)),
        "envelope_main_freq_series": env_freq_arr,
        "envelope_main_freq_mean": float(np.mean(env_freq_arr)),
        "envelope_main_freq_std": float(np.std(env_freq_arr)),
    }


def extract_features(
    matrix_result: Dict[str, Any],
    config: Optional[WaterfallConfig] = None,
) -> Dict[str, Any]:
    """统一特征提取入口。

    函数版本: F-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数:
        matrix_result，build_waterfall_matrix 或 build_cascade_matrix 的返回值；
        config，可选 WaterfallConfig 配置对象。
    输出参数: 包含 summary、timeseries 和各组特征的字典。
    """
    matrix = matrix_result["matrix"]
    freq_axis = matrix_result["freq_axis"]
    mode = matrix_result.get("mode", "waterfall")

    if config is None:
        config = WaterfallConfig()
    config_dict = config.to_dict()

    main_freq = calc_main_frequency(matrix, freq_axis, mode)
    entropy = calc_spectral_entropy(matrix, freq_axis, mode)
    kurtosis = calc_spectral_kurtosis_from_matrix(matrix, freq_axis, mode)
    time_domain = calc_time_domain_features(matrix, mode)
    freq_domain = calc_frequency_domain_features(matrix, freq_axis, config_dict, mode)
    envelope = calc_envelope_features(matrix, freq_axis, mode)

    timeseries = {
        "main_freq": main_freq["main_freq_series"],
        "entropy": entropy["entropy_series"],
        "kurtosis": kurtosis["kurtosis_series"],
        "rms": time_domain["rms_series"],
        "crest_factor": time_domain["crest_factor_series"],
        "impulse_factor": time_domain["impulse_factor_series"],
        "skewness": time_domain["skewness_series"],
        "kurtosis_factor": time_domain["kurtosis_factor_series"],
        "spectral_centroid": freq_domain["spectral_centroid_series"],
        "harmonic_ratio": freq_domain["harmonic_ratio_series"],
        "band_energy_high": freq_domain["band_energy_high_series"],
        "envelope_peak": envelope["envelope_peak_series"],
    }

    summary = {
        "package_version": PACKAGE_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "mode": mode,
        "main_freq_mean": main_freq["main_freq_mean"],
        "main_freq_std": main_freq["main_freq_std"],
        "entropy_mean": entropy["entropy_mean"],
        "entropy_std": entropy["entropy_std"],
        "kurtosis_mean": kurtosis["kurtosis_mean"],
        "kurtosis_std": kurtosis["kurtosis_std"],
        "kurtosis_peak_freq": kurtosis["kurtosis_peak_freq"],
        "rms_mean": time_domain["rms_mean"],
        "rms_std": time_domain["rms_std"],
        "rms_max": time_domain["rms_max"],
        "crest_factor_mean": time_domain["crest_factor_mean"],
        "crest_factor_max": time_domain["crest_factor_max"],
        "impulse_factor_mean": time_domain["impulse_factor_mean"],
        "impulse_factor_max": time_domain["impulse_factor_max"],
        "skewness_mean": time_domain["skewness_mean"],
        "kurtosis_factor_mean": time_domain["kurtosis_factor_mean"],
        "kurtosis_factor_max": time_domain["kurtosis_factor_max"],
        "spectral_centroid_mean": freq_domain["spectral_centroid_mean"],
        "spectral_spread_mean": freq_domain["spectral_spread_mean"],
        "spectral_rolloff_mean": freq_domain["spectral_rolloff_mean"],
        "band_energy_low_mean": freq_domain["band_energy_low_mean"],
        "band_energy_mid_mean": freq_domain["band_energy_mid_mean"],
        "band_energy_high_mean": freq_domain["band_energy_high_mean"],
        "harmonic_ratio_mean": freq_domain["harmonic_ratio_mean"],
        "sub_sync_ratio_mean": freq_domain["sub_sync_ratio_mean"],
        "envelope_peak_mean": envelope["envelope_peak_mean"],
        "envelope_main_freq_mean": envelope["envelope_main_freq_mean"],
    }

    return {
        "summary": summary,
        "timeseries": timeseries,
        "main_freq": main_freq,
        "entropy": entropy,
        "kurtosis": kurtosis,
        "time_domain": time_domain,
        "freq_domain": freq_domain,
        "envelope": envelope,
    }


# ============================================================
# 第六部分：高级封装类
# ============================================================

class WaterfallExtractor:
    """频谱瀑布图/级联图特征提取器。

    函数版本: C-1.0.0
    创建人: 位豪
    编辑日期: 2026-06-29
    输入参数: config，WaterfallConfig 配置对象。
    输出参数: WaterfallExtractor 实例。

    使用示例:
        config = WaterfallConfig(sampling_rate=20000, window_size=1024)
        extractor = WaterfallExtractor(config)
        result = extractor.build_waterfall(signal)
        features = extractor.extract_features(result)
    """

    def __init__(self, config: Optional[WaterfallConfig] = None) -> None:
        if config is None:
            config = WaterfallConfig()
        config.validate()
        self._config = config

    @property
    def config(self) -> WaterfallConfig:
        """返回当前配置。"""
        return self._config

    def build_waterfall(self, signal: np.ndarray) -> Dict[str, Any]:
        """构建瀑布图矩阵（跟踪时间模式）。

        输入参数: signal，一维振动信号数组。
        输出参数: 包含 matrix、freq_axis、time_axis、mode、metadata 的字典。
        """
        signal = np.asarray(signal, dtype=float).reshape(-1)
        if signal.size == 0:
            raise SignalInputError("signal must not be empty.")
        return build_waterfall_matrix(
            signal, self._config.sampling_rate,
            self._config.window_size, self._config.overlap,
            self._config.nfft, self._config.window_type,
        )

    def build_cascade(self, signal: np.ndarray, tacho: np.ndarray) -> Dict[str, Any]:
        """构建级联图矩阵（跟踪转速模式）。

        输入参数:
            signal，一维振动信号数组；
            tacho，一维转速脉冲信号数组。
        输出参数: 包含 matrix、rpm_axis、freq_axis、mode、frame_quality、metadata 的字典。
        """
        signal = np.asarray(signal, dtype=float).reshape(-1)
        tacho = np.asarray(tacho, dtype=float).reshape(-1)
        if signal.size == 0:
            raise SignalInputError("signal must not be empty.")
        if tacho.size == 0:
            raise SignalInputError("tacho must not be empty.")
        if len(signal) != len(tacho):
            raise SignalInputError("signal and tacho must have the same length.")

        _, rpm_timeseries = tacho_to_rpm(tacho, self._config.sampling_rate)
        return build_cascade_matrix(
            signal, rpm_timeseries, self._config.sampling_rate,
            self._config.cascade_rpm_step, self._config.cascade_rpm_range,
            self._config.window_size, self._config.window_type,
            self._config.rpm_spread_threshold,
        )

    def extract_features(self, matrix_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取特征。

        输入参数: matrix_result，build_waterfall 或 build_cascade 的返回值。
        输出参数: 包含 summary、timeseries 和各组特征的字典。
        """
        return extract_features(matrix_result, self._config)

    def analyze_waterfall(self, signal: np.ndarray) -> Dict[str, Any]:
        """一步完成：构建瀑布图 + 提取特征。

        输入参数: signal，一维振动信号数组。
        输出参数: 包含 matrix_result 和 features 的字典。
        """
        matrix_result = self.build_waterfall(signal)
        features = self.extract_features(matrix_result)
        return {"matrix_result": matrix_result, "features": features}

    def analyze_cascade(self, signal: np.ndarray, tacho: np.ndarray) -> Dict[str, Any]:
        """一步完成：构建级联图 + 提取特征。

        输入参数:
            signal，一维振动信号数组；
            tacho，一维转速脉冲信号数组。
        输出参数: 包含 matrix_result 和 features 的字典。
        """
        matrix_result = self.build_cascade(signal, tacho)
        features = self.extract_features(matrix_result)
        return {"matrix_result": matrix_result, "features": features}
