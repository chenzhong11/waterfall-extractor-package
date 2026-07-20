"""级联图模式示例：从数组直接调用封装包。

函数版本: F-1.0.0
创建人: 位豪
编辑日期: 2026-06-29
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 添加包路径
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from waterfall_extractor import WaterfallConfig, WaterfallExtractor


def main() -> None:
    """演示级联图模式：有转速信号，跟踪转速。"""
    # 1. 生成合成信号（带转速变化的振动信号）
    sample_rate = 20000.0
    duration = 2.0
    t = np.arange(0, duration, 1.0 / sample_rate)

    # 模拟转速从 1000 RPM 线性增加到 2000 RPM
    rpm_start, rpm_end = 1000.0, 2000.0
    rpm_series = np.linspace(rpm_start, rpm_end, len(t))
    freq_series = rpm_series / 60.0  # 转频

    # 生成转速脉冲信号（每转 1 个脉冲）
    ppr = 1
    phase = np.cumsum(freq_series) / sample_rate
    tacho_signal = np.sin(2 * np.pi * ppr * phase)
    # 转为方波脉冲
    tacho_signal = (tacho_signal > 0).astype(float)

    # 生成振动信号（1×转频 + 2×转频 + 噪声）
    signal = (
        np.sin(2 * np.pi * np.cumsum(freq_series) / sample_rate)
        + 0.3 * np.sin(2 * np.pi * 2 * np.cumsum(freq_series) / sample_rate)
        + 0.1 * np.random.randn(len(t))
    )

    # 2. 创建配置
    config = WaterfallConfig(
        sampling_rate=sample_rate,
        window_size=1024,
        cascade_rpm_step=100,
        cascade_rpm_range=(800.0, 2500.0),
    )

    # 3. 创建提取器
    extractor = WaterfallExtractor(config)

    # 4. 构建级联图矩阵
    result = extractor.build_cascade(signal, tacho_signal)
    print("mode:", result["mode"])
    print("matrix shape:", result["matrix"].shape)
    print("rpm_axis range:", f"{result['rpm_axis'][0]:.0f} - {result['rpm_axis'][-1]:.0f} RPM")
    print("freq_axis range:", f"{result['freq_axis'][0]:.1f} - {result['freq_axis'][-1]:.1f} Hz")
    print("good_frames:", sum(result["frame_quality"]), "/", len(result["frame_quality"]))

    # 5. 提取特征
    features = extractor.extract_features(result)
    summary = features["summary"]

    print("\n=== 特征汇总 ===")
    print(f"package_version: {summary['package_version']}")
    print(f"algorithm_id: {summary['algorithm_id']}")
    print(f"mode: {summary['mode']}")
    print(f"main_freq_mean: {summary['main_freq_mean']:.2f} Hz")
    print(f"entropy_mean: {summary['entropy_mean']:.4f}")
    print(f"kurtosis_mean: {summary['kurtosis_mean']:.4f}")
    print(f"rms_mean: {summary['rms_mean']:.4f}")


if __name__ == "__main__":
    main()
