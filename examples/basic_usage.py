"""瀑布图模式示例：从数组直接调用封装包。

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
    """演示瀑布图模式：无转速信号，跟踪时间。"""
    # 1. 生成合成信号（60Hz + 180Hz + 噪声）
    sample_rate = 20000.0
    duration = 2.0  # 秒
    t = np.arange(0, duration, 1.0 / sample_rate)
    signal = (
        np.sin(2 * np.pi * 60.0 * t)
        + 0.5 * np.sin(2 * np.pi * 180.0 * t)
        + 0.1 * np.random.randn(len(t))
    )

    # 2. 创建配置
    config = WaterfallConfig(
        sampling_rate=sample_rate,
        window_size=1024,
        overlap=0.5,
        nfft=4096,
    )

    # 3. 创建提取器
    extractor = WaterfallExtractor(config)

    # 4. 构建瀑布图矩阵
    result = extractor.build_waterfall(signal)
    print("mode:", result["mode"])
    print("matrix shape:", result["matrix"].shape)
    print("freq_axis range:", f"{result['freq_axis'][0]:.1f} - {result['freq_axis'][-1]:.1f} Hz")
    print("time_axis range:", f"{result['time_axis'][0]:.3f} - {result['time_axis'][-1]:.3f} s")
    print("metadata:", result["metadata"])

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
    print(f"crest_factor_mean: {summary['crest_factor_mean']:.4f}")
    print(f"spectral_centroid_mean: {summary['spectral_centroid_mean']:.2f} Hz")


if __name__ == "__main__":
    main()
