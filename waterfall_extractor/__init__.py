"""频谱瀑布图/级联图特征提取封装包。

提供从振动信号中构建瀑布图/级联图矩阵并提取频域特征的核心算法。
"""

from .waterfall_core import (
    # 配置类
    WaterfallConfig,

    # 异常类
    WaterfallExtractorError,
    ConfigError,
    SignalInputError,
    ComputationError,

    # 高级封装类
    WaterfallExtractor,

    # 核心函数
    build_waterfall_matrix,
    build_cascade_matrix,
    extract_features,
    convert_to_dB,
    tacho_to_rpm,

    # 信号处理函数
    compute_fft_spectrum,
    compute_stft,
    bandpass_filter,
    compute_envelope,
    compute_envelope_spectrum,

    # 特征提取函数
    calc_main_frequency,
    calc_spectral_entropy,
    calc_spectral_kurtosis_from_matrix,
    calc_time_domain_features,
    calc_frequency_domain_features,
    calc_envelope_features,
)
from .version import (
    PACKAGE_VERSION,
    ALGORITHM_ID,
    ALGORITHM_VERSION,
    FEATURE_SCHEMA_VERSION,
    get_version_info,
)

__version__ = PACKAGE_VERSION

__all__ = [
    # 配置类
    "WaterfallConfig",
    # 异常类
    "WaterfallExtractorError",
    "ConfigError",
    "SignalInputError",
    "ComputationError",
    # 高级封装类
    "WaterfallExtractor",
    # 核心函数
    "build_waterfall_matrix",
    "build_cascade_matrix",
    "extract_features",
    "convert_to_dB",
    "tacho_to_rpm",
    # 信号处理函数
    "compute_fft_spectrum",
    "compute_stft",
    "bandpass_filter",
    "compute_envelope",
    "compute_envelope_spectrum",
    # 特征提取函数
    "calc_main_frequency",
    "calc_spectral_entropy",
    "calc_spectral_kurtosis_from_matrix",
    "calc_time_domain_features",
    "calc_frequency_domain_features",
    "calc_envelope_features",
    # 版本信息
    "PACKAGE_VERSION",
    "ALGORITHM_ID",
    "ALGORITHM_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "get_version_info",
]
