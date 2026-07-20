# 频谱瀑布图/级联图特征提取封装包

`waterfall_extractor` 是一个面向工程集成的频谱瀑布图/级联图特征提取包。

核心特点：
- **统一入口**：自动判断有无转速信号，自适应选择分析路径
- **数组输入优先**：核心算法只接收一维数组，文件读取由调用方负责
- **特征丰富**：3 类核心特征 + 时域统计 + 频域增强 + 包络谱特征

## 当前版本

- 包版本：`1.0.0`
- 算法编号：`spectrum_waterfall_v1`
- 算法版本：`A-1.0.0`
- 特征字段版本：`2026.07`

## 安装

```bash
cd waterfall_extractor
pip install -e .
```

## 推荐集成方式

### 瀑布图模式（无转速信号）

```python
from waterfall_extractor import WaterfallConfig, WaterfallExtractor
import numpy as np

signal = np.loadtxt("vibration_data.txt")  # 一维振动信号

config = WaterfallConfig(
    sampling_rate=20000.0,
    window_size=1024,
    overlap=0.5,
    nfft=4096,
)

extractor = WaterfallExtractor(config)
result = extractor.build_waterfall(signal)
features = extractor.extract_features(result)

print(features["summary"]["main_freq_mean"])
print(features["summary"]["entropy_mean"])
print(features["summary"]["rms_mean"])
```

### 级联图模式（有转速信号）

```python
from waterfall_extractor import WaterfallConfig, WaterfallExtractor
import numpy as np

signal = np.loadtxt("vibration_data.txt")    # 一维振动信号
tacho = np.loadtxt("tacho_pulse.txt")        # 一维转速脉冲信号

config = WaterfallConfig(
    sampling_rate=20000.0,
    window_size=1024,
    cascade_rpm_step=25,
    cascade_rpm_range=(500, 5000),
)

extractor = WaterfallExtractor(config)
result = extractor.build_cascade(signal, tacho)
features = extractor.extract_features(result)

print(features["summary"]["main_freq_mean"])
print(features["summary"]["kurtosis_peak_freq"])
```

### 一步分析

```python
# 瀑布图模式
result = extractor.analyze_waterfall(signal)
matrix_result = result["matrix_result"]
features = result["features"]

# 级联图模式
result = extractor.analyze_cascade(signal, tacho)
```

## 异常处理

```python
from waterfall_extractor import (
    WaterfallConfig,
    WaterfallExtractor,
    WaterfallExtractorError,
    ConfigError,
    SignalInputError,
    ComputationError,
)

try:
    result = extractor.build_waterfall(signal)
except ConfigError as e:
    print("配置错误:", e)
except SignalInputError as e:
    print("输入信号错误:", e)
except ComputationError as e:
    print("计算过程错误:", e)
except WaterfallExtractorError as e:
    print("封装包错误:", e)
```

## 输出字段

### 汇总特征（`features["summary"]`）

| 字段 | 说明 |
|------|------|
| `package_version` | 封装包版本 |
| `algorithm_id` | 算法编号 |
| `mode` | 分析模式（waterfall/cascade） |
| `main_freq_mean` | 主频均值 Hz |
| `entropy_mean` | 频谱熵均值 |
| `kurtosis_mean` | 谱峭度均值 |
| `kurtosis_peak_freq` | 峰值峭度对应频率 Hz |
| `rms_mean` | RMS 均值 |
| `crest_factor_mean` | 波峰因子均值 |
| `spectral_centroid_mean` | 频谱重心均值 Hz |
| `harmonic_ratio_mean` | 谐波能量比均值 |
| `envelope_peak_mean` | 包络谱峰值均值 |

### 时序曲线（`features["timeseries"]`）

| 字段 | 说明 |
|------|------|
| `main_freq` | 每帧主频 Hz |
| `entropy` | 每帧频谱熵 |
| `kurtosis` | 每帧谱峭度 |
| `rms` | 每帧 RMS |
| `crest_factor` | 每帧波峰因子 |
| `spectral_centroid` | 每帧频谱重心 |
| `harmonic_ratio` | 每帧谐波能量比 |
| `envelope_peak` | 每帧包络谱峰值 |

## 核心文件

- `waterfall_extractor/waterfall_core.py`：核心算法（信号处理 + 瀑布图/级联图 + 特征提取）
- `waterfall_extractor/version.py`：版本元数据
- `waterfall_extractor/__init__.py`：包入口
- `examples/basic_usage.py`：瀑布图模式示例
- `examples/cascade_usage.py`：级联图模式示例

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sampling_rate` | 20000.0 | 采样率 Hz |
| `window_size` | 1024 | FFT 窗长 |
| `overlap` | 0.5 | 重叠率 |
| `nfft` | 4096 | FFT 点数 |
| `window_type` | "hann" | 窗函数类型 |
| `cascade_rpm_step` | 25.0 | 级联图转速步长 rpm |
| `cascade_rpm_range` | (500, 5000) | 级联图转速范围 rpm |
| `device_type` | "universal" | 设备类型 |
| `rotational_freq_hz` | 24.67 | 转频 Hz |
