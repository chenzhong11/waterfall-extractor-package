# 算法封装 SOP 方案

> 基于 `vibration_feature_extractor` 封装示例包总结的标准操作流程

---

## 1. 封装目标与原则

### 1.1 核心目标

将算法从「能跑的脚本」封装成「可集成的包」，让上位机、数据库、Web 服务等调用方可以直接 `pip install` 后使用。

### 1.2 封装原则

| 原则 | 说明 |
|------|------|
| **数组输入优先** | 核心算法只接收一维数组，文件读取由调用方负责 |
| **零外部依赖** | 核心包仅依赖 numpy，不绑定 pandas/scipy/matplotlib |
| **稳定输出契约** | 返回字典结构固定，带版本号，便于后续迭代 |
| **异常分级** | 配置错误、输入错误、计算错误分类抛出 |
| **函数级版本追踪** | 每个函数注释中标注版本、作者、日期 |

---

## 2. 目录结构规范

```
封装示例包/
├── pyproject.toml                  # 包构建配置
├── CHANGELOG.md                    # 版本更新记录
├── README.md                       # 使用文档
├── vibration_feature_extractor/    # 核心包目录
│   ├── __init__.py                 # 包入口，暴露公开 API
│   ├── version.py                  # 版本元数据集中管理
│   └── power_cepstrum_core.py      # 核心算法（单文件集中）
└── examples/                       # 使用示例
    ├── basic_usage.py              # 从文件加载示例
    └── use_from_array.py           # 从数组直接调用示例
```

### 2.1 目录职责划分

| 文件/目录 | 职责 |
|-----------|------|
| `pyproject.toml` | 定义包名、版本、依赖、构建方式 |
| `version.py` | 集中管理 `PACKAGE_VERSION`、`ALGORITHM_ID`、`ALGORITHM_VERSION`、`FEATURE_SCHEMA_VERSION` |
| `__init__.py` | 从子模块导入并暴露公开接口，定义 `__all__` |
| `核心算法.py` | 所有算法函数、配置类、异常类集中在一个文件 |
| `examples/` | 独立可运行的示例脚本 |

---

## 3. 封装流程

### Step 1: 版本元数据管理

创建 `version.py`，集中管理所有版本信息：

```python
"""Version metadata for the package."""

PACKAGE_VERSION = "1.1.1"           # 封装包版本
ALGORITHM_ID = "power_cepstrum_neg_sum_v1"  # 算法编号
ALGORITHM_VERSION = "A-1.1.1"       # 算法版本
FEATURE_SCHEMA_VERSION = "2026.07"  # 输出字段版本

def get_version_info() -> dict[str, str]:
    """返回版本信息字典，便于集成方记录算法来源。"""
    return {
        "package_version": PACKAGE_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
```

**要点：**
- 版本号与 `pyproject.toml` 保持同步
- 算法编号包含算法类型和版本，便于追溯
- `FEATURE_SCHEMA_VERSION` 用于标识输出字段结构版本

### Step 2: 异常体系设计

在核心算法文件中定义异常层级：

```python
class VibrationFeatureExtractorError(Exception):
    """封装包异常基类"""

class FeatureConfigError(ValueError, VibrationFeatureExtractorError):
    """配置参数非法"""

class SignalInputError(ValueError, VibrationFeatureExtractorError):
    """输入信号非法"""

class FeatureComputationError(RuntimeError, VibrationFeatureExtractorError):
    """计算过程异常"""
```

**异常分层原则：**
- 基类统一捕获：`except VibrationFeatureExtractorError`
- 子类精确处理：`except FeatureConfigError`
- 继承 Python 内置异常，保持兼容性

### Step 3: 配置类设计

使用 `@dataclass(frozen=True)` 实现不可变配置：

```python
@dataclass(frozen=True)
class FeatureConfig:
    sample_rate_hz: float = 2048.0
    window_size: int = 4096
    hop_size: int = 2048
    min_fft_size: int = 1024
    remove_mean: bool = True
    use_hann_window: bool = True
    quefrency_bands_ms: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"1_10ms": (1.0, 10.0), ...}
    )

    def validate(self) -> None:
        """校验所有配置项，尽早报错。"""
        ...
```

**设计要点：**
- `frozen=True` 防止运行时意外修改
- 提供合理默认值，降低调用门槛
- 内置 `validate()` 方法，统一校验逻辑

### Step 4: 核心算法封装

#### 4.1 输入校验函数

```python
def as_1d_signal(signal: Iterable[float]) -> np.ndarray:
    """将任意可迭代对象转为一维浮点数组，过滤 NaN/inf。"""
    ...
```

#### 4.2 预处理函数

```python
def preprocess_window(window, config) -> np.ndarray:
    """去均值 + 加窗，返回预处理后的窗口。"""
    ...
```

#### 4.3 核心计算函数

```python
def compute_power_spectrum(window, config) -> Tuple[np.ndarray, np.ndarray, float]:
    """计算单边功率谱，返回 (频率轴, 功率谱, 功率总和)。"""
    ...

def compute_power_cepstrum(window, config) -> np.ndarray:
    """计算功率倒谱：功率谱 → 取对数 → 反FFT。"""
    ...
```

#### 4.4 特征提取函数

```python
def extract_window_features(window, config) -> Dict[str, float]:
    """提取单窗口标量特征（neg_sum_1_10ms 等）。"""
    ...

def extract_feature_series(signal, config) -> List[Dict[str, float]]:
    """滑窗提取，返回每个窗口的特征明细。"""
    ...

def extract_features(signal, config) -> Dict[str, object]:
    """主入口：汇总特征 + 版本信息 + 统计量。"""
    ...
```

### Step 5: 包入口设计

`__init__.py` 负责暴露公开 API：

```python
"""面向集成调用的振动信号功率倒谱特征提取包。"""

from .power_cepstrum_core import (
    FeatureConfig,
    VibrationFeatureExtractorError,
    extract_features,
    extract_feature_series,
    # ... 其他公开接口
)
from .version import PACKAGE_VERSION, get_version_info

__version__ = PACKAGE_VERSION
__all__ = [
    "FeatureConfig",
    "VibrationFeatureExtractorError",
    "extract_features",
    # ...
]
```

### Step 6: 构建配置

`pyproject.toml` 示例：

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vibration-feature-extractor"
version = "1.1.1"
description = "Integration-oriented power cepstrum feature extraction package."
requires-python = ">=3.9"
dependencies = ["numpy"]

[tool.setuptools.packages.find]
include = ["vibration_feature_extractor*"]
```

### Step 7: 编写示例

#### 示例 1：从数组直接调用（推荐）

```python
from vibration_feature_extractor import FeatureConfig, extract_features
import numpy as np

signal = np.sin(2 * np.pi * 60.0 * t)  # 一维数组
config = FeatureConfig(sample_rate_hz=2048.0, window_size=4096, hop_size=2048)
result = extract_features(signal, config=config)

print(result["neg_sum_1_10ms"])
```

#### 示例 2：从文件加载后调用

```python
signal = np.loadtxt("data.txt").reshape(-1)  # 文件读取在外部完成
result = extract_features(signal, config=config)
```

### Step 8: 文档与更新记录

#### CHANGELOG.md 格式

```markdown
# 更新记录

## v1.1.1
- 核心函数注释补充 `创建人`、`编辑日期` 信息
- 新增异常基类，方便统一捕获
- 增强输入校验

## v1.1.0
- 核心算法集中到单文件
- 新增版本元数据管理
- 输出结果增加版本信息

## v1.0.0
- 初始封装版本
```

#### README.md 必含内容

1. **当前版本** — 包版本、算法版本、Schema 版本
2. **推荐集成方式** — 最简代码示例
3. **异常处理** — 异常层级说明
4. **输出字段** — 每个字段的含义
5. **核心文件** — 文件职责说明

---

## 4. 函数注释规范

每个核心函数必须包含以下注释信息：

```python
def extract_features(signal, config):
    """函数版本: F-1.1.1
    创建人: 刘博林
    编辑日期: 2026-07-07
    输入参数:
        signal，完整一维振动信号；
        config，可选 FeatureConfig 配置对象。
    输出参数: dict[str, object]，带版本信息和统计特征的汇总结果。

    面向外部程序集成的主入口函数。

    异常: 输入或配置非法时抛出对应异常。
    """
```

**注释字段说明：**

| 字段 | 说明 |
|------|------|
| 函数版本 | `F-` 前缀 + 版本号，区别于包版本 |
| 创建人 | 函数原始作者 |
| 编辑日期 | 最后修改日期 |
| 输入参数 | 每个参数的含义和类型 |
| 输出参数 | 返回值的类型和含义 |
| 异常 | 可能抛出的异常类型 |

---

## 5. 测试与验证

### 5.1 示例脚本验证

```bash
cd 封装示例包
python examples/use_from_array.py   # 合成信号测试
python examples/basic_usage.py      # 真实数据测试
```

### 5.2 验证检查清单

- [ ] `import` 不报错
- [ ] 默认参数能跑通
- [ ] 自定义参数能跑通
- [ ] 异常输入能正确抛出异常
- [ ] 输出字段包含版本信息
- [ ] 输出字段包含核心特征（如 `neg_sum_1_10ms`）

---

## 6. 发布与安装

### 6.1 本地开发安装

```bash
cd 封装示例包
pip install -e .
```

### 6.2 其他项目使用

```python
from vibration_feature_extractor import FeatureConfig, extract_features
```

---

## 7. 完整封装检查清单

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| **结构** | 目录结构符合规范 | ☐ |
| **结构** | `pyproject.toml` 配置正确 | ☐ |
| **版本** | `version.py` 集中管理版本 | ☐ |
| **版本** | 版本号与 `pyproject.toml` 同步 | ☐ |
| **异常** | 定义异常基类 | ☐ |
| **异常** | 配置/输入/计算异常分类 | ☐ |
| **配置** | 使用 `@dataclass(frozen=True)` | ☐ |
| **配置** | 提供合理默认值 | ☐ |
| **配置** | 内置 `validate()` 方法 | ☐ |
| **算法** | 核心算法只接收数组 | ☐ |
| **算法** | 文件读取在外部完成 | ☐ |
| **算法** | 每个函数有版本注释 | ☐ |
| **入口** | `__init__.py` 暴露公开 API | ☐ |
| **入口** | 定义 `__all__` | ☐ |
| **输出** | 返回字典带版本信息 | ☐ |
| **输出** | 核心特征字段稳定 | ☐ |
| **文档** | CHANGELOG.md 记录变更 | ☐ |
| **文档** | README.md 包含集成示例 | ☐ |
| **示例** | 提供数组调用示例 | ☐ |
| **示例** | 提供文件调用示例 | ☐ |
| **测试** | 示例脚本可运行 | ☐ |
| **测试** | 异常输入能正确报错 | ☐ |

---

## 附录：版本号管理建议

| 类型 | 格式 | 示例 |
|------|------|------|
| 封装包版本 | `MAJOR.MINOR.PATCH` | `1.1.1` |
| 算法版本 | `A-MAJOR.MINOR.PATCH` | `A-1.1.1` |
| 算法编号 | `{类型}_{特征}_{版本}` | `power_cepstrum_neg_sum_v1` |
| Schema 版本 | `YYYY.MM` | `2026.07` |
| 函数版本 | `F-MAJOR.MINOR.PATCH` | `F-1.1.1` |
