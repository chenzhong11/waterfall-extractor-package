# 更新记录

## v1.0.0

- 初始封装版本。
- 合并 `signal_processor.py`、`waterfall_core.py`、`feature_extractor.py` 为单个 `waterfall_core.py`。
- 新增 `WaterfallConfig` 配置类，支持瀑布图/级联图参数配置。
- 新增 `WaterfallExtractor` 高级封装类，提供一站式分析接口。
- 新增异常体系：`WaterfallExtractorError`、`ConfigError`、`SignalInputError`、`ComputationError`。
- 输出结果增加版本信息：`package_version`、`algorithm_id`、`algorithm_version`、`feature_schema_version`。
- 核心算法只接收数组输入，文件读取由调用方负责。
