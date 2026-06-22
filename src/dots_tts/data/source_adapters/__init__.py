"""数据源适配器包 / Source adapter package.

中文：把异构数据源统一成流水线可消费的样本迭代器。base_adapter(适配器抽象基类，
定义统一接口)、jsonl_manifest_adapter(从 JSONL manifest 读取 音频路径+文本 等条目)、
multi_source_adapter(把多个数据源按权重/比例混合成一个逻辑源)。新增数据源时实现基类
接口即可接入，无需改动下游 pipeline。

EN: Normalizes heterogeneous data sources into a uniform sample iterator the pipeline can
consume — base_adapter (abstract adapter interface), jsonl_manifest_adapter (read
audio-path + text entries from a JSONL manifest) and multi_source_adapter (mix several
sources into one logical stream by weight/ratio).
"""
