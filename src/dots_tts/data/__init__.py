"""数据包 / Data package.

中文：负责把原始语料(音频+文本)转成可喂给模型的批次。包含 source_adapters(读取
各种数据源，如 JSONL manifest)、pipelines(预处理/分词/TTS 流水线)、builders(数据集
构建)、batchers(成批与采样策略)、collator(整理为定长/padding 张量并构造 mask)、
streaming(流式数据读取)。这是训练侧 data flow 的入口层。

EN: Turns raw corpora (audio + text) into model-ready batches: source_adapters (read
data sources such as JSONL manifests), pipelines (preprocessing/tokenizing/TTS),
builders (dataset construction), batchers (batching & sampling), collator (pad to
tensors and build masks) and streaming (streaming data reads).
"""
