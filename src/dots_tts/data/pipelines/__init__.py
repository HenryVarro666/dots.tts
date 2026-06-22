"""数据流水线包 / Data pipelines package.

中文：把单条样本逐步变换成模型输入的可组合流水线。base(流水线基类/接口)、
preprocessing(音频重采样、文本规整等预处理)、tokenizing(文本分词、构造 token 序列)、
tts_pipeline(串起以上步骤的端到端 TTS 数据流水线)。每个 stage 输入输出约定清晰，便于
增删步骤或复用。

EN: Composable per-sample transform stages that build model inputs — base (pipeline
base classes/interfaces), preprocessing (audio resampling, text normalization),
tokenizing (text tokenization / token-sequence construction) and tts_pipeline (the
end-to-end TTS data pipeline wiring those stages together).
"""
