"""工具包 / Utilities package.

中文：跨训练与推理共享的通用工具函数。audio(音频读写/重采样/规整)、text(文本规整与
清洗)、tokenizer(文本分词器封装)、logging(日志配置)、profiling(性能/耗时分析)、
util(其它零散通用辅助)。这些与具体模型解耦，可被各层自由复用。

EN: General-purpose helpers shared across training and inference — audio (audio
I/O / resampling / normalization), text (text normalization & cleanup), tokenizer (text
tokenizer wrapper), logging (logging setup), profiling (timing/perf profiling) and util
(misc helpers). Decoupled from any specific model so every layer can reuse them.
"""
