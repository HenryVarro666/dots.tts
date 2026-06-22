"""模型族 / Model families.

中文：存放完整的、可端到端使用的模型定义(把 modules 里的网络组件组装成完整模型)。
当前包含 dots_tts 子包：连续潜在自回归 TTS 主模型(自回归 backbone + flow-matching DiT
声学头 + AudioVAE 解码)。modules 提供「零件」，本包提供「整机」。

EN: Houses complete, end-to-end model definitions that assemble the network components
from `modules` into full models. Currently contains the dots_tts subpackage — the
continuous-latent autoregressive TTS model (AR backbone + flow-matching DiT acoustic
head + AudioVAE decoding). `modules` provides the parts; this package the assembled models.
"""
