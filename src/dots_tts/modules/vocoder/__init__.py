"""声码器组件 / Vocoder modules.

中文：BigVGAN 声码器/AudioVAE 相关组件，负责连续 latent 与波形之间的转换(本系统中
AudioVAE 在连续 latent 空间编解码，无离散音频 token)。bigvgan(BigVGAN 生成器主体)、
config(其结构配置)、alias_free_act / alias_free_filter / alias_free_resample
(anti-aliasing 抗混叠激活与上/下采样滤波，抑制高频混叠、提升音质)。

EN: BigVGAN vocoder / AudioVAE components converting between continuous latents and
waveforms (this system's AudioVAE encodes/decodes in continuous latent space, with no
discrete audio tokens) — bigvgan (the BigVGAN generator), config (its architecture
config) and alias_free_act / alias_free_filter / alias_free_resample (anti-aliasing
activations and up/down-sampling filters that suppress high-frequency aliasing).
"""
