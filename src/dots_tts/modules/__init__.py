"""网络组件包 / Network modules package.

中文：存放可复用的底层网络「零件」，由 models 层组装成完整模型。三个子包：
backbone(主干：flow-matching 用的 DiT、通用 layers、语义编码器)、
speaker(说话人/音色条件：CAM++ x-vector、fbank 特征)、
vocoder(声码器：BigVGAN / AudioVAE，连续 latent 与波形互转)。
这里只放组件级模块，不含端到端模型(那在 models 包)。

EN: Reusable low-level network building blocks, assembled into full models by the
`models` layer. Three subpackages: backbone (the DiT used for flow-matching, generic
layers, the semantic encoder), speaker (speaker/timbre conditioning: CAM++ x-vector,
fbank features) and vocoder (BigVGAN / AudioVAE converting between continuous latents
and waveforms). Component-level only — end-to-end models live in `models`.
"""
