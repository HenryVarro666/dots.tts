"""AudioVAE / BigVGAN 声码器超参配置 (vocoder config)。

本文件只声明一个 pydantic 配置类 ``AudioVAEConfig``，集中描述 dots.tts 里
**连续潜在 (continuous-latent)** 声码器 AudioVAE(基于 BigVGAN)的全部结构超参。
它不含任何前向逻辑——真正消费这些字段的是同目录的 ``bigvgan.py``：编码器(Encoder)把
波形下采样进连续 latent 空间，解码器(Generator)再上采样回 24 kHz 波形。

在数据流里的位置 (where it sits)：
  - 训练/推理时先用这份 config 实例化 AudioVAE；
  - **编码侧**：waveform (B, 1, samples) → Encoder → 连续 latent (B, latent_dim, T)，
    供自回归主干 / flow-matching DiT 作为声学目标(无离散音频 token)；
  - **解码侧**：DiT 预测出的 latent (B, latent_dim, T) → Generator → 波形 (B, 1, samples)。

字段与 ``bigvgan.py`` 用法一一对应(下方逐行注释给出消费点)。AudioVAE 是一个
**VAE**：编码器输出 (mean, log-std) 两半、可重参数化采样，故 latent 是连续高斯潜变量
而非码本索引——这正是 dots.tts 与离散 codec TTS(如 Higgs v3)的关键区别。

关键类 (key classes)：
  - ``AudioVAEConfig``: 声码器结构超参容器，继承自 ``ConfigBase`` (pydantic, extra="allow")。
"""

from __future__ import annotations

from pydantic import Field

from dots_tts.config.base import ConfigBase


class AudioVAEConfig(ConfigBase):
    """AudioVAE (BigVGAN) 声码器的结构超参 (architecture hyper-parameters)。

    这是一个纯**配置 schema**：每个字段都是 ``bigvgan.py`` 里 Encoder/Generator 构造时
    读取的形状或开关参数；本类自身不做任何张量运算。继承 ``ConfigBase`` 后默认
    ``extra="allow"``，因此 checkpoint 里多出的字段会被宽容保留(便于跨版本加载)。

    设计要点 (design notes)：
      - 多数 list 字段默认空(``Field(default_factory=list)``)，**期望从 checkpoint 的
        config 注入真实取值**；空默认只是占位，直接用空 list 实例化无法构网。
      - 上采样 (Generator) 与下采样 (Encoder) 两组率/通道彼此呼应：
        总上采样率 ``∏upsample_rates`` 须等于 ``hop_size = ∏downsample_rates``，
        才能让解码输出长度精确还原编码前的波形长度。
      - 因为是 VAE，编码器实际输出通道是 ``latent_dim * 2``(mean + log-std 各占一半，
        见 ``bigvgan.py`` 的 ``extract_latents`` 重参数化)。

    字段速览(详见各行注释)：采样率、上/下采样结构、ResBlock 形状、激活函数、
    latent 维度、causal(流式因果)开关，以及若干末层 (final layer) 细节开关。
    """

    # 目标波形采样率(Hz)。注：dots.tts 端到端为 48 kHz，但本 AudioVAE 配置默认 24000；
    # 实际值以 checkpoint 注入为准。sample_rate 主要用于元信息/重采样，不直接决定卷积结构。
    sample_rate: int = 24000
    # Generator 各级转置卷积 (ConvTranspose1d) 的上采样倍率，逐级相乘 = 总上采样率 = hop_size。
    # 列表长度 = 上采样级数 ``num_upsamples``(bigvgan.py 用 len(upsample_rates) 计数)。
    upsample_rates: list[int] = Field(default_factory=list)
    # 与 upsample_rates 一一配对的转置卷积 kernel_size；通常取 2*rate 以减少棋盘伪影 (checkerboard)。
    upsample_kernel_sizes: list[int] = Field(default_factory=list)
    # Generator 入口通道数(从 latent_dim 先升维到这里)，之后每上采样一级通道数减半。
    upsample_initial_channel: int = 1536
    # 残差块类型选择："1"→AMPBlock1(带 anti-aliased 激活的多膨胀残差)，"2"→AMPBlock2(更轻)。
    resblock: str = "1"
    # 每个上采样级里并联的多个 ResBlock 的 kernel_size 列表；其长度 = ``num_kernels``，
    # 同级多核结果取平均(BigVGAN 的 multi-receptive-field fusion, MRF)。
    resblock_kernel_sizes: list[int] = Field(default_factory=list)
    # 与上面每个 kernel 对应的一组膨胀率 (dilation) — 故是 list[list[int]]，外层对齐 kernel,
    # 内层是该 ResBlock 内多层卷积的 dilation 序列(扩大感受野而不增下采样)。
    resblock_dilation_sizes: list[list[int]] = Field(default_factory=list)
    # 编码器 (Encoder) 各级下采样倍率；∏downsample_rates = hop_size(1 个 latent 帧 ↔ 多少波形样本)，
    # 必须与 ∏upsample_rates 相等，编解码长度才对齐。
    downsample_rates: list[int] = Field(default_factory=list)
    # 编码器各下采样级的输出通道数，与 downsample_rates 配对(bigvgan.py 的 Encoder channels 入参)。
    downsample_channels: list[int] = Field(default_factory=list)
    # 周期性非线性激活：默认 "snakebeta"(带可学习 β 的 Snake)，另支持 "snake"；其余值会在
    # bigvgan.py 里报错。Snake 类激活有助于建模音频的周期结构。
    activation: str = "snakebeta"
    # Snake/SnakeBeta 的 α(以及 β)是否在 log 空间参数化(alpha_logscale)，提升数值稳定与可学性。
    snake_logscale: bool = True
    # 连续 latent 的通道维 D。注意编码器最终输出 latent_dim*2(mean 与 log-std 各半，供 VAE 重参数化)。
    latent_dim: int = 128
    # 全局因果开关：True 时卷积只用左侧 padding(因果卷积)，使解码器支持 double-streaming 流式;
    # False 时用对称 same padding(离线、低延迟无要求)。本字段主控 Generator 一侧。
    causal: bool = False
    # 编码器里互信息/瓶颈 SLSTM 的层数(bigvgan.py 中 SLSTM(num_layers=mi_num_layers))，
    # 对 latent 做时序建模/正则。"mi" 指 mutual-information 风格的中间模块。
    mi_num_layers: int = 4
    # 编码器是否也走因果卷积；与 Generator 的 `causal` 分开控制(编码通常可非因果，解码才需流式)。
    causal_encoder: bool = False
    # Generator 末层输出卷积 conv_post 是否带 bias(bigvgan.py 用 h.get("use_bias_at_final", True))。
    use_bias_at_final: bool = True
    # 末层是否对输出波形过 tanh 收敛到 [-1, 1];False 则改用 clamp(见 bigvgan.py forward 收尾)。
    use_tanh_at_final: bool = True


__all__ = ["AudioVAEConfig"]
