"""dots.tts 主模型的超参配置 / Hyper-parameter configs for the dots.tts model.

本文件定义 DotsTts 模型族用到的全部配置类（基于 pydantic 的 ``ConfigBase``，
而非标准 dataclass —— 因此**没有默认值的字段是必填项**，从权重附带的 config 文件
里加载/校验）。这些配置在 ``model.py`` / ``core.py`` 里被消费，决定了 AR 主干、
flow-matching 声学头(DiT)、AudioVAE vocoder 与说话人条件(x-vector)的形状与行为。

This module declares every config class used by the dots.tts model family. The
classes subclass the pydantic-based ``ConfigBase`` (NOT a plain dataclass), so any
field without a default is *required* and is filled/validated when loading the
``config`` shipped alongside the checkpoint. They are consumed in ``model.py`` /
``core.py`` and pin down the shapes/behaviour of the autoregressive (AR) backbone,
the flow-matching DiT acoustic head, the AudioVAE vocoder, and the speaker
conditioning (x-vector).

数据流位置 / Where this sits in the pipeline:
    text/audio  →  Qwen2.5 AR backbone (hidden states)
                →  ``PatchEncoder`` (_EncoderConfig)  把 latent patch 编码进上下文
                →  ``DiT`` (_DiTConfig) flow-matching velocity field  预测连续 latent
                →  AudioVAE (``vocoder``) 在连续潜在空间 decode 回 24kHz 波形
    (无离散音频 token / no discrete audio tokens; everything is continuous-latent.)

关键类清单 / Key classes:
    - ``_EncoderConfig``  : PatchEncoder（把已生成的 latent patch 压回 AR 上下文）的结构超参。
    - ``_DiTConfig``      : flow-matching 声学头 DiT 的结构超参（注意 ``modulation=True``，启用 adaLN 时间步调制）。
    - ``LossConfig``      : 训练三项损失权重（CE / flow-matching / EOS）；StrictConfigBase => 禁止多余字段。
    - ``MeanFlowConfig``  : MeanFlow 一步/少步蒸馏的开关与 duration-embedding 选项。
    - ``ModelConfig``     : 顶层模型配置，聚合上述子配置 + latent/patch/CFG/x-vector 等全局超参。
"""

from __future__ import annotations

from dots_tts.config.base import ConfigBase, StrictConfigBase
from dots_tts.modules.vocoder.config import AudioVAEConfig


class _EncoderConfig(ConfigBase):
    """PatchEncoder 的 Transformer 结构超参 / Transformer hyper-params for the PatchEncoder.

    PatchEncoder 负责把一个已生成的连续 latent **patch**（多帧 latent 打成一组）重新编码、
    压回 AR 主干的上下文里，让自回归地预测下一个 patch 时能"看见"刚生成的声学内容。
    它是一个标准多层自注意力 encoder：``causal=True`` 表示沿时间因果(只看过去)，
    与流式/自回归生成一致；``modulation=False`` 表示这里**不**做 adaLN 时间步调制
    （时间步调制只属于 flow-matching 的 DiT 头，见 ``_DiTConfig``）。

    The PatchEncoder re-encodes an already-generated continuous-latent *patch* (a group
    of latent frames) back into the AR backbone's context, so the next-patch prediction
    can attend to the acoustic content just produced. It is a plain multi-layer
    self-attention encoder. ``causal=True`` keeps attention causal over time (matches
    streaming / AR decoding); ``modulation=False`` means *no* adaLN timestep modulation
    here (that belongs only to the flow-matching DiT head, see ``_DiTConfig``).
    """

    num_layers: int = 6  # encoder Transformer 层数 / number of stacked attention blocks
    num_heads: int = 16  # 多头注意力的 head 数 / attention heads
    hidden_size: int = 1024  # 模型宽度(d_model) / hidden width per token
    ffn_hidden_size: int = 4096  # FFN 中间维度，约 4×hidden / feed-forward inner dim (~4x width)
    modulation: bool = False  # 是否启用 adaLN 调制；encoder 不需要时间步 => False
    qkv_bias: bool = False  # QKV 线性层是否带 bias / bias on the q/k/v projections
    qk_norm: bool = False  # 是否对 Q、K 做归一化(注意力数值稳定) / normalize Q,K before attention
    attn_dropout: float = 0.0  # 注意力权重 dropout / dropout on attention weights
    dropout: float = 0.0  # 残差/FFN 输出 dropout / dropout on block outputs
    norm_layer: str = "LayerNorm"  # 归一化类型 / which norm layer to use
    alibi_bias: bool = False  # 是否用 ALiBi 位置偏置 / use ALiBi positional bias
    rotary_bias: bool = False  # 是否用 RoPE 旋转位置编码 / use rotary position embedding (RoPE)
    rotary_theta: float | None = 10000  # RoPE 频率基底 theta / RoPE base frequency
    input_dim: int = 1024  # 输入投影维度(进入 encoder 前的特征维) / dim of features fed in
    causal: bool = True  # 因果注意力(只看过去帧)，匹配 AR/流式 / causal attention for AR/streaming


class _DiTConfig(ConfigBase):
    """flow-matching 声学头 DiT 的结构超参 / Hyper-params for the flow-matching DiT head.

    这个 DiT(Diffusion Transformer) 不是去噪 diffusion，而是 **flow-matching** 的
    velocity field 网络：给定带噪 latent x_t 和时间步 t，预测把 x_t 推向干净 latent 的
    速度场 v(x_t, t)；推理时用 ODE solver 沿该速度场积分得到目标 latent patch。
    与 ``_EncoderConfig`` 的关键差异是 ``modulation=True`` —— 即用 adaLN(adaptive LayerNorm)
    把时间步 t（及可选条件）注入每一层，这是 DiT 类网络做条件化的标准做法。
    相对 encoder 更深(18 vs 6 层)，因为声学细节由它负责。

    This DiT (Diffusion Transformer) does NOT run denoising diffusion; it is the
    **flow-matching** velocity-field network: given a noised latent x_t and timestep t,
    it predicts the velocity v(x_t, t) that transports x_t toward the clean latent. At
    inference an ODE solver integrates this field to produce the target latent patch.
    The key difference from ``_EncoderConfig`` is ``modulation=True`` — adaLN (adaptive
    LayerNorm) injects the timestep t (and optional conditioning) into every block, the
    standard conditioning trick for DiT-style nets. It is deeper than the encoder
    (18 vs 6 layers) because it carries the acoustic-detail modelling.
    """

    num_layers: int = 18  # DiT 深度(depth)，比 encoder 深 => 更强声学建模 / DiT depth
    num_heads: int = 16  # 多头注意力 head 数 / attention heads
    hidden_size: int = 1024  # DiT 宽度(width, d_model) / hidden width
    ffn_hidden_size: int = 4096  # FFN 中间维度(~4×width) / feed-forward inner dim
    modulation: bool = True  # 启用 adaLN 时间步调制 => flow-matching 条件化的关键 / adaLN timestep modulation
    qkv_bias: bool = False  # QKV 线性层 bias / bias on q/k/v projections
    qk_norm: bool = False  # 是否归一化 Q、K / normalize Q,K before attention
    attn_dropout: float = 0.0  # 注意力 dropout / attention dropout
    dropout: float = 0.0  # 块输出 dropout / block-output dropout
    norm_layer: str = "LayerNorm"  # 归一化类型 / norm layer type
    alibi_bias: bool = False  # 是否用 ALiBi 位置偏置 / use ALiBi positional bias
    rotary_bias: bool = True  # DiT 默认开 RoPE(encoder 默认关) / RoPE on by default here
    rotary_theta: float | None = 10000  # RoPE 频率基底 / RoPE base frequency theta


class LossConfig(StrictConfigBase):
    """训练损失各项权重 / Weights for the training loss terms.

    继承 ``StrictConfigBase`` => ``extra="forbid"``：配置里出现未声明字段会直接报错，
    避免 loss 权重被打错名字而静默失效。总损失大致是三项加权和：AR 主干的交叉熵、
    flow-matching 的回归损失、以及句末 EOS 预测损失。

    Subclasses ``StrictConfigBase`` => ``extra="forbid"``: any undeclared field raises,
    so a misspelled weight key fails loudly instead of being silently ignored. The total
    loss is roughly a weighted sum of three terms: the AR backbone cross-entropy, the
    flow-matching regression loss, and the end-of-sequence (EOS) prediction loss.
    """

    ce_weight: float = 1.0  # AR 主干交叉熵(cross-entropy)权重 / cross-entropy loss weight
    fm_weight: float = 1.0  # flow-matching velocity 回归损失权重 / flow-matching loss weight
    eos_weight: float = 1.0  # 停止符 EOS 预测损失权重 / end-of-sequence loss weight


class MeanFlowConfig(ConfigBase):
    """MeanFlow 蒸馏配置 / Config for MeanFlow few-step distillation.

    MeanFlow 把多步 flow-matching ODE 求解蒸馏成**一步/少步**采样：网络改为预测一段时间
    区间上的"平均速度场"，从而大幅减少 ODE solver 步数、加速生成。``enabled=False`` 时
    走常规多步 flow-matching；为 ``True`` 时模型进入 ``meanflow`` 模式（见 ``core.py``）。

    MeanFlow distills the multi-step flow-matching ODE into a one-/few-step sampler: the
    net instead predicts an *average* velocity field over a time interval, slashing the
    number of ODE solver steps and speeding up generation. With ``enabled=False`` the
    model runs ordinary multi-step flow-matching; when ``True`` the model switches to its
    ``meanflow`` mode (see ``core.py``).
    """

    enabled: bool = False  # 是否启用 MeanFlow 蒸馏采样 / turn on MeanFlow distillation
    use_duration_embedding: bool = True  # MeanFlow 下是否注入时长(duration)嵌入条件 / inject duration embedding


class ModelConfig(ConfigBase):
    """dots.tts 顶层模型配置 / Top-level config aggregating all sub-configs.

    聚合 PatchEncoder / DiT / vocoder 三个子模块配置，并定义连续潜在空间、patch 化、
    classifier-free guidance (CFG)、说话人 x-vector 条件等全局超参。注意它是 pydantic
    模型而非 dataclass：``latent_dim`` / ``patch_size`` / ``PatchEncoder`` / ``DiT`` /
    ``vocoder`` **没有默认值 => 必填**，必须由 checkpoint 的 config 提供。

    Aggregates the PatchEncoder / DiT / vocoder sub-configs and defines the global
    hyper-params for the continuous-latent space, patchification, classifier-free
    guidance (CFG), and speaker x-vector conditioning. Being a pydantic model (not a
    dataclass), the fields without defaults — ``latent_dim``, ``patch_size``,
    ``PatchEncoder``, ``DiT``, ``vocoder`` — are *required* and must come from the
    checkpoint's config.
    """

    model_type: str = "dots_tts"  # 模型族标识，用于注册/分发 / model registry tag
    latent_dim: int  # AudioVAE 连续 latent 的通道维 D；AR/DiT/VAE 必须一致 / VAE latent channel dim (must match vocoder.latent_dim)
    patch_size: int  # 每个 AR step 生成的 latent 帧数(一个 patch=patch_size 帧) / latent frames produced per AR step
    cfg_droprate: float = 0.2  # 训练时丢弃条件的概率 => 启用 classifier-free guidance (CFG) / cond-drop prob enabling CFG
    PatchEncoder: _EncoderConfig  # 必填：把生成的 latent patch 压回 AR 上下文的 encoder 配置 / required PatchEncoder config
    DiT: _DiTConfig  # 必填：flow-matching velocity field 头(DiT)配置 / required flow-matching DiT-head config
    vocoder: AudioVAEConfig  # 必填：BigVGAN AudioVAE 编解码器配置(连续潜在空间) / required AudioVAE vocoder config
    fm_sigma: float = 0.0  # flow-matching 概率路径噪声尺度 σ(0=确定性直线路径) / flow-matching path noise scale
    xvec_drop_rate: float = 0.2  # 训练时丢弃 x-vector 说话人条件的概率(对 x-vector 的 CFG) / speaker-cond drop prob
    campplus_embedding_size: int | None = 512  # CAM++ x-vector 维度(说话人/音色嵌入) / CAM++ x-vector dim
    xvec_max_audio_seconds: float = 10.0  # 提取 x-vector 时参考音频的最大秒数(超出截断) / max ref-audio seconds for x-vector
    meanflow: MeanFlowConfig | None = None  # MeanFlow 蒸馏配置；None=不启用(走多步 flow-matching) / optional MeanFlow distillation


__all__ = [
    "LossConfig",
    "MeanFlowConfig",
    "ModelConfig",
]
