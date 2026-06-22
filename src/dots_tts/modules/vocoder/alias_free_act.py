# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

"""Alias-free 激活函数 / Alias-free activations for the BigVGAN AudioVAE vocoder.

本文件做什么 (What this file does)
---------------------------------
dots.tts 用 BigVGAN 风格的 AudioVAE 在「连续 latent 空间」做编解码（无离散音频
token）。BigVGAN 的核心技巧之一是 **alias-free activation**：在 vocoder 里使用
周期性激活（Snake / SnakeBeta，基于 sin）能更好地刻画音频的谐波结构，但任何非线性
都会产生高于 Nyquist 频率的谐波，直接采样会造成 **aliasing（混叠）**——表现为可听见
的金属/嗡鸣噪声。

解决办法（Karras et al. "Alias-Free GAN" 的思路）：把「激活」夹在一对重采样之间——
  upsample (x up_ratio)  ->  activation  ->  downsample (/ down_ratio)
先上采样把信号搬到更高采样率，让非线性新产生的谐波留在频带内；激活后再用低通滤波
（low-pass filter）下采样，把超出 Nyquist 的成分滤掉，从而抑制混叠。这就是
``Activation1d`` 的全部职责。

在数据流里的位置 (Where it sits)
---------------------------------
属于推理时的 **解码端**：自回归主干 + flow-matching DiT 预测出连续 latent 后，
AudioVAE / BigVGAN 解码器把 latent 还原成 24/48 kHz 波形；这些 alias-free 激活
就嵌在解码器的每个上采样卷积块中。训练时同样使用（alpha/beta 是可训练参数）。

关键类清单 (Key classes)
------------------------
- ``Activation1d``: upsample→act→downsample 的 alias-free 包装器。
- ``Snake``:     周期性激活  x + (1/alpha) * sin^2(alpha * x)。
- ``SnakeBeta``: Snake 的变体，用独立的 beta 控制周期分量的幅度（magnitude）。

依赖 (Deps): ``alias_free_resample`` 提供 ``UpSample1d`` / ``DownSample1d``
（其低通核来自 ``alias_free_filter`` 的 kaiser-windowed sinc）。
"""

import torch
import torch.nn as nn
from torch import pow, sin
from torch.nn import Parameter

from .alias_free_resample import DownSample1d, UpSample1d


class Activation1d(nn.Module):
    """Alias-free 激活包装器 / anti-aliased activation wrapper.

    把任意逐元素激活 ``activation`` 夹进「上采样 → 激活 → 下采样」三明治，
    使非线性新产生的高频谐波在被下采样低通滤掉之前先有足够带宽容纳，从而
    避免 aliasing。这正是 BigVGAN 里 Snake 激活的标准用法。

    参数 (Args):
        activation: 逐元素激活模块（如 ``Snake`` / ``SnakeBeta``）；需带
            ``.in_features`` 属性以告知重采样器通道数 C。
        up_ratio / down_ratio: 上/下采样倍率，通常成对相等（默认 2，即 2x 过采样）。
        up_kernel_size / down_kernel_size: 重采样低通核长度。
        causal: 是否使用因果（causal）重采样滤波——流式 / double-streaming 推理
            需要因果以避免依赖未来样本。
        fixed_filter: True 时滤波核为固定 buffer（不学习），False 时核作为可训练
            ``nn.Parameter``（逐通道扩展）。

    形状 (Shape): 输入/输出均为 (B, C, T)，T 经上采样再下采样后保持不变。
    """

    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        causal=True,
        fixed_filter=False,
    ):
        super().__init__()
        # causal=False
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(
            up_ratio,
            up_kernel_size,
            activation.in_features,
            causal=causal,
            fixed_filter=fixed_filter,
        )
        self.downsample = DownSample1d(
            down_ratio,
            down_kernel_size,
            activation.in_features,
            causal=causal,
            fixed_filter=fixed_filter,
        )

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)  # 过采样：给激活新生的高频谐波留出带宽 / oversample first
        x = self.act(x)  # 在高采样率下施加非线性，谐波尚未折叠 / nonlinearity at high rate
        return self.downsample(x)  # 低通后下采样，滤除 >Nyquist 成分以抑制混叠 / LPF + decimate


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)

    中文说明 (Notes)
    ----------------
    Snake 激活: ``f(x) = x + (1/alpha) * sin^2(alpha * x)``。它在 x 上叠加一个
    周期项，自带「周期性归纳偏置（periodic inductive bias）」，比 ReLU/GELU 更适合
    建模音频这类强谐波信号——这正是 BigVGAN 选它做 vocoder 激活的原因。
    - ``alpha``: 逐通道（per-channel，长度 = in_features）可训练参数，控制周期项
      的频率；alpha 越大频率越高。
    - 公式里的 ``x`` 这一线性项保证了恒等通路（identity path），利于梯度传播。
    形状 (Shape): 输入/输出同形 (B, C, T)。
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        # alpha_logscale: 是否在对数域参数化 alpha——log 域下 exp(alpha) 恒为正，
        # 训练更稳定、动态范围更大 / log-scale keeps alpha strictly positive.
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            # 注意: zeros * alpha == 0，故 log 域恒从 0 起步（即 exp 后 = 1），
            # 此处 alpha 实参不影响初值 / log-scale always starts at 0.
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001  # 防止 1/alpha 除零的小常数 / eps guard

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake := x + 1/a * sin^2 (xa)

        逐元素施加 ``x + (1/alpha) * sin^2(alpha * x)``。返回与输入同形 (B, C, T)。
        """
        # alpha 形状 (C,) -> (1, C, 1)，以便与 x=(B, C, T) 按通道广播 / broadcast over B,T
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)  # 对数域参数转回线性正值 / map back to positive
        # + no_div_by_zero 防止 alpha=0 时除零 / eps in denominator avoids div-by-zero
        return x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)

    中文说明 (Notes)
    ----------------
    SnakeBeta 把 Snake 的「一个参数同时管频率与幅度」解耦成两个独立可训练参数:
    ``f(x) = x + (1/beta) * sin^2(alpha * x)``。
    - ``alpha`` 控制周期项的频率（frequency，在 sin 内部）。
    - ``beta``  控制周期项的幅度（magnitude，作分母缩放）。
    解耦后表达力更强、更易拟合不同频段的谐波能量。两者均为逐通道参数，形状随通道
    广播；输入/输出同形 (B, C, T)。
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        # 同 Snake: log 域参数化使 alpha/beta 恒正、训练更稳 / log-scale keeps them positive.
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable  # beta 与 alpha 同步开关可训练性

        self.no_div_by_zero = 0.000000001  # 防止 1/beta 除零的小常数 / eps guard

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)

        逐元素施加 ``x + (1/beta) * sin^2(alpha * x)``——alpha 管频率、beta 管幅度。
        返回与输入同形 (B, C, T)。
        """
        # alpha/beta 均由 (C,) reshape 成 (1, C, 1) 以按通道广播到 (B, C, T)
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)  # 对数域 -> 正频率系数 / positive frequency
            beta = torch.exp(beta)  # 对数域 -> 正幅度系数 / positive magnitude
        # 分母用 beta（而非 alpha）缩放幅度，是 SnakeBeta 区别于 Snake 的关键
        return x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
