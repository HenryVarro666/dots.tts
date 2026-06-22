# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

"""防混叠（anti-aliasing）的上/下采样模块 —— BigVGAN AudioVAE vocoder 的基础算子之一。

本文件做什么 / What this file does:
    提供 1D 信号的 **alias-free（防混叠）上采样 `UpSample1d` 与下采样 `DownSample1d`**。
    在神经声码器（BigVGAN 系）里，每次改变采样率前后都要配合一个低通 sinc 滤波器，
    把超过 Nyquist 频率的成分滤掉，从而避免点态非线性激活（如 Snake/GELU）产生的高频
    分量在重采样时折叠（fold）回低频造成的 **aliasing**。这正是 "alias-free activation"
    思路（源自 StyleGAN3 / alias-free-torch）的核心算子。

在数据流里的位置 / Position in the pipeline:
    dots.tts 的 AudioVAE 解码端（BigVGAN）把连续 latent 逐级上采样回 48 kHz 波形；
    其上采样块前后会插入这里的 `UpSample1d`/`DownSample1d` 来保证 alias-free。
    与离散音频 token 无关 —— dots.tts 在**连续潜在空间**做 flow-matching，
    本模块只负责最终波形侧的重采样滤波。

关键类 / Key classes:
    - UpSample1d   : 倍率为 `ratio` 的防混叠上采样（转置卷积 + sinc 滤波核）。
    - DownSample1d : 倍率为 `ratio` 的防混叠下采样（低通滤波后按 stride 抽取）。

依赖 / Dependencies:
    `kaiser_sinc_filter1d`：用 Kaiser 窗截断的 sinc 生成低通 FIR 核（见 alias_free_filter.py）。
    `LowPassFilter1d`     ：把上述核包成一个可做 stride 卷积的低通模块（下采样用）。
"""

import torch.nn as nn
from torch.nn import functional as F

from .alias_free_filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    """防混叠上采样 / Alias-free 1D up-sampling（倍率 = ``ratio``）。

    原理 / How it works:
        先用 `F.conv_transpose1d`（转置卷积）把序列在时间维插值放大 `ratio` 倍，
        卷积核就是一个**低通 sinc 滤波器**（由 `kaiser_sinc_filter1d` 生成），
        从而在插入新样本点的同时抑制镜像谱（imaging）带来的高频混叠。
        乘以 `ratio` 是为了补偿上采样导致的能量/幅度缩放（保持 DC 增益≈1）。

    关键参数 / Key args:
        ratio (int)        : 上采样倍率，同时作为转置卷积的 stride。
        kernel_size (int)  : 滤波核长度；默认按 `int(6*ratio//2)*2`（偶数）自适应，
                             ratio 越大需要越长的核才能保持过渡带锐度。
        channels (int)     : 通道数 C；非固定核时每个通道各持一份可学习核（depthwise）。
        causal (bool)      : 是否做**因果**（causal）上采样。流式（streaming）推理需要因果，
                             即输出只依赖当前及历史样本，便于 double-streaming 低延迟生成。
        fixed_filter (bool): True → 核为不可训练 buffer（固定 DSP 滤波器）；
                             False → 核为 `nn.Parameter`，随训练微调。

    形状 / Shapes:
        输入 x : (B, C, T)
        输出   : (B, C, T*ratio)（裁剪掉卷积引入的边缘后）
    """

    def __init__(
        self, ratio=2, kernel_size=None, channels=None, causal=True, fixed_filter=False
    ):
        super().__init__()
        self.ratio = ratio
        # 核长默认 6*ratio 取偶；ratio 越大、过渡带越窄，需要更长的 FIR 核
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.channels = channels
        self.causal = causal
        self.fixed_filter = fixed_filter
        if causal:
            # 因果模式：不在两侧对称补零，边缘多余样本改由 forward 末端裁掉（见下）
            self.pad = 0
        else:
            # 非因果（对称）模式：预先算好转置卷积输出两端要裁掉的左右长度，
            # 使有效输出严格对齐到 T*ratio 且时延居中
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = (
                self.pad * self.stride + (self.kernel_size - self.stride) // 2
            )
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
        # cutoff/half_width 随 ratio 缩放：上采样 ratio 倍后 Nyquist 相对位置变为 0.5/ratio
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        if self.fixed_filter:
            # 固定核：作为单份 (1,1,K) buffer 注册，forward 时再按通道 expand
            self.register_buffer("filter", filter)
        else:
            # 可学习核：复制到每个通道 (C,1,K) 作为参数，配合 groups=C 做 depthwise 卷积
            self.filter = nn.Parameter(filter.expand(channels, -1, -1).clone())

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape
        # 用 replicate（边缘复制）补边而非补零，避免边界处引入虚假阶跃/响铃
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        if self.fixed_filter:
            # 固定核：把单份核 expand 到 C 个通道，groups=C 即对每通道独立做转置卷积；
            # 乘 ratio 补偿上采样的幅度缩放
            x = self.ratio * F.conv_transpose1d(
                x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
            )
        else:
            # 可学习核：参数本身已是 (C,1,K)，直接 depthwise 转置卷积
            x = self.ratio * F.conv_transpose1d(
                x, self.filter, stride=self.stride, groups=C
            )
        if self.causal:
            # 因果：只丢尾部由卷积“看到未来”而多出的 (K-stride) 个样本，保证输出不依赖未来
            x = x[..., : -(self.kernel_size - self.stride)]
        else:
            # 非因果：对称裁掉两端 padding 残留，使输出居中对齐到 T*ratio
            x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    """防混叠下采样 / Alias-free 1D down-sampling（倍率 = ``ratio``）。

    原理 / How it works:
        下采样的关键是**先低通、再抽取**：直接按 stride 抽样会把高于新 Nyquist 的成分
        折叠成混叠。这里把全部工作交给 `LowPassFilter1d` —— 它内部用同一族 Kaiser-sinc 核
        做 stride=`ratio` 的低通卷积，等价于「滤波 + 降采样」一步完成（decimation）。

    关键参数 / Key args:
        ratio (int)        : 下采样倍率，作为低通模块的 stride。
        kernel_size (int)  : 同 `UpSample1d`，默认 `int(6*ratio//2)*2`（偶数）。
        channels / causal / fixed_filter: 透传给 `LowPassFilter1d`，含义与上采样一致
                             （causal=True 即流式因果，fixed_filter 控制核是否可训练）。

    形状 / Shapes:
        输入 x : (B, C, T)
        输出   : (B, C, T//ratio)
    """

    def __init__(
        self, ratio=2, kernel_size=None, channels=None, causal=True, fixed_filter=False
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        # 低通+抽取合一：cutoff/half_width 按 ratio 缩放到新 Nyquist，stride=ratio 完成降采样
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
            channels=channels,
            causal=causal,
            fixed_filter=fixed_filter,
        )

    def forward(self, x):
        # 直接复用低通模块：内部已完成 padding → 低通卷积 → stride 抽取
        return self.lowpass(x)
