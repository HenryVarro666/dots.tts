# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

"""防混叠低通滤波器：设计 (filter design) 与一维卷积应用。
Anti-aliasing low-pass filter: filter design + 1-D convolution.

【在数据流里的位置 / Where this sits】
本文件属于 BigVGAN AudioVAE 的 vocoder 子模块。BigVGAN 在每个非线性激活
(Snake / 各类 activation) 前后会对信号做 upsample → 非线性 → downsample，
这一"过采样 (oversampling)"流程要求在升/降采样时插入理想低通滤波器，把激活
引入的高频谐波 (harmonics) 滤掉，从而抑制 aliasing(混叠)。本文件提供该低通
滤波器的核 (kernel) 设计与卷积实现；被 ``alias_free_resample.py`` 的
``UpSample1d`` / ``DownSample1d`` 调用。

理论依据是 StyleGAN3 "Alias-Free GAN" 的思想:对连续信号做非线性后必须先
带限 (band-limit) 再降采样，否则高频会折叠回基带造成失真。

【关键函数/类 / Key functions & classes】
- ``sinc``        : sin(pi*x)/(pi*x),理想 brick-wall 低通的时域形式。
- ``kaiser_sinc_filter1d`` : 用 Kaiser window 加窗的 sinc,得到有限长 (FIR)
                   低通核,平衡过渡带宽 (transition bandwidth) 与旁瓣 (sidelobe)。
- ``LowPassFilter1d``      : 把上面的核包成 ``nn.Module``,支持 causal/非 causal
                   padding、grouped depthwise 卷积、可学习或固定核。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!

        归一化 sinc 函数 sin(pi*x)/(pi*x),理想低通滤波器的时域冲激响应。
        Normalized sinc — the time-domain impulse response of an ideal
        brick-wall low-pass filter.
        当且仅当旧版 torch 没有内建 ``torch.sinc`` 时才用到这个回退实现 (fallback)。
        """
        # x == 0 处 sin(pi*x)/(pi*x) 是 0/0,用 torch.where 显式取极限值 1.0,
        # 避免数值除零 (numerical stability);其余位置走解析公式。
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(
    cutoff, half_width, kernel_size
):  # return filter [1,1,kernel_size]
    """设计 Kaiser-windowed sinc 一维低通滤波器核 (FIR low-pass kernel)。

    Design a Kaiser-windowed sinc 1-D low-pass FIR filter.

    理想低通核是无限长的 sinc,直接截断会产生 Gibbs 振铃 (ringing);这里用
    Kaiser window 对 sinc 加窗,通过 ``beta`` 在过渡带宽 (transition bandwidth)
    与阻带衰减 (stopband attenuation / 旁瓣) 之间折中。

    参数 / Args:
        cutoff:      归一化截止频率 (cycles/sample),范围 (0, 0.5];0.5=Nyquist。
                     上/下采样时通常取 0.5/ratio,即把带宽限制到目标采样率的一半。
        half_width:  过渡带半宽 (normalized),越小过渡越陡但需要更长的核。
        kernel_size: FIR 核长度 (抽头数 / taps);偶/奇都支持,见下方 even 分支。

    返回 / Returns:
        filter: 形状 (1, 1, kernel_size) 的张量,可直接喂给 grouped conv1d 作为
                depthwise 卷积核;sum 归一化为 1 以保证 DC 增益为 1。
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    # Kaiser 窗参数估计 (Kaiser design formula):由阻带衰减 A(dB) 反推形状参数
    # beta。delta_f 是归一化过渡带宽,A 越大窗越"尖"、旁瓣越低、过渡越宽。
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    # 经典 Kaiser 三段经验公式 (Oppenheim & Schafer):按所需阻带衰减 A 选 beta。
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0  # 衰减需求很低,退化为矩形窗 (beta=0)
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    # 构造时间轴 (sample index),使核关于中心对称、采样在 sinc 峰值附近:
    if even:
        # 偶数核没有正中心抽头,把采样点整体偏移 +0.5 落在两个半整数之间,
        # 对应 StyleGAN3 的偶数核约定。
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        # 奇数核以 index=half_size 为中心 (time=0 即 sinc 峰值)。
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)  # 截止为 0 = 全阻带,核全零
    else:
        # 加窗 sinc:理想低通 2*cutoff*sinc(2*cutoff*t) 乘以 Kaiser window 截断成 FIR。
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        # 归一化使 sum=1 → 频率响应 DC(0Hz) 增益为 1,常量分量不被衰减/放大。
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)  # (kernel,) -> (1,1,kernel) 适配 conv1d

    return filter


class LowPassFilter1d(nn.Module):
    """一维低通滤波层:把 Kaiser-sinc 核包成可微的 depthwise conv1d 模块。

    1-D low-pass filter as a differentiable depthwise conv1d module.

    用途:在 anti-aliased 降采样 (``DownSample1d``) 中,先低通滤波带限信号再按
    ``stride`` 抽取,防止 aliasing。每个通道用同一/各自的核做 grouped(=depthwise)
    卷积 (``groups=C``),通道之间不混合。

    关键设计 / Design notes:
        - causal 模式:只在左侧补 ``kernel_size-1`` 的 padding,右侧不补,保证输出
          只依赖当前及过去样本 → 支持 double-streaming 流式推理 (streaming),不引入
          未来信息泄漏。非 causal 模式则左右对称补零以保持相位对齐。
        - fixed_filter=True:核作为不可训练 buffer(纯信号处理意义的固定低通);
          False:核作为 ``nn.Parameter`` 随训练微调,并 expand 到每通道一份。
    """

    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
        channels: int = 1,
        causal: bool = True,
        fixed_filter: bool = False,
    ):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        if causal:
            # 因果 padding:全部补在左侧,输出长度与输入对齐且不看未来样本。
            self.pad_left = kernel_size - 1
            self.pad_right = 0
        else:
            # 非因果:左右近似对称补 padding,偶数核左侧少补 1 以保持中心对齐。
            self.even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(self.even)
            self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.fixed_filter = fixed_filter
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)  # (1,1,kernel)
        if fixed_filter:
            # 固定核:注册为 buffer(随 .to()/state_dict 走但不参与梯度更新)。
            self.register_buffer("filter", filter)
        else:
            # 可学习核:expand 成 (channels,1,kernel) 让每个通道一份独立可训练核;
            # expand 是视图,.clone() 后再做 Parameter 以拿到可写、可求导的实存张量。
            self.filter = nn.Parameter(filter.expand(channels, -1, -1).clone())

    # input [B, C, T]
    def forward(self, x):
        """对 (B, C, T) 输入做低通滤波 (+可选降采样)。

        Apply low-pass filtering (and optional downsampling) to (B, C, T).

        流程:边界 padding → depthwise conv1d(groups=C,逐通道独立卷积)。
        当 ``stride>1`` 时,卷积同时完成带限后的抽取 (decimation)。
        返回 (B, C, T') 张量,T' 取决于 padding/stride。
        """
        _, C, _ = x.shape
        if self.padding:
            # replicate 等模式补边,避免零填充在边界引入虚假高频/能量塌陷。
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        if self.fixed_filter:
            # 固定核只有 (1,1,kernel),需 expand 到 (C,1,kernel) 才能配 groups=C。
            out = F.conv1d(
                x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
            )
        else:
            # 可学习核已是 (C,1,kernel),直接做 depthwise 卷积。
            out = F.conv1d(x, self.filter, stride=self.stride, groups=C)
        return out
