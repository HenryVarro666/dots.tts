# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""CAM++ 说话人编码器的基础网络层 / Building-block layers for the CAM++ speaker encoder.

本文件做什么 (What this file does)
---------------------------------
定义 CAMPPlus (campplus.py) 用来搭建 **x-vector 说话人编码器** 的所有可复用层。
CAM++ 是一个基于 **TDNN (Time-Delay Neural Network, 一维时序卷积)** 的说话人
embedding 网络: 输入一段语音的 fbank 特征 (B, T, F), 输出一个固定维度的
speaker embedding (x-vector), 用来表征"谁在说话 / 什么音色"。

在 dots.tts 数据流里的位置 (Position in the dots.tts pipeline)
-------------------------------------------------------------
dots.tts 是连续潜在 (continuous-latent) 的自回归 TTS。要做 **voice cloning / 音色
控制**, 它需要把一段参考音频 (reference audio) 压成一个 speaker embedding, 作为
flow-matching 声学头与自回归主干的 **说话人/音色条件 (speaker conditioning)**。
CAM++ 就是这个 x-vector 提取器: fbank (fbank.py) -> CAMPPlus (campplus.py) ->
本文件提供的层 -> 512 维 x-vector。它只在条件编码阶段使用, 不参与声学 latent 的
生成本身。

关键 CAM++ 思想 (Key CAM++ idea)
--------------------------------
CAM = Context-Aware Masking。普通 TDNN 的每个卷积只看局部感受野; CAM++ 额外用全局
/分段 (segment) 的上下文算出一组 0~1 的 **mask (注意力门控)**, 去重标定 (re-weight)
局部特征 —— 即 ``output = local_conv(x) * sigmoid(context)``。这让每一帧的局部特征能
感知更长程的上下文, 又比 self-attention 便宜。

关键类/函数清单 (Key classes / functions)
------------------------------------------
- ``get_nonlinear``        : 按字符串配置 (如 "batchnorm-relu") 组装非线性激活序列。
- ``statistics_pooling``   : 沿时间维做均值+标准差池化, 把变长序列压成定长向量。
- ``StatsPool``            : ``statistics_pooling`` 的 nn.Module 包装。
- ``TDNNLayer``            : 一维卷积 + 非线性, TDNN 的最小单元。
- ``CAMLayer``             : CAM++ 的核心 —— 局部卷积 × 上下文 mask 的门控层。
- ``CAMDenseTDNNLayer``    : bottleneck + CAMLayer, DenseNet 风格的单层。
- ``CAMDenseTDNNBlock``    : 多个 CAMDenseTDNNLayer 的 dense 连接 (特征沿通道拼接)。
- ``TransitLayer``         : dense block 之间的过渡层, 用 1x1 卷积压通道。
- ``DenseLayer``           : 1x1 卷积 + 非线性, 兼容 2D/3D 输入的全连接式层。
- ``BasicResBlock``        : 二维 ResNet 残差块, 给前端 FCM 下采样 fbank 用。
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn


def get_nonlinear(config_str, channels):
    """按字符串配置组装一段非线性激活/归一化序列 / Build a normalization+activation stack from a config string.

    用一个短横线分隔的字符串声明这一层后面要接哪些归一化/激活, 例如 "batchnorm-relu"
    表示先 BatchNorm1d 再 ReLU。这样整张网络只需传 ``config_str`` 就能统一切换激活配置。
    A hyphen-separated string declares the post-conv norm/activation, e.g. "batchnorm-relu".

    参数 (Args):
        config_str: 形如 "batchnorm-relu" 的配置串, 支持的 token:
            relu / prelu / batchnorm / batchnorm_ (末尾下划线表示 affine=False 不学缩放偏移)。
        channels: 通道数 C, 用于 PReLU 与 BatchNorm1d 的参数维度。

    返回 (Returns):
        nn.Sequential, 作用在 (B, C, T) 的一维特征上。
    """
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            # 末尾下划线变体: affine=False, 只做标准化不学习 gamma/beta, 常用于网络末端
            # 的 embedding 层, 避免再引入可学习缩放 / no learnable scale-shift.
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, _eps=1e-2):
    """统计池化: 沿时间维聚合成 (均值, 标准差) / Statistics pooling over the time axis.

    x-vector 网络的关键一步: 把变长的帧级特征 (B, C, T) 在时间维 T 上压成定长的
    utterance-level 表示。沿 ``dim`` 求 mean 和 std, 再在通道维拼接 ->  (B, 2C)。
    均值刻画"平均音色", 标准差刻画"音色的波动", 两者一起比单纯求均值更有区分度。
    This collapses the variable-length time axis into a fixed-size (mean, std) summary.

    参数 (Args):
        x: 帧级特征, 形状 (B, C, T)。
        dim: 聚合的时间维, 默认 -1。
        keepdim: 是否保留被聚合掉的维度 (输出再 unsqueeze 回 dim)。
        unbiased: std 是否用无偏估计 (除以 T-1)。
        _eps: 占位参数, 此函数未使用 (掩码版方差稳定见 campplus.py 的 _masked_stats_pooling)。

    返回 (Returns):
        stats: (B, 2C) 的定长统计向量 (keepdim=True 时为 (B, 2C, 1))。
    """
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    # 通道维拼接 [mean; std], 通道数翻倍: C -> 2C / concat along channel dim.
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    """``statistics_pooling`` 的 nn.Module 包装 / Module wrapper around statistics_pooling.

    作为 ``CAMPPlus.xvector`` 里名为 "stats" 的子层使用。注意推理时若传了 lengths,
    CAMPPlus.forward 会绕过本模块改用 _masked_stats_pooling (按真实长度做掩码统计),
    以避免 padding 帧污染统计量 / bypassed in favor of the masked version when lengths given.
    """

    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    """TDNN 最小单元: 一维卷积 + 非线性 / Basic TDNN unit: a 1D conv followed by norm+activation.

    TDNN (Time-Delay Neural Network) 本质就是沿时间维滑动的一维卷积, 一个 ``kernel_size``
    的卷积核同时看相邻若干帧, 即在时间上"建模上下文"。这里把 conv + (BatchNorm/激活) 打包
    成一层。CAMPPlus 把它作为前端的第一层 (带 stride=2 做时间下采样)。
    A 1D temporal conv that aggregates neighbouring frames, plus the configured norm/activation.

    参数 (Args):
        in_channels / out_channels: 输入/输出通道 C_in / C_out。
        kernel_size: 时间感受野 (看几帧)。
        stride: 时间下采样步长。
        padding: 时间维左右补零; 传入负数表示"自动算对称 padding 使长度不变"(见下)。
        dilation: 空洞卷积膨胀率, 在不增加参数的情况下放大感受野。
        bias / config_str: 卷积是否带偏置 / 后接的非线性配置串。

    形状 (Shape): 输入 (B, C_in, T) -> 输出 (B, C_out, T')。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            # padding<0 是一个约定: 让本层自动选择"same"对称 padding (需奇数核),
            # 使输出时间长度与输入对齐 / negative padding => auto symmetric "same" padding.
            assert kernel_size % 2 == 1, (
                f"Expect equal paddings, but got even kernel size ({kernel_size})"
            )
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)  # 一维时序卷积 / temporal conv: (B,C_in,T) -> (B,C_out,T')
        return self.nonlinear(x)


class CAMLayer(nn.Module):
    """CAM++ 的核心门控层: 局部卷积 × 上下文 mask / Context-Aware Masking layer.

    这是 CAM++ 名字里 "CAM" 的来源。计算分两路:
      1) 局部路 ``y = linear_local(x)`` —— 普通 TDNN 卷积, 只看局部感受野。
      2) 上下文路 —— 用全局均值 + 分段池化得到的上下文, 经 1x1 降维-ReLU-1x1 升维-Sigmoid,
         产生一组 0~1 的 **mask m** (逐通道的注意力门控)。
    输出 ``y * m``: 用长程上下文去重标定 (re-weight) 每个通道的局部响应, 让局部特征"看见"
    更大范围的信息, 计算量却远小于 self-attention。
    output = local_conv(x) * sigmoid(context_gate(x)); context blends global mean + segment pooling.

    参数 (Args):
        bn_channels: 输入 (bottleneck 后) 的通道数。
        out_channels: 局部卷积与门控的输出通道。
        kernel_size/stride/padding/dilation/bias: 局部卷积的参数。
        reduction: 上下文门控的瓶颈降维比 (linear1 把通道降到 bn_channels//reduction)。
    """

    def __init__(
        self,
        bn_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        reduction=2,
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)  # 局部路: 普通 TDNN 卷积 / local conv branch
        # 上下文路: 全局均值 (整段一个值, 广播回每帧) + 分段池化 (每 seg_len 帧一个值),
        # 二者相加得到"既有全局又有局部段"的上下文 / global mean + segment-level context.
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))  # 1x1 降维 + ReLU (SE 风格瓶颈)
        m = self.sigmoid(self.linear2(context))  # 1x1 升维 + Sigmoid -> 0~1 门控 mask m
        return y * m  # Context-Aware Masking: 用 mask 逐通道重标定局部特征

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        """分段池化: 每 seg_len 帧聚合再广播回原长度 / Segment pooling broadcast back to T.

        把时间轴切成长度 seg_len 的段, 段内取均值/最大值得到段级上下文, 再把每个段的值
        复制 seg_len 份铺回原始时间分辨率。结果与输入同长 (B, C, T), 使其能与逐帧的局部
        特征做乘法门控。相比全局均值, 它保留了"分段"的中等尺度上下文。
        """
        if stype == "avg":
            # ceil_mode=True: 末尾不足 seg_len 的残段也保留为一个段 / keep the trailing partial segment.
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        # 把每个段值在新维度复制 seg_len 份再展平, 即"上采样回逐帧": (B,C,n_seg) -> (B,C,n_seg*seg_len)
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]  # ceil_mode 可能多出尾巴, 裁回原始长度 T / crop to T


class CAMDenseTDNNLayer(nn.Module):
    """DenseNet 风格的单层: bottleneck 1x1 卷积 + CAMLayer / One dense layer with a bottleneck + CAM.

    DenseNet 的设计: 每层先用 1x1 卷积把 (越来越宽的) 输入通道压到固定的 ``bn_channels``
    瓶颈, 再过 CAMLayer 算出 ``out_channels`` (= growth_rate) 个新特征。Block 里会把这些
    新特征不断拼接到输入上 (见 CAMDenseTDNNBlock), 所以输入通道随层数线性增长, 用 1x1
    bottleneck 控制计算量。
    Bottleneck 1x1 conv compresses the growing input, then CAMLayer produces growth-rate new channels.

    参数 (Args):
        in_channels: 当前层的输入通道 (随 dense 拼接而增长)。
        out_channels: 本层新产出的通道数 (即 growth_rate)。
        bn_channels: bottleneck 中间通道数。
        kernel_size/stride/dilation/bias: 传给内部 CAMLayer 的卷积参数 (padding 由奇数核自动算)。
        memory_efficient: 训练时是否用 gradient checkpointing 省显存 (用算力换内存)。

    形状 (Shape): (B, in_channels, T) -> (B, out_channels, T)。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, (
            f"Expect equal paddings, but got even kernel size ({kernel_size})"
        )
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)  # 1x1 bottleneck 降维
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        """bottleneck 前半段 (norm/act + 1x1 降维), 单独拆出以便 checkpoint / pre-activation bottleneck."""
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            # gradient checkpointing: 前向不存激活, 反向重算 bn_function, 省显存 / recompute in backward.
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        return self.cam_layer(self.nonlinear2(x))


class CAMDenseTDNNBlock(nn.ModuleList):
    """DenseNet 块: 多个 CAMDenseTDNNLayer 的 dense 连接 / A dense block of CAM TDNN layers.

    DenseNet 的标志性结构: 第 i 层的输入是"原始输入 + 前面所有层的输出"在通道维拼接后的
    结果, 所以第 i 层的 ``in_channels = in_channels + i * out_channels``。每层只新增
    ``out_channels`` (= growth_rate) 个通道, 但每层都能直接访问之前所有特征 (特征复用),
    梯度也更易回传。Block 输出通道 = in_channels + num_layers * out_channels。
    Each layer sees the concatenation of all previous outputs (feature reuse, growth-rate per layer).

    参数 (Args):
        num_layers: 块内 CAMDenseTDNNLayer 的层数 (CAMPPlus 用 12/24/16)。
        in_channels: 进入本块的通道数。
        out_channels: 每层的 growth_rate。
        其余: 透传给每个 CAMDenseTDNNLayer。
    """

    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                # 第 i 层输入通道 = 初始通道 + 前 i 层各 growth_rate 个新通道 / dense growth.
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"tdnnd{i + 1}", layer)

    def forward(self, x):
        for layer in self:
            # dense 连接: 把本层新特征拼回输入, 通道数随层数累加 / concat new features back onto x.
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    """过渡层: dense block 之间用 1x1 卷积压通道 / Transition layer between dense blocks.

    一个 dense block 后通道会膨胀得很多 (in + num_layers*growth_rate); TransitLayer 用
    1x1 卷积把通道砍半 (CAMPPlus 里 ``channels // 2``), 控制网络宽度。注意这里是
    **pre-activation** 顺序: 先 norm/激活再卷积。
    A 1x1 conv that halves the channel count after a dense block (pre-activation order).
    """

    def __init__(
        self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"
    ):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)  # 先 norm/激活 (pre-activation)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)  # 1x1 压通道

    def forward(self, x):
        x = self.nonlinear(x)
        return self.linear(x)


class DenseLayer(nn.Module):
    """1x1 卷积 + 非线性, 兼容 2D/3D 输入的全连接式层 / 1x1 conv + activation, accepts (B,C) or (B,C,T).

    用 1x1 Conv1d 等价于在通道维上做线性变换 (逐帧全连接)。CAMPPlus 把它放在 stats
    pooling 之后作为最后的 embedding 投影层 (输出 512 维 x-vector)。它额外兼容二维输入:
    若传入 (B, C) 这种已无时间维的张量, 会临时升/降一个维度以复用同一套卷积权重。
    Doubles as the final embedding projection; transparently handles 2D inputs via squeeze/unsqueeze.
    """

    def __init__(
        self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"
    ):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            # (B, C) 无时间维: 临时补一个长度 1 的时间维过 Conv1d, 再 squeeze 回去 / treat as T=1.
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)


class BasicResBlock(nn.Module):
    """二维 ResNet 残差块, 给前端 FCM 下采样 fbank 用 / 2D ResNet basic block for the FCM frontend.

    与 TDNN 那批一维层不同, 这是 **二维** 卷积块: CAMPPlus 的前端 FCM (见 campplus.py)
    把 fbank 当成单通道图像 (B, 1, F, T) 处理, 用几个 BasicResBlock 在 **频率维** 上做下采样
    (``stride=(stride, 1)`` 只在频率维降采样、时间维保持), 再展平进 TDNN 主干。结构是标准
    ResNet basic block: conv-bn-relu × 2 + 恒等/投影 shortcut, 最后整体 ReLU。
    Standard ResNet basic block; downsamples only the frequency axis via stride=(stride, 1).

    属性 (Attr):
        expansion: 残差块输出通道相对 planes 的倍率 (basic block 恒为 1)。
    参数 (Args):
        in_planes / planes: 输入/输出通道数 (二维特征图的通道)。
        stride: 频率维下采样步长 (时间维固定为 1)。
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()  # 默认恒等捷径 (空 Sequential 即直通) / identity shortcut
        if stride != 1 or in_planes != self.expansion * planes:
            # 当下采样或通道数变化时, 主路与捷径形状对不上, 用 1x1 卷积把捷径投影到相同形状
            # / projection shortcut to match shape when stride or channels change.
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差相加: 主路 + 捷径 / residual add before final activation
        return F.relu(out)
