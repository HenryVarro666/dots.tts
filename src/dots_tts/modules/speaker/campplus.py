# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""CAM++ 说话人编码器（speaker encoder）/ x-vector extractor.

本文件实现 3D-Speaker 的 **CAM++** 架构，在 dots.tts 中充当「声纹/音色」编码器：
把一段参考音频压成一个固定维度（默认 512）的 **x-vector / speaker embedding**，
这个向量代表说话人的音色（timbre / voice identity），与说什么内容无关。

在 dots.tts 推理数据流里的位置（zero-shot 音色克隆 / voice cloning）：
    参考音频 waveform
      └─> 16 kHz 重采样 + 80 维 fbank（见 fbank.py，feat_dim=_SPEAKER_FBANK_N_MELS=80）
            └─> CAMPPlus.forward(x)  -> speaker embedding (B, 512)
                  └─> 作为条件（conditioning）注入声学头（flow-matching DiT）
该 embedding 让模型「凭一句话学会一个人的嗓音」：训练时它从目标音频里抽，
推理（zero-shot clone）时只需换成「想模仿的人」的参考音频，DiT 就会朝那个音色生成。
注意这里只产出音色条件，与离散音频 token 无关——dots.tts 走的是连续潜在（continuous-latent）路线。

整体结构（self.xvector 这个 Sequential 的顺序）：
    FCM head            : 2D conv 在频率轴下采样，把 (B,F,T) 当单通道图处理，再展平回 1D 序列
      -> TDNNLayer      : 时延神经网络（time-delay NN，本质带 stride/dilation 的 1D conv）做时间下采样
      -> block1/2/3     : 3 个 CAMDenseTDNNBlock（DenseNet 式特征复用 + CAM 上下文注意力），
                          每个 block 后跟一个 TransitLayer 把通道数减半（压缩）
      -> out_nonlinear  : 收尾的 BN+激活
      -> stats          : StatsPool，沿时间维取 mean 和 std 并拼接 -> 把变长序列汇聚成定长向量
      -> dense          : 线性投影到 embedding_size（512），即最终 x-vector

关键类/函数清单：
    - FCM       : Feature Context Module，2D-conv 频率下采样前端（frequency-axis frontend）。
    - CAMPPlus  : 顶层模型，组装 FCM + TDNN 主干 + StatsPool + 投影头；
                  额外实现了 length-aware 的 masked statistics pooling，支持 batch 内变长输入。
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from dots_tts.modules.speaker.campplus_layers import (
    BasicResBlock,
    CAMDenseTDNNBlock,
    DenseLayer,
    StatsPool,
    TDNNLayer,
    TransitLayer,
    get_nonlinear,
)
from dots_tts.modules.speaker.fbank import _SPEAKER_FBANK_N_MELS


class FCM(nn.Module):
    """Feature Context Module —— 频率轴 2D-conv 前端（frequency-axis frontend）。

    职责：把 fbank 特征 (B, F, T) 当作一张「单通道图」(B, 1, F, T)，
    用若干 2D ResBlock + conv 沿 **频率维 F** 做下采样（共 1/8），同时在通道维提特征，
    最后把「下采样后的频率」折叠进通道维，得到一条 1D 序列 (B, C', T) 交给后面的 TDNN 主干。

    为什么这么设计：纯 1D-TDNN 只把每帧的 80 维 mel 当无结构向量；FCM 先用 2D conv
    在「频率×时间」上做局部感受野卷积（类似图像里的纹理），让网络利用频谱的局部结构
    （共振峰/harmonics 等音色线索），这正是 CAM++ 相比传统 x-vector 的改进点之一。
    关键：只在频率 F 上 stride 下采样，时间 T 始终保持，避免过早丢时序分辨率。

    形状流（feat_dim=80 时）：
        in   : (B, F=80, T)
        +chan: (B, 1, 80, T)            # unsqueeze 出单通道
        layer1 (stride=2 on F): (B, 32, 40, T)
        layer2 (stride=2 on F): (B, 32, 20, T)
        conv2  (stride=2 on F): (B, 32, 10, T)
        flatten F into C      : (B, 32*10=320, T)   # out_channels = m_channels*(feat_dim//8)
    """

    def __init__(
        self,
        block=BasicResBlock,
        num_blocks=(2, 2),
        m_channels=32,
        feat_dim=_SPEAKER_FBANK_N_MELS,
    ):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        # 两个 ResBlock 阶段，各自首层 stride=2 仅作用于频率轴 F（见 BasicResBlock 的 stride=(stride,1)），
        # 时间轴 T 不下采样；合计把 F 降到 1/4。
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        # conv2 再在频率轴上 stride=2（时间轴 stride=1），把 F 总共降到 1/8。
        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        # 折叠后给到 1D 主干的通道数 = 通道数 × 剩余频率格点数（feat_dim 被下采样 8 倍）。
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        # 经典 ResNet 堆叠：只有该 stage 的第一个 block 用给定 stride 下采样，其余 block stride=1。
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # 维护跨 block 的输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,F,T) -> (B,1,F,T)：把 fbank 视作单通道 2D 图
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        # 把 (B, C, F', T) 的频率维 F' 并入通道维 -> (B, C*F', T)，得到喂给 1D TDNN 的序列。
        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])


class CAMPPlus(nn.Module):
    """CAM++ 顶层模型：把变长参考音频特征编码成定长 x-vector（speaker embedding）。

    职责：组装 FCM 前端 + TDNN 主干 + StatsPool + 投影头，端到端把 fbank 序列
    汇聚成一个表征「说话人音色」的固定维度向量，供下游 flow-matching DiT 做 voice conditioning。

    关键参数：
        feat_dim       : 输入 fbank 的 mel 维数（80）。
        embedding_size : 输出 x-vector 维度（512）。
        growth_rate    : DenseTDNN 每层新增的通道数（DenseNet 的 growth rate）。
        bn_size        : bottleneck 系数，bn_channels = bn_size * growth_rate。
        init_channels  : 首个 TDNNLayer 的输出通道数。
        memory_efficient: 训练时对 DenseTDNN 的 bottleneck 用 gradient checkpointing 省显存。

    forward 形状：
        in  : x (B, T, F)  [, lengths (B,)]   # 注意输入是 (B,T,F)，forward 内部会 permute
        out : (B, embedding_size)             # 每条样本一个 x-vector

    设计要点：StatsPool 把整段时间序列压成 mean+std 拼接的向量，这是 x-vector 范式的核心——
    用一阶/二阶统计量做时序汇聚（temporal aggregation），从而对语音「长度无关」，
    天然适合 zero-shot：任意长度的参考音频都能映射到同一空间里的一个音色点。
    本实现额外支持传入 lengths 做 masked pooling，正确处理 batch 内 padding（见下方方法）。
    """

    # 主干第一层 TDNN 的卷积超参（用类常量集中声明，便于 forward 里同步计算下采样后的有效长度）。
    _TDNN_KERNEL_SIZE = 5
    _TDNN_STRIDE = 2
    _TDNN_PADDING = 2

    def __init__(
        self,
        feat_dim=_SPEAKER_FBANK_N_MELS,
        embedding_size=512,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels  # 跟踪「当前通道数」，随主干逐层更新

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            self._TDNN_KERNEL_SIZE,
                            stride=self._TDNN_STRIDE,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        # 3 个 DenseTDNN stage：层数 (12,24,16)、kernel 全 3、dilation (1,2,2)。
        # dilation 递增 = 不增参数地扩大时间感受野（捕捉更长的发音/韵律上下文）。
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=True)
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            # DenseNet 式拼接：block 输出通道 = 输入通道 + num_layers*growth_rate（每层特征都被保留并 concat）。
            channels = channels + num_layers * growth_rate
            # TransitLayer 把膨胀后的通道压缩一半，控制参数量与显存（DenseNet 的 transition 思想）。
            self.xvector.add_module(
                f"transit{i + 1}",
                TransitLayer(
                    channels, channels // 2, bias=False, config_str=config_str
                ),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        # StatsPool：沿时间维取 mean+std 并拼接，把变长 (B,C,T) 汇聚成定长 (B,2C)。
        self.xvector.add_module("stats", StatsPool())
        # 投影头：输入 channels*2（mean 和 std 各 channels 维），输出 embedding_size 的 x-vector。
        # config_str="batchnorm_" 用的是 affine=False 的 BN，对 embedding 做无仿射归一化（稳定声纹分布）。
        self.xvector.add_module(
            "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        )

        # 权重初始化：所有 Conv1d/Linear 用 Kaiming normal（配 ReLU 系激活），bias 清零。
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _conv_output_lengths(lengths, kernel_size, stride=1, padding=0, dilation=1):
        """按 conv1d 公式把「输入有效长度」换算成「卷积后有效长度」。

        套用 PyTorch 卷积输出长度公式：
            L_out = floor((L_in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
        用途：序列经 TDNN 下采样后，原 lengths 不再对应新时间轴，需同步更新，
        否则后面的 masked pooling 会按错误长度去截断有效帧。
        """
        return (
            torch.div(
                lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
                stride,
                rounding_mode="floor",
            )
            + 1
        )

    @staticmethod
    def _make_length_mask(lengths, max_len, device):
        """根据每条样本的有效长度生成布尔 mask (B, max_len)。

        返回 mask[b, t] = (t < lengths[b])，True 表示有效帧、False 表示 padding。
        实现：用一行 broadcast 比较 arange(max_len)[None,:] < lengths[:,None]，无需 Python 循环。
        """
        lengths = lengths.to(device=device, dtype=torch.long).clamp(min=0, max=max_len)
        return torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    def _masked_stats_pooling(self, x, lengths, unbiased=True, eps=1e-2):
        """支持变长输入的 statistics pooling：只对「有效帧」算 mean/std 并拼接。

        参数 x: (B, C, T)，lengths: (B,) 每条样本的有效时间步数。
        返回   : (B, 2C)，即 [mean; std] 沿通道拼接，与 StatsPool 输出语义一致。

        为何需要它：batch 内序列被 padding 到等长，普通 StatsPool 会把补零帧也算进
        统计量，污染 x-vector。这里用 length mask 把 padding 帧置零并从分母里剔除，
        保证统计量只来自真实音频帧。这是正确处理变长 batch 推理的关键。
        """
        # 有效长度 clamp 到 [1, T]：下限 1 避免空序列除零，上限 T 防越界。
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(
            min=1, max=x.size(-1)
        )
        # mask: (B, 1, T)，加 channel 维以便和 x (B,C,T) 广播相乘。
        mask = self._make_length_mask(lengths, x.size(-1), x.device).unsqueeze(1)
        mask = mask.to(dtype=x.dtype)

        # mean：对有效帧求和后除以真实帧数（而非 T），padding 帧已被 mask 乘成 0。
        denom = lengths.to(dtype=x.dtype).view(-1, 1).clamp_min(1.0)
        mean = (x * mask).sum(dim=-1) / denom

        # 去均值后再次乘 mask，确保 padding 位置的偏差不计入方差。
        centered = (x - mean.unsqueeze(-1)) * mask
        # 无偏方差用 (n-1) 作分母；clamp_min(1) 防止单帧样本除零。
        var_denom = (
            (lengths - 1).clamp_min(1).to(dtype=x.dtype).view(-1, 1)
            if unbiased
            else denom
        )
        var = centered.pow(2).sum(dim=-1) / var_denom
        # 方差先 clamp 到 >=eps 再开方：数值稳定，避免 sqrt(0)/sqrt(负数极小值) 的 NaN/inf 梯度。
        std = torch.sqrt(var.clamp_min(eps))
        return torch.cat([mean, std], dim=1)

    def forward(self, x, lengths=None):
        """把 fbank 序列编码成 speaker embedding（x-vector）。

        参数 x: (B, T, F)，T 帧、每帧 F=feat_dim 维 mel；可选 lengths: (B,) 每条样本有效帧数。
        返回   : (B, embedding_size) 的 x-vector。

        lengths 的作用：传入时启用 masked 路径，正确处理 batch 内 padding；
        不传时退化为对整段（含 padding）做普通 StatsPool（单条样本推理时无 padding，可省略）。
        """
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)：转成 conv1d/2d 期望的 (batch, feat, time) 布局
        x = self.head(x)  # FCM 前端：频率下采样并折叠 -> (B, C', T)
        if lengths is not None:
            # FCM 只在频率轴下采样、时间轴不变，故有效时间长度此处保持不变（仅做 dtype/下限规整）。
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1)

        # 顺序遍历 self.xvector 的子模块；对 stats 这一步特殊处理（要按 lengths 走 masked pooling）。
        for name, module in self.xvector.named_children():
            if name == "stats":
                # 汇聚步：有 lengths 用 masked 版剔除 padding，否则用原始 StatsPool。
                x = (
                    self._masked_stats_pooling(x, lengths)
                    if lengths is not None
                    else module(x)
                )
                continue

            x = module(x)
            # 只有第一层 "tdnn" 会沿时间轴下采样（stride=2），需同步更新 lengths；
            # 后续 DenseTDNN/Transit 都是 stride=1，不改变时间长度，故无需再换算。
            if name == "tdnn" and lengths is not None:
                lengths = self._conv_output_lengths(
                    lengths,
                    kernel_size=self._TDNN_KERNEL_SIZE,
                    stride=self._TDNN_STRIDE,
                    padding=self._TDNN_PADDING,
                )

        return x
