# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.
"""BigVGAN 系连续潜在声码器 / AudioVAE(continuous-latent vocoder)。

本文件做什么 / What this file does
----------------------------------
实现 dots.tts 的 **AudioVAE**：一个 BigVGAN 风格的神经声码器，但工作在
**连续潜在(continuous latent)空间**而非离散音频 token 上。
- `Encoder`(`extract_latents`)：把 24/48 kHz 波形下采样压成连续 latent (B, D, T)，
  D = `latent_dim`；可选 VAE 采样(`do_sample`，重参数化 m_q + ε·exp(logs_q))。
- `Decoder`(`inference_from_latents`)：把 latent 经 transposed-conv 上采样 + AMP
  残差块重建回波形 (B, 1, samples)。无离散 codebook、无 argmax，全程连续。
这正是 dots.tts 与 Higgs v3 等离散 token TTS 的根本差异：自回归主干 + flow-matching
DiT 在这个 **连续 latent 空间**里预测 velocity field，最后由本声码器解码成声音。

在数据流里的位置 / Where it sits in the pipeline
-----------------------------------------------
训练：waveform --Encoder--> latent(目标分布) ；推理：DiT 采样出 latent --Decoder--> waveform。
所以本文件是「latent ↔ 波形」的两端编解码器，是整条 TTS 链路的最后一棒。

关键类/函数清单 / Key classes & functions
-----------------------------------------
- `AudioVAE`：顶层模块，聚合 encoder / 互信息(mi)层 / pre&post 投影 / decoder，
  并提供整段推理(`inference*`)与 **流式(streaming)** 接口(`init_stream_state` /
  `stream_step` / `stream_flush` / `compiled_stream_step`)。
- `Encoder` / `Decoder`：下采样编码器 / 上采样解码器(Decoder 即 BigVGAN 主体)。
- `AMPBlock1` / `AMPBlock2`：Anti-aliased Multi-Periodicity(AMP)残差块，
  把 Snake/SnakeBeta 周期激活与 anti-aliasing(alias-free)重采样夹在膨胀卷积之间。
- `ResStack`：编码器侧的膨胀残差栈。
- `SLSTM`：流式 LSTM —— 既有标准 `forward`(整段)，又有手写逐帧 `stream_step`
  (按层索引权重、显式维护 (h, c) 隐状态)以支持 double-streaming。
- `Conv1d_S`：带 weight_norm、可选 causal 左 padding 的 1D 卷积。
- `BigVGANStreamState` / `DecoderStreamState`：流式状态容器(LSTM 隐状态 + 解码窗口)。
- 流式核心技巧：用 `Fraction` 精确累加各层 **左上下文(left context)**，据此开一个
  滑动 latent 窗口，每步 scatter 新帧、gather 出固定长窗口送解码器，再按 lookahead
  切出「已稳定」的音频帧 —— 整个图固定，可 torch.compile。

注：本文件改编自 NVIDIA BigVGAN(MIT)，dots.tts 在其上加了连续 latent VAE 头与
流式解码逻辑。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from dots_tts.modules.backbone.layers import Conv1d, ConvTranspose1d
from dots_tts.modules.vocoder.alias_free_act import Activation1d, Snake, SnakeBeta
from dots_tts.modules.vocoder.config import AudioVAEConfig


@dataclass(slots=True)
class BigVGANStreamState:
    """流式解码的整体状态 / Whole streaming state for one utterance.

    在 double-streaming(主干流式 + 声码器流式)里，调用方每来一小段 latent 就
    调一次 `AudioVAE.stream_step`，所有跨步要记住的东西都装在这里：
    - `lstm_hidden`: `dec_mi_layer` 里那个 SLSTM 的 (h, c) 隐状态，形状
      ((num_layers, B, H), (num_layers, B, H))，逐 step 滚动更新。
    - `decoder`: 解码器侧的滑动窗口状态(见 `DecoderStreamState`)。
    `slots=True` 让 dataclass 用 __slots__，省内存且字段写死、不可乱加属性。
    """

    lstm_hidden: tuple[torch.Tensor, torch.Tensor]
    decoder: "DecoderStreamState"


@dataclass(slots=True)
class DecoderStreamState:
    """解码器滑动窗口状态 / Sliding-window state for the BigVGAN decoder.

    解码器是有左上下文(left context)的因果网络：要正确解出当前 chunk 的音频，
    必须连同它前面若干历史 latent 帧一起喂。这里用一个 **定长窗口** `window`
    充当 latent ring buffer：
    - `window`: (B, latent_dim, window_size) 的历史 latent 缓存；每步把新 chunk
      scatter 进去、再 gather 回定长窗口(见 `_append_stream_decoder_input_tensor`)。
    - `chunk_size`: 每步喂入的 latent 帧数，流式过程中必须固定(便于固定计算图)。
    - `total_frames`: 至今累计喂入的 latent 帧数(可超过 window_size)。
    - `emitted_frames`: 至今已经「定稿」输出给调用方的 latent 帧数(对应音频样本数 =
      emitted_frames × hop_size)。total 与 emitted 之差就是因 lookahead 暂时扣住、
      尚未稳定的尾部帧。
    """

    window: torch.Tensor
    chunk_size: int
    total_frames: int = 0
    emitted_frames: int = 0

def _empty_chunk(
    ref: torch.Tensor,
    *,
    channels: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """造一个时间维长度为 0 的空张量 / make a (B, C, 0) empty chunk.

    流式输出时若本步还没有任何「已稳定」音频可吐(全被 lookahead 扣住)，就返回这个
    长度 0 的占位张量，方便调用方无脑 `torch.cat` 拼接而不必特判 None。
    形状照搬 `ref` 的 batch 维与 device，channels 默认沿用 ref 的通道数。
    """
    return ref.new_zeros(
        (ref.size(0), channels or ref.size(1), 0),
        dtype=dtype or ref.dtype,
    )


def _module_state_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """探测某个子模块的参数 device/dtype / infer where a module's tensors live.

    初始化流式状态张量(零窗口等)时，需要和模块权重放在同一 device、同一 dtype
    (尤其是 fp16/bf16 推理)。优先看常见的 weight/bias/filter 属性，找不到再退而
    遍历 parameters() / buffers() 取第一个张量；彻底没有张量才报错。
    """
    for name in ("weight", "bias", "filter"):
        tensor = getattr(module, name, None)
        if tensor is not None:
            return tensor.device, tensor.dtype
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        return tensor.device, tensor.dtype
    raise RuntimeError(f"Unable to infer state dtype/device for {type(module).__name__}.")


def _stream_state_zeros(
    batch_size: int,
    channels: int,
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """分配一块清零的流式状态张量 / allocate a zeroed (B, C, length) state buffer."""
    return torch.zeros(
        (batch_size, channels, max(0, int(length))),  # 防御性 clamp：length 不为负
        device=device,
        dtype=dtype,
    )


def init_weights(m, mean=0.0, std=0.01):
    """对所有 Conv* 层做 N(mean, std) 权重初始化 / Gaussian-init every conv layer.

    用 `module.apply(init_weights)` 递归调用：只要类名里含 "Conv"(Conv1d /
    ConvTranspose1d / Conv1d_S …)就把 weight 重置为正态分布。BigVGAN 的标准做法，
    给上采样卷积一个小方差的起点。
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Conv1d_S(nn.Module):
    "Conv1d for spectral normalisation and orthogonal initialisation"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        groups=1,
        causal=False,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        # 非 causal：用「same」对称 padding，输出长度≈输入长度；
        # causal：内部 conv 不 padding(pad=0)，改在 forward 里只往左补(见下)。
        pad = 0 if causal else dilation * (kernel_size - 1) // 2
        # causal 模式下需要在 forward 手动补的左侧零数量 = 感受野-1。
        self.causal_pad = dilation * (kernel_size - 1) if causal else 0

        self.layer = weight_norm(  # weight_norm 把权重重参数化为方向+幅度，稳训练
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=groups,
            )
        )

    def forward(self, inputs):
        # 只在左侧 pad：保证输出第 t 帧只依赖 ≤t 的输入，不偷看未来 → 因果卷积。
        if self.causal and self.causal_pad > 0:
            inputs = F.pad(inputs, (self.causal_pad, 0))
        return self.layer(inputs)


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.

    流式 LSTM(Streaming LSTM)。封装 `nn.LSTM`，对外提供两条路径：
    - `forward`：整段一次跑完(训练 / 非流式推理)，隐状态由 PyTorch 内部管理、用完即弃。
    - `stream_step`：手写逐帧 LSTM(见该方法)，把 (h, c) 隐状态显式吐回给调用方跨
      step 续传，从而支持流式增量解码。这是为什么要在 `__init__` 里把每层的
      weight_ih/weight_hh/bias_ih/bias_hh 按层索引缓存成 tuple —— `F.linear`
      手动展开门控时要直接拿这些权重，避免每帧再去 getattr。
    设计动机：`nn.LSTM` 不暴露逐时间步、可外部续传隐状态的接口(尤其在 jit/compile 下)，
    所以流式分支必须自己实现 LSTM cell 的递推。
    `skip=True` 时整体走残差(y = lstm(x) + x)，稳定深层堆叠。
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.skip = skip
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=dimension,
            hidden_size=dimension,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        # 下面把 nn.LSTM 内部每层的权重/偏置按层号取出存成 tuple，供 stream_step
        # 逐帧手动计算门控时直接引用(它们仍是同一批 Parameter，训练照样更新)。
        self._stream_num_layers = num_layers
        self._stream_weight_ih = tuple(  # 输入→门控权重 W_ih，每层一个
            getattr(self.lstm, f"weight_ih_l{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self._stream_weight_hh = tuple(  # 隐状态→门控权重 W_hh，每层一个
            getattr(self.lstm, f"weight_hh_l{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self._stream_bias_ih = tuple(
            getattr(self.lstm, f"bias_ih_l{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self._stream_bias_hh = tuple(
            getattr(self.lstm, f"bias_hh_l{layer_idx}")
            for layer_idx in range(num_layers)
        )
        if self.bidirectional:
            # 双向时拼接两个方向(2*dim)再投回 dim；注意流式只支持单向(见 stream_step)。
            self.proj_out = nn.Linear(dimension * 2, dimension)

    def forward(self, x):
        # 非流式：整段交给 cuDNN LSTM，隐状态自动从 0 初始化、不外传。x: (B, T, dim)
        y, _ = self.lstm(x)
        if self.bidirectional:
            y = self.proj_out(y)
        if self.skip:
            y = y + x  # 残差直连
        return y

    def stream_step(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """逐帧手写 LSTM 递推，并把隐状态外传 / manual LSTM unroll for streaming.

        参数 / Args:
            x: (B, T, dim) 本步要处理的若干 latent 帧(T 可为 chunk 长度)。
            hidden: 上一步返回的 (h, c)，形状各 (num_layers, B, dim)；首步传 None
                则零初始化。
        返回 / Returns:
            (y, (next_h, next_c))，y 与 x 同形 (B, T, dim)，next_h/next_c 供下一步续传。
        为什么手写：见类 docstring —— nn.LSTM 不给「外部续传隐状态的逐步接口」，
        流式必须自己展开标准 LSTM cell 的四门(input/forget/cell/output)递推。
        """
        if self.bidirectional:
            # 双向需要看到未来帧，与因果流式互斥，直接禁止。
            raise RuntimeError("Streaming only supports unidirectional SLSTM.")

        residual = x  # 留作末尾的 skip 残差
        if hidden is None:
            hidden = self.init_stream_state(x.size(0))

        hidden_h, hidden_c = hidden
        next_hidden_h = []
        next_hidden_c = []
        # 逐层堆叠：第 layer 层的输入是第 layer-1 层的全时序输出(下面的 x 在每层末尾更新)。
        for layer_idx in range(self._stream_num_layers):
            layer_input = x
            hx = hidden_h[layer_idx]  # 该层进入本步时的 h_{t-1}
            cx = hidden_c[layer_idx]  # 该层进入本步时的 cell c_{t-1}
            outputs = []
            weight_ih = self._stream_weight_ih[layer_idx]
            weight_hh = self._stream_weight_hh[layer_idx]
            bias_ih = self._stream_bias_ih[layer_idx]
            bias_hh = self._stream_bias_hh[layer_idx]

            # 沿时间维逐帧推进：每帧把输入贡献 + 隐状态贡献相加得到 4 门拼接 logits。
            for frame_idx in range(x.size(1)):
                gates = F.linear(layer_input[:, frame_idx, :], weight_ih, bias_ih)
                gates = gates + F.linear(hx, weight_hh, bias_hh)
                # PyTorch LSTM 的门顺序固定为 i,f,g,o，按这个切 4 块。
                input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
                input_gate = torch.sigmoid(input_gate)
                forget_gate = torch.sigmoid(forget_gate)
                cell_gate = torch.tanh(cell_gate)
                output_gate = torch.sigmoid(output_gate)
                cx = forget_gate * cx + input_gate * cell_gate  # cell 更新
                hx = output_gate * torch.tanh(cx)  # 隐状态输出
                outputs.append(hx)

            x = torch.stack(outputs, dim=1)  # 该层整段输出 (B, T, dim)，作为下层输入
            next_hidden_h.append(hx)  # 保存该层最后一帧的 (h, c) 供下一步续传
            next_hidden_c.append(cx)

        y = x
        if self.skip:
            y = y + residual
        # 把各层最后一帧的 h/c 沿层维 stack 回 (num_layers, B, dim)，与 forward 的隐状态布局一致。
        return y, (torch.stack(next_hidden_h, dim=0), torch.stack(next_hidden_c, dim=0))

    def init_stream_state(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """零初始化流式隐状态 / build zeroed (h, c) on the right device & dtype.

        形状 (num_layers*num_directions, B, hidden_size)，与 nn.LSTM 隐状态约定一致；
        借 weight_hh_l0 的 new_zeros 保证 device/dtype 跟权重对齐。
        """
        num_directions = 2 if self.bidirectional else 1
        state_shape = (
            self.lstm.num_layers * num_directions,
            batch_size,
            self.lstm.hidden_size,
        )
        weight = self.lstm.weight_hh_l0
        return (
            weight.new_zeros(state_shape),
            weight.new_zeros(state_shape),
        )


class ResStack(nn.Module):
    """编码器侧膨胀残差栈 / dilated residual stack used by the Encoder.

    堆 `nums` 个残差子块，第 i 块用膨胀率 dil = base**i(指数增长以快速扩大感受野)。
    每块结构: LeakyReLU → 膨胀 conv → LeakyReLU → conv，外加残差直连(forward 里 x + layer(x))。
    causal 模式下不用对称 padding，改用 ConstantPad1d 只往左补，保证因果性。
    """

    def __init__(self, channel, kernel_size=3, base=3, nums=4, causal=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(nums):
            dil = base**i  # 膨胀率指数增长：1, base, base^2, ...
            # causal: 左补 = 感受野-1；非 causal: 用对称 same padding(pad1=dil, pad2=1)。
            pad1 = dil * (kernel_size - 1) if causal else dil
            pad2 = (kernel_size - 1) if causal else 1
            block = [
                nn.LeakyReLU(),
            ]
            if causal and pad1 > 0:
                block.append(nn.ConstantPad1d((pad1, 0), 0.0))
            block.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=dil,
                        padding=0 if causal else pad1,
                    )
                )
            )
            block.append(nn.LeakyReLU())
            if causal and pad2 > 0:
                block.append(nn.ConstantPad1d((pad2, 0), 0.0))
            block.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=0 if causal else pad2,
                    )
                )
            )
            self.layers.append(nn.Sequential(*block))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # 逐块残差累加
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=100,
        base_channels=12,
        proj_kernel_size=3,
        stack_kernel_size=3,
        stack_dilation_base=2,
        stacks=6,
        channels=(12, 24, 48, 96, 192, 384, 768),
        down_sample_factors=(2, 2, 2, 2, 4, 4),
        causal=False,
        lookahead=0,
    ):
        """AudioVAE 编码器 / waveform → continuous latent.

        把 (B, 1, samples) 波形逐级下采样到 (B, out_channels, T_latent)：
        通道按 `channels` 逐级翻倍、时间按 `down_sample_factors` 逐级压缩，每级是
        「跨步卷积下采样 + ResStack 膨胀残差 + LeakyReLU」。总下采样率 = ∏down_sample_factors
        = hop_size(每个 latent 帧对应这么多波形样本)。`lookahead>0` 时末层用一个
        非因果、kernel=2*lookahead+1 的投影 conv 允许看一点未来以提升质量。
        """
        super().__init__()

        act_slope = 0.2
        layers = []
        # pre proj_layer
        layers += [
            Conv1d_S(
                in_channels,
                base_channels,
                kernel_size=proj_kernel_size,
                stride=1,
                causal=causal,
            ),
            nn.LeakyReLU(act_slope, True),
        ]

        # channels: [512, 256, 128, 64], upsample_factors: [5, 2, 2]
        # pairwise(channels) 给出相邻通道对 (in_c, out_c)，与对应下采样率配对建一级。
        for (in_c, out_c), down_f in zip(
            itertools.pairwise(channels), down_sample_factors, strict=True
        ):
            layers += [
                Conv1d_S(
                    in_c, out_c, kernel_size=down_f * 2, stride=down_f, causal=causal
                ),
                ResStack(
                    out_c, stack_kernel_size, stack_dilation_base, stacks, causal=causal
                ),
                nn.LeakyReLU(act_slope, True),
            ]

        # post layers
        if lookahead > 0:
            layers += [
                Conv1d_S(
                    channels[-1],
                    out_channels,
                    kernel_size=lookahead * 2 + 1,
                    stride=1,
                    causal=False,
                ),
            ]
        else:
            layers += [
                Conv1d_S(
                    channels[-1],
                    out_channels,
                    kernel_size=proj_kernel_size,
                    stride=1,
                    causal=causal,
                ),
            ]
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions, _z_inputs=None):
        # conditions: (B, in_channels, samples) 波形 → (B, out_channels, T_latent)。
        # _z_inputs 仅为接口兼容，本编码器不用。
        return self.generator(conditions)


class AMPBlock1(torch.nn.Module):
    """Anti-aliased Multi-Periodicity 残差块(BigVGAN 默认款) / AMP residual block.

    BigVGAN 的核心积木。每个块内有 3 对 (convs1[i] 膨胀卷积 + convs2[i] 普通卷积)，
    每个卷积前夹一个 **alias-free 的 Snake/SnakeBeta 周期激活**(`Activation1d`，内部先
    上采样→激活→低通下采样以抑制周期非线性引入的混叠/aliasing)。
    forward 里 convs1 用奇数位激活、convs2 用偶数位激活(见 acts1/acts2 切片)，
    全块走残差 x = x + 卷积结果。这种「周期激活 + anti-aliasing」正是 BigVGAN 相对
    HiFi-GAN 提升高频质量的关键。
    `dilation=(1,3,5)`：convs1 三层用不同膨胀率扩大感受野；convs2 固定 dilation=1。
    """

    def __init__(
        self,
        h,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        activation=None,
        causal=True,
    ):
        super().__init__()
        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        causal=causal,
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, causal=causal
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, causal=causal
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, causal=causal
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # total number of conv layers

        if (
            activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        # 激活按奇偶位分给两组卷积：acts1 配 convs1(膨胀)、acts2 配 convs2(普通)。
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            xt = a1(x)   # alias-free 周期激活
            xt = c1(xt)  # 膨胀卷积
            xt = a2(xt)
            xt = c2(xt)  # 普通卷积
            x = xt + x   # 残差
        return x

    def remove_weight_norm(self):
        # 推理前剥掉 weight_norm 的重参数化，融合成普通卷积权重以加速/省显存。
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(torch.nn.Module):
    """AMP 残差块的轻量版 / lighter AMP block (single conv per sub-layer).

    与 AMPBlock1 同理但每个子层只有 1 个卷积(共 2 层、dilation=(1,3))，参数更省。
    `resblock == "2"` 时启用。结构: 激活 → 卷积 → 残差，循环 num_layers 次。
    """

    def __init__(
        self, h, channels, kernel_size=3, dilation=(1, 3), activation=None, causal=True
    ):
        super().__init__()
        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        causal=causal,
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # total number of conv layers

        if (
            activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations, strict=True):
            xt = a(x)   # alias-free 周期激活
            xt = c(xt)  # 卷积
            x = xt + x  # 残差
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class Decoder(nn.Module):
    """AudioVAE 解码器(BigVGAN 主体) / continuous latent → waveform.

    把 (B, latent_dim, T) 的连续 latent 一级级 **转置卷积上采样** 回 (B, 1, samples) 波形：
    `conv_pre`(非因果、带 lookahead 的入口投影) → 多级 [ConvTranspose1d 上采样 +
    `num_kernels` 个 AMP 残差块(结果取平均)] → `activation_post` → `conv_post` → tanh/clamp。
    总上采样率 = ∏upsample_rates = hop_size，正好抵消编码器的下采样。

    流式相关 / Streaming-related
    ---------------------------
    本类还负责算出流式所需的 **左上下文(left context)** 与 **窗口大小**。因为各级
    上采样会改变时间分辨率，把「单帧依赖多少历史」这种量在不同尺度间换算时必须用
    分数(`Fraction`)精确累加(见 `_stream_left_context`)，最后向上取整成整数 latent 帧。
    `stream_lookahead` 则是 conv_pre 的非因果半窗(要看的未来帧数)。
    """

    def __init__(self, h):
        super().__init__()
        self.h = h
        causal = h.causal
        # chunk_size → window_size 的缓存，避免每步重算左上下文(见 stream_window_size)。
        self._stream_window_sizes: dict[int, int] = {}

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # 入口投影看的未来帧数(非因果半窗)，决定 stream_lookahead = (kernel-1)//2。
        num_decoder_lookahead = h.get("num_decoder_lookahead", 2)
        # pre conv
        self.conv_pre = weight_norm(
            Conv1d(
                h.latent_dim,
                h.upsample_initial_channel,
                kernel_size=2 * num_decoder_lookahead + 1,
                stride=1,
                causal=False,
            )
        )

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == "1" else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(h.upsample_rates, h.upsample_kernel_sizes, strict=True)
        ):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(
                h.resblock_kernel_sizes, h.resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(
                    resblock(h, ch, k, d, activation=h.activation, causal=causal)
                )

        # post conv
        if (
            h.activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(
                activation=activation_post, causal=causal, fixed_filter=False
            )
        elif (
            h.activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(
                activation=activation_post, causal=causal, fixed_filter=False
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, causal=causal, bias=h.get("use_bias_at_final", True))
        )

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, z):
        # z: (B, latent_dim, T_latent) → 返回 (B, 1, samples) 波形。
        # pre conv
        x = self.conv_pre(z)

        for i in range(self.num_upsamples):
            # upsampling：第 i 级转置卷积，时间维放大 upsample_rates[i] 倍。
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks：本级的 num_kernels 个残差块各跑一遍，输出求和后取平均(多周期融合)。
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels  # 多核结果平均

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.h.get("use_tanh_at_final", True):
            x = torch.tanh(x)  # 软压到 (-1, 1)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    @property
    def stream_lookahead(self) -> int:
        """流式向前看的 latent 帧数 / non-causal half-window of conv_pre.

        conv_pre 是非因果卷积，要看左右各 (kernel-1)//2 帧；右边那半就是「未来」，
        所以每一步结尾要扣住这么多尾帧不输出，等下一步补足上下文后再定稿(见
        `_slice_stream_audio_window` 里 stable_end 的计算)。
        """
        return (self.conv_pre.kernel_size[0] - 1) // 2

    @staticmethod
    def _conv1d_left_context(layer) -> int:
        """一个 Conv1d 需要的左侧历史帧数 / left receptive field of a Conv1d.

        causal 卷积要往左看 dilation*(kernel-1) 帧；非 causal 用对称 same padding 时
        左上下文就等于它的 padding。这些量后面会按当时的时间分辨率换算累加。
        """
        dilation = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
        kernel_size = (
            layer.kernel_size[0]
            if isinstance(layer.kernel_size, tuple)
            else layer.kernel_size
        )
        if getattr(layer, "causal", False):
            return dilation * (kernel_size - 1)
        return layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding

    @staticmethod
    def _convtranspose1d_left_context(layer) -> int:
        """因果转置卷积的左上下文(以输入帧计) / left context of a causal upsampler.

        只支持 kernel_size == 2*stride 的因果 ConvTranspose1d(见 backbone 的约束)；
        这种规整上采样器对应的左侧依赖正好是 1 个输入帧。其它配置直接报错，避免悄悄算错。
        """
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        kernel_size = (
            layer.kernel_size[0]
            if isinstance(layer.kernel_size, tuple)
            else layer.kernel_size
        )
        if not getattr(layer, "causal", False):
            raise NotImplementedError("Streaming only supports causal ConvTranspose1d.")
        if kernel_size != 2 * stride:
            raise ValueError(
                "Streaming ConvTranspose1d expects kernel_size == 2 * stride, got "
                f"kernel_size={kernel_size} stride={stride}."
            )
        return 1

    @classmethod
    def _activation_left_context(cls, activation: Activation1d) -> int:
        """alias-free 周期激活的左上下文(以外层帧计) / left context of one Activation1d.

        `Activation1d` 内部先 up_ratio 上采样 → Snake 激活 → 同 ratio 低通下采样。
        上采样滤波器和下采样低通各贡献 (kernel-1) 帧左侧依赖，但这些发生在「放大 ratio 倍」
        的中间分辨率上；除以 ratio 并向上取整((total_left + ratio - 1)//ratio)换算回
        激活外部的帧数。只支持因果(causal)且右侧无 padding 的配置，否则报错。
        """
        upsample = activation.upsample
        downsample = activation.downsample.lowpass
        # 流式要求严格因果：上采样 causal、下采样有左 padding 且右侧 padding 为 0。
        if not upsample.causal or not downsample.padding or downsample.pad_right != 0:
            raise NotImplementedError("Streaming only supports causal alias-free activations.")
        ratio = int(upsample.ratio)
        if ratio != int(downsample.stride):
            raise ValueError(
                "Alias-free activation expects matched up/down ratios, got "
                f"up_ratio={ratio} down_ratio={downsample.stride}."
            )
        total_left = (upsample.kernel_size - 1) + (downsample.kernel_size - 1)
        # +ratio-1 再整除 = 向上取整(ceil division)，把中间分辨率的依赖换算回外层帧。
        return (total_left + ratio - 1) // ratio

    @classmethod
    def _ampblock_left_context(cls, block) -> int:
        """一个 AMP 残差块的总左上下文 / accumulated left context through an AMP block.

        AMP 块内是「激活 → 卷积」串联，其总左上下文等于沿这条串联路径把每个激活/卷积的
        左上下文逐个相加(因为它们都不改变这一级的时间分辨率)。AMPBlock1 走 convs1+convs2
        两条并联子层各一条串联路径求和；AMPBlock2 只有一组。类型不符直接 TypeError。
        """
        if isinstance(block, AMPBlock1):
            left_context = 0
            acts1 = block.activations[::2]
            acts2 = block.activations[1::2]
            for conv1, conv2, act1, act2 in zip(
                block.convs1,
                block.convs2,
                acts1,
                acts2,
                strict=True,
            ):
                left_context += (
                    cls._activation_left_context(act1)
                    + cls._conv1d_left_context(conv1)
                    + cls._activation_left_context(act2)
                    + cls._conv1d_left_context(conv2)
                )
            return left_context
        if isinstance(block, AMPBlock2):
            left_context = 0
            for conv, activation in zip(block.convs, block.activations, strict=True):
                left_context += (
                    cls._activation_left_context(activation)
                    + cls._conv1d_left_context(conv)
                )
            return left_context
        raise TypeError(f"Unsupported resblock type: {type(block).__name__}.")

    def _stream_left_context(self) -> int:
        """折算整条解码链所需的左上下文(以输入 latent 帧计) / total left context.

        难点所在。解码器逐级上采样会成倍提高时间分辨率，所以「某层需要 N 帧历史」在
        不同级上对应的「输入 latent 帧数」不同。这里用 `current_scale`(一个 `Fraction`)
        记录「当前级 1 帧 = 多少 latent 输入帧」：每过一级上采样(stride 倍)，分辨率涨
        stride 倍，于是 current_scale 除以 stride。把每层的本地左上下文乘上当时的
        current_scale 累加，就得到统一折算到输入帧的总左上下文。
        用 Fraction 而非 float 是为了精确(整除链上避免浮点误差)，最后向上取整成整数帧。
        """
        left_context = Fraction(self._conv1d_left_context(self.conv_pre), 1)
        current_scale = Fraction(1, 1)  # 起点：1 输入帧 = 1 输入帧
        for stage_idx, upsample_layers in enumerate(self.ups):
            for upsample in upsample_layers:
                # 上采样器的左上下文以「该级输入帧」计，乘 current_scale 折回输入帧。
                left_context += current_scale * self._convtranspose1d_left_context(
                    upsample
                )
                stride = (
                    upsample.stride[0]
                    if isinstance(upsample.stride, tuple)
                    else upsample.stride
                )
                current_scale /= int(stride)  # 过此上采样后分辨率涨 stride 倍

            # 本级的 AMP 残差块(num_kernels 个)在已上采样的分辨率上工作，取最大左上下文。
            stage_start = stage_idx * self.num_kernels
            stage_end = stage_start + self.num_kernels
            stage_context = max(
                self._ampblock_left_context(block)
                for block in self.resblocks[stage_start:stage_end]
            )
            left_context += current_scale * stage_context

        # 末端的 post 激活与 post 卷积在最高(波形)分辨率上，同样折回输入帧后累加。
        left_context += current_scale * self._activation_left_context(self.activation_post)
        left_context += current_scale * self._conv1d_left_context(self.conv_post)
        return int(left_context.__ceil__())  # 向上取整到整数 latent 帧

    def stream_window_size(self, chunk_size: int) -> int:
        """给定每步 chunk，算解码所需的滑动窗口帧数 / required window length.

        窗口 = 本步 chunk_size + 向前看的 lookahead + 历史左上下文。窗口里必须同时容下
        「要解的这段 + 它依赖的全部历史 + 非因果要看的未来」，解码出来才与整段解码一致。
        结果按 chunk_size 缓存，固定 chunk 时只算一次。
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        cached = self._stream_window_sizes.get(chunk_size)
        if cached is not None:
            return cached

        window_size = chunk_size + self.stream_lookahead + self._stream_left_context()
        self._stream_window_sizes[chunk_size] = window_size
        return window_size

    def remove_weight_norm(self):
        for upsample_layers in self.ups:
            for upsample_layer in upsample_layers:
                remove_weight_norm(upsample_layer)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class AudioVAE(nn.Module):
    """连续潜在音频 VAE 顶层 / continuous-latent audio VAE (codec + vocoder).

    串起整套连续 latent 编解码：
        waveform --audio_encoder--> --enc_mi_layer(SLSTM)--> --pre_proj-->  latent(可 VAE 采样)
        latent  --post_proj--> --dec_mi_layer(SLSTM)--> --decoder--> waveform
    `enc_mi_layer`/`dec_mi_layer` 是夹在编解码两端的 SLSTM 互信息(mi)层(Linear→SLSTM→Linear)，
    用来在 latent 序列上建立时序依赖。`pre_proj` 把通道翻倍成 (mean, log-std) 以支持 VAE
    重参数化采样；`post_proj` 在解码前做一次 1×1 投影。

    关键尺度 / Key scales:
        - `hop_size` = ∏downsample_rates：1 个 latent 帧 ↔ 多少波形样本。
        - `latent_dim`：连续 latent 的通道数 D。
    本类同时提供整段推理(`inference` / `extract_latents` / `inference_from_latents`)与
    **严格因果流式**(`init_stream_state` → 多次 `stream_step` → `stream_flush`，以及
    可 compile 的纯函数版 `compiled_stream_step`)。
    """

    def __init__(self, h: AudioVAEConfig):
        super().__init__()
        self.config = h

        self.h = h
        # hop_size = 所有下采样率连乘 = 编码器把波形压缩的总倍率 = 每帧 latent 对应的样本数。
        self.hop_size = int(np.prod(h.downsample_rates))
        self.sample_rate = h.sample_rate
        self.decoder_lookahead = int(h.get("num_decoder_lookahead", 2))

        self.audio_encoder = Encoder(
            out_channels=h.latent_dim,
            down_sample_factors=h.downsample_rates,
            channels=h.downsample_channels,
            causal=h.causal_encoder,
            lookahead=h.get("num_encoder_lookahead", 2),
        )

        # mi 层(mutual-information layer)：Linear 升维 → SLSTM 建时序依赖 → Linear 降回。
        # enc 端处理编码后的 latent，dec 端处理解码前的 latent；流式时 SLSTM 走 stream_step。
        intermediate_size = h.latent_dim * 4
        self.enc_mi_layer = nn.Sequential(
            nn.Linear(h.latent_dim, intermediate_size),
            SLSTM(intermediate_size, num_layers=h.mi_num_layers),
            nn.Linear(intermediate_size, h.latent_dim),
        )
        self.dec_mi_layer = nn.Sequential(
            nn.Linear(h.latent_dim, intermediate_size),
            SLSTM(intermediate_size, num_layers=h.mi_num_layers),
            nn.Linear(intermediate_size, h.latent_dim),
        )
        self.pre_proj = Conv1d(  # 1×1 conv：通道翻倍 → (mean, log-std)，供 VAE 采样
            in_channels=h.latent_dim,
            out_channels=h.latent_dim * 2,
            kernel_size=1,
            stride=1,
        )
        self.post_proj = Conv1d(  # 解码前的 1×1 投影，通道不变
            in_channels=h.latent_dim, out_channels=h.latent_dim, kernel_size=1, stride=1
        )

        self.decoder = Decoder(h)

    def inference(self, data):
        """一站式编码再解码 / encode then decode in one call (round-trip).

        data["sample"] 是波形 → 抽 latent → 立即解码回波形，用于自重建/评测。
        """
        latents = self.extract_latents(data["sample"])
        return {"sample": self.inference_from_latents(latents)}

    @torch.autocast(enabled=False, device_type="cuda")
    def extract_latents(self, x, do_sample=False):
        """波形 → 连续 latent / encode waveform to continuous latent.

        x: (B, 1, samples) → 返回 (B, latent_dim*2, T) 的 (mean, log-std) 拼接；
        若 do_sample 则做 VAE 重参数化采样得到 (B, latent_dim, T)。
        强制 float32(autocast 关闭)以保证 VAE 统计量数值稳定。
        permute 是因为 SLSTM 要 (B, T, C) 布局而卷积是 (B, C, T)，故来回转。
        """
        x = x.float()  # 关 autocast + 转 fp32，保 VAE 统计稳定
        x = self.audio_encoder(x)
        x = x.permute(0, 2, 1)        # (B, C, T) → (B, T, C) 喂 SLSTM
        x = self.enc_mi_layer(x)
        x = x.permute(0, 2, 1)        # 转回 (B, C, T) 给 1×1 conv
        x = self.pre_proj(x)          # 通道翻倍成 (mean, log-std)
        if do_sample:
            # VAE 重参数化：从 N(m_q, exp(logs_q)^2) 采样，保持可微。
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        return x

    def inference_from_latents(self, x, do_sample=True, noise_scale=1.0):
        """连续 latent → 波形 / decode latent back to waveform.

        x: 若 do_sample 则形状 (B, latent_dim*2, T)(含 mean/log-std)，先采样降到
        latent_dim；否则直接是 (B, latent_dim, T)。`noise_scale` 缩放采样噪声(<1 更确定)。
        DiT 在推理时通常吐出确定 latent，这里多以 do_sample 控制是否再注入 VAE 噪声。
        """
        if do_sample:
            assert x.size(1) == self.h.latent_dim * 2, (
                f"Input must be like [B, D, H], got {x.shape}"
            )
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q) * noise_scale
        else:
            assert x.size(1) == self.h.latent_dim, (
                f"Input must be like [B, D, H], got {x.shape}"
            )
        x = self.post_proj(x)
        x = x.permute(0, 2, 1)    # (B, C, T) → (B, T, C) 喂 SLSTM
        x = self.dec_mi_layer(x)
        x = x.permute(0, 2, 1)    # 转回 (B, C, T) 给 decoder
        return self.decoder(x)

    def _validate_stream_latents(self, latents: torch.Tensor) -> None:
        """校验流式输入 latent 形状 / sanity-check (B, latent_dim, frames)."""
        if latents.ndim != 3:
            raise ValueError(
                "Streaming latents must have shape [batch, latent_dim, frames], "
                f"got {tuple(latents.shape)}."
            )
        if latents.size(1) != self.h.latent_dim:
            raise ValueError(
                f"Streaming latent_dim must be {self.h.latent_dim}, got {latents.size(1)}."
            )

    def init_stream_state(
        self,
        batch_size: int = 1,
        chunk_size: int = 8,
    ) -> BigVGANStreamState:
        """开一个流式会话的初始状态 / create fresh streaming state.

        先按 chunk_size 算好定长解码窗口大小，再分配一块清零的 latent 窗口与零初始化的
        SLSTM 隐状态，全部对齐到 decoder.conv_pre 的 device/dtype。**严格流式要求声码器
        causal**(否则无法只靠历史+有限 lookahead 增量解码)，非 causal 直接报错。
        注意 dec_mi_layer[1] 才是那一层 Sequential 里的 SLSTM(索引 0/2 是两个 Linear)。
        """
        if not self.h.causal:
            raise RuntimeError("Strict streaming requires a causal vocoder.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        window_size = self.decoder.stream_window_size(chunk_size)
        device, dtype = _module_state_device_dtype(self.decoder.conv_pre)
        return BigVGANStreamState(
            lstm_hidden=self.dec_mi_layer[1].init_stream_state(batch_size),  # [1] = SLSTM
            decoder=DecoderStreamState(
                window=_stream_state_zeros(
                    batch_size,
                    self.h.latent_dim,
                    window_size,
                    device=device,
                    dtype=dtype,
                ),
                chunk_size=int(chunk_size),
            ),
        )

    def _decode_stream_latents(
        self,
        latents: torch.Tensor,
        lstm_hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """流式跑 post_proj + dec_mi_layer / pre-decoder stage for one chunk.

        把这一步 latent 经 post_proj → dec_mi_layer(Linear → 流式 SLSTM → Linear)，
        其中 SLSTM 走 `stream_step` 续传隐状态(整段 forward 在这里不能用)。返回准备好喂
        decoder 的张量(转回 (B, C, T) 且转成 decoder 权重的 dtype)和更新后的隐状态。
        latents 强制 fp32 保数值稳定，与整段 extract/inference 一致。
        """
        self._validate_stream_latents(latents)
        latents = latents.float()
        x = self.post_proj(latents)
        x = x.permute(0, 2, 1)            # (B, C, T) → (B, T, C) 喂 mi 层
        x = self.dec_mi_layer[0](x)       # 第一个 Linear
        x, lstm_hidden = self.dec_mi_layer[1].stream_step(  # SLSTM 流式逐帧 + 续传隐状态
            x, lstm_hidden
        )
        x = self.dec_mi_layer[2](x)       # 第二个 Linear
        decoder_dtype = self.decoder.conv_pre.weight.dtype
        # 转回 (B, C, T) 并对齐 decoder 的 dtype(可能是 fp16/bf16)再交给 decoder。
        return x.permute(0, 2, 1).to(dtype=decoder_dtype), lstm_hidden

    def _prepare_stream_decoder_input(
        self,
        latents: torch.Tensor,
        state: BigVGANStreamState,
    ) -> torch.Tensor:
        """有状态封装 / stateful wrapper: 解一步并把新隐状态写回 state."""
        decoder_input, state.lstm_hidden = self._decode_stream_latents(
            latents,
            state.lstm_hidden,
        )
        return decoder_input

    def _append_stream_decoder_input_tensor(
        self,
        decoder_input: torch.Tensor,
        window: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> torch.Tensor:
        """把新 chunk 追加进定长滑动窗口(全张量、固定计算图) / append chunk into the ring window.

        难点·重点。目标：维护一个长度固定 = window_size 的 latent 窗口，左对齐存「最近的
        有效历史帧」，右侧未填满处补零。**关键约束**是整个过程不能用依赖运行期数值的
        Python 控制流(否则无法 torch.compile / 形成固定图)，所以用 scatter_ + gather +
        mask 这套纯张量操作来表达「左移并追加」。

        参数 / Args:
            decoder_input: (B, latent_dim, chunk_size) 本步新 latent。
            window: (B, latent_dim, window_size) 当前窗口(左对齐有效 + 右侧零填充)。
            valid_frames: 标量张量，窗口里当前真正有效(非零填充)的帧数。
        返回 / Returns:
            更新后的 (B, latent_dim, window_size) 窗口。

        步骤:
          1) combined = [window | zeros(chunk_size)]，临时扩成 window_size+chunk_size，
             把新帧 scatter_ 写到紧接有效区之后的位置(insert_index)。
          2) new_valid = min(valid + chunk, window_size)；start = 超出窗口的溢出量(需左移多少)。
          3) gather 从 start 开始取 window_size 帧(实现「左移丢弃最旧帧」)，再用 mask 把
             尚未填满的右侧清零。
        """
        if window.dtype != decoder_input.dtype:
            window = window.to(dtype=decoder_input.dtype)
        chunk_size = int(decoder_input.size(-1))
        if chunk_size >= window.size(-1):
            raise ValueError(
                f"decoder window size {window.size(-1)} must be larger than chunk_size {chunk_size}."
            )
        # 目标窗口每个位置的下标 0..window_size-1，dtype 跟 valid_frames 对齐以便比较/索引。
        positions = torch.arange(
            window.size(-1),
            device=window.device,
            dtype=valid_frames.dtype,
        )
        clipped_valid = valid_frames.clamp(min=0, max=window.size(-1))
        # 在右侧临时多接 chunk_size 个零位，给新帧腾出写入空间(避免越界)。
        combined = torch.cat(
            [window, decoder_input.new_zeros(window.size(0), window.size(1), chunk_size)],
            dim=-1,
        )
        # 新帧写到「现有有效区之后」连续 chunk_size 个位置。
        insert_index = clipped_valid + torch.arange(
            chunk_size,
            device=window.device,
            dtype=valid_frames.dtype,
        )
        combined.scatter_(
            -1,
            insert_index.view(1, 1, -1).expand_as(decoder_input),
            decoder_input,
        )
        new_valid = (clipped_valid + chunk_size).clamp(max=window.size(-1))  # 追加后的有效帧数(封顶)
        # start>0 表示有效区已超过 window_size，需要丢掉最旧的 start 帧(整体左移)。
        start = (clipped_valid + chunk_size - window.size(-1)).clamp(min=0)
        gather_index = (start + positions).clamp(max=combined.size(-1) - 1)  # clamp 防越界(尾部会被 mask 掉)
        gathered = combined.gather(
            -1,
            gather_index.view(1, 1, -1).expand_as(window),
        )
        # 窗口还没填满时(new_valid<window_size)，把右侧无效位置清零，保证零填充语义。
        mask = (positions < new_valid).to(dtype=window.dtype).view(1, 1, -1)
        return gathered * mask

    def _append_stream_decoder_input(
        self,
        decoder_input: torch.Tensor,
        state: BigVGANStreamState,
    ) -> torch.Tensor:
        """有状态版追加 / stateful append: 更新 state 的 window 与 total_frames.

        chunk_size 必须与初始化时一致(固定图前提)，否则报错。valid_frames 取
        min(total_frames, window_size)：窗口装不下时只算「窗口内」有效帧。
        """
        decoder_state = state.decoder
        chunk_size = int(decoder_input.size(-1))
        if chunk_size != decoder_state.chunk_size:
            raise ValueError(
                f"Streaming chunk_size must stay fixed at {decoder_state.chunk_size}, got {chunk_size}."
            )
        window = decoder_state.window
        valid_frames = min(decoder_state.total_frames, window.size(-1))
        valid_frames_tensor = window.new_tensor(valid_frames, dtype=torch.int64)
        new_window = self._append_stream_decoder_input_tensor(
            decoder_input,
            window,
            valid_frames_tensor,
        )
        decoder_state.window = new_window
        decoder_state.total_frames += chunk_size  # 累计喂入帧数(可超 window_size)
        return new_window

    def compiled_stream_step(
        self,
        latents: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """纯函数版单步流式 / pure-tensor stream step for torch.compile.

        与 `stream_step` 等价，但所有状态都以张量参数显式进出、不碰 dataclass，
        因而整个函数是无副作用的纯计算图，适合 torch.compile / CUDA graph 捕获。
        输入隐状态 (h, c) 与窗口、有效帧数；输出整窗音频 + 新隐状态 + 新窗口。
        切片(slice out 稳定帧)由外部用 total/emitted 计数另行完成。
        """
        decoder_input, (hidden_h, hidden_c) = self._decode_stream_latents(
            latents,
            (hidden_h, hidden_c),
        )
        new_window = self._append_stream_decoder_input_tensor(
            decoder_input,
            window,
            valid_frames,
        )
        audio_window = self.decode_stream_window(new_window)
        return audio_window, hidden_h, hidden_c, new_window

    def decode_stream_window(self, window: torch.Tensor) -> torch.Tensor:
        """对整个 latent 窗口跑一遍 decoder / decode the whole window to audio.

        返回 (B, 1, window_size*hop_size) 的音频窗；其中只有「稳定」的一段会被切出输出
        (见 `_slice_stream_audio_window`)，窗口左侧历史与右侧 lookahead 区是为正确性陪跑。
        """
        decoder_dtype = self.decoder.conv_pre.weight.dtype
        if window.dtype != decoder_dtype:
            window = window.to(dtype=decoder_dtype)
        return self.decoder(window)

    def _slice_stream_audio_window(
        self,
        audio_window: torch.Tensor,
        state: BigVGANStreamState,
        *,
        final: bool,
    ) -> torch.Tensor:
        """从整窗音频里切出本步要吐的「已稳定」片段 / slice out newly-stable audio.

        非 final 步：窗口最右 `stream_lookahead` 个 latent 帧依赖未来、还可能变，先扣住
        不输出，所以 stable_end = total - lookahead；final(flush)步把剩余尾帧全部定稿。
        只输出 (emitted_frames, stable_end) 这段新帧对应的样本，再把 emitted 推进到 stable_end，
        保证各步首尾相接、不重不漏。把帧坐标换算到样本坐标用 hop_size 乘。
        """
        decoder_state = state.decoder
        stable_end = (
            decoder_state.total_frames
            if final
            else max(0, decoder_state.total_frames - self.decoder.stream_lookahead)
        )
        if stable_end <= decoder_state.emitted_frames:
            # 本步没有新稳定帧(全被 lookahead 扣住)，返回长度 0 占位张量。
            return _empty_chunk(audio_window, channels=1)

        # 把「全局帧坐标」换算到「当前窗口内的局部坐标」：窗口只存最近 valid_frames 帧。
        window_size = decoder_state.window.size(-1)
        valid_frames = min(decoder_state.total_frames, window_size)
        window_start = decoder_state.total_frames - valid_frames  # 窗口左端对应的全局帧号
        if decoder_state.emitted_frames < window_start:
            # 已发出的帧竟落在窗口左端之外 → 窗口太短，待发帧被挤掉了，固定图无法补救。
            raise RuntimeError(
                "Decoder stream window is too short for fixed-graph decoding."
            )

        local_start = decoder_state.emitted_frames - window_start
        local_end = stable_end - window_start
        sample_start = local_start * self.hop_size  # 帧 → 样本
        sample_end = local_end * self.hop_size
        decoder_state.emitted_frames = stable_end  # 推进已发出计数
        return audio_window[..., sample_start:sample_end]

    def stream_step(
        self,
        latents: torch.Tensor,
        state: BigVGANStreamState,
    ) -> torch.Tensor:
        """流式主循环的一步 / one streaming step (有状态、对外接口).

        latents: (B, latent_dim, chunk_size) 本步新 latent；返回本步可吐的稳定音频
        (B, 1, n_samples)，n_samples 可能为 0。一步内串起:
        post_proj+SLSTM → 追加滑窗 → 整窗解码 → 切出稳定段。
        """
        decoder_input = self._prepare_stream_decoder_input(latents, state)
        window = self._append_stream_decoder_input(decoder_input, state)
        audio_window = self.decode_stream_window(window)
        return self._slice_stream_audio_window(audio_window, state, final=False)

    def stream_flush(self, state: BigVGANStreamState) -> torch.Tensor:
        """收尾 / flush: 解码当前窗口并吐出之前被 lookahead 扣住的全部尾帧.

        流式结束时调用一次：不再追加新 latent，直接对现存窗口解码并以 final=True 切出
        到 total_frames 为止的所有剩余样本。
        """
        audio_window = self.decode_stream_window(state.decoder.window)
        return self._slice_stream_audio_window(audio_window, state, final=True)

    def remove_weight_norm(self):
        # 推理前剥掉 decoder 各卷积的 weight_norm 重参数化(融合加速)。
        self.decoder.remove_weight_norm()
