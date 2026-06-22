"""Semantic encoder over continuous AudioVAE latents — 连续潜在的"语义/patch 编码器"。

本文件做什么 / What this file does
----------------------------------
dots.tts 是连续潜在(continuous-latent) TTS：AudioVAE(BigVGAN) 把波形编/解码到一个
连续 latent 空间(没有离散音频 token)。本文件实现 ``VAESemanticEncoder``——把高帧率的
AudioVAE latent 序列 **下采样 + 打包成 patch**，再用一个 **因果(causal) transformer**
编码成给自回归主干(Qwen2.5-1.5B backbone)消费的、更低帧率的 patch 级嵌入(embedding)。
"Semantic" 指这里产出的是供 LM 条件/对齐用的高层表示，而非声学细节。

在推理/训练数据流里的位置 / Position in the pipeline
----------------------------------------------------
   waveform --(AudioVAE encode)--> latent (B, T, in_dim)
            --(VAESemanticEncoder)--> patch embedding (B, T', out_dim)
            --> 自回归 backbone (作为音频侧的输入/条件表示)

下采样链路 / Downsampling chain:
  - ``ds_proj``: 因果 Conv1d，stride=2，把 latent 帧率降到 1/2 (in_ds_rate)。
  - ``encoder``: ``SuperviseEncoder``(若干层因果 ``TransformerEncoderLayer``)在 1/2 帧率上做注意力。
  - ``_project_embeddings``: 把每 ``out_ds_rate`` 个 token 在通道维拼起来再线性投影，
    最终帧率降到 1/patch_size。``patch_size = in_ds_rate * out_ds_rate``。

两套调用路径 / Two call paths:
  - ``forward``: 整段并行编码(训练 / 一次性编码 prompt)。
  - ``prefill`` + ``decode_patch``: double-streaming 流式推理，借助 KV cache 与一段
    ``conv_tail`` 卷积历史，逐 patch 增量编码而结果与并行版等价。

关键类清单 / Key classes
------------------------
  - ``SemanticEncoderDecodeState``: 流式解码状态(卷积历史 + 每层 KV cache + 已写长度)。
  - ``TransformerEncoderLayer``: 因果 self-attention + FFN 的单层(含 mask 构造/融合 + 流式 decode_step)。
  - ``SuperviseEncoder``: 多层 ``TransformerEncoderLayer`` 堆叠，管理 KV cache 生命周期。
  - ``VAESemanticEncoder``: 顶层，下采样 + patch 打包 + 因果 transformer 编码。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dots_tts.modules.backbone.layers import Conv1d, Mlp, MultiHeadAttention


@dataclass
class SemanticEncoderDecodeState:
    """流式解码状态 / Streaming decode state for ``VAESemanticEncoder``.

    在 double-streaming 推理中跨步保存的可变状态，使逐 patch 的增量编码与一次性
    并行编码(``forward``)在数值上等价。包含两类历史：

    Attributes:
        conv_tail: 因果下采样 conv(``ds_proj``)的 **左侧卷积历史**，形状
            (B, in_channels, left_padding)。下一步把它拼到新 latent 前面，
            替代并行版里的左 padding，保证跨 patch 边界的卷积感受野连续。
        layer_caches: 每个 transformer 层的 KV cache，元组
            ``(key_cache, value_cache)`` × num_layers；每个 cache 形状
            (B, num_heads, max_seq_len, head_dim)。原地(in-place)按位置写入。
        seq_len: 已写入 KV cache 的有效 token 数(下一次写入的起始位置)。
    """

    conv_tail: torch.Tensor
    layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seq_len: int


class TransformerEncoderLayer(nn.Module):
    """单层 pre-norm transformer 编码块 / One pre-norm transformer encoder layer.

    结构 = ``x + Attn(Norm(x))`` 然后 ``x + FFN(Norm(x))``，即 **pre-norm 残差**
    (norm 在子层之前)。注意力支持因果(causal) + padding 两种 mask 的融合。
    提供两条前向：``forward``(整段并行)与 ``decode_step``(KV cache 流式单步)。

    Args:
        hidden_size: 隐藏维 D；须与输入通道一致。
        num_heads: multi-head attention 头数。
        ffn_hidden_size: FFN 中间维(SiLU 激活)。
        attn_dropout / ffn_dropout: 注意力 / FFN dropout。
        norm_layer: ``nn`` 下的归一化类名(如 "LayerNorm")，按名取类。
    """

    def __init__(
        self,
        hidden_size,
        num_heads=16,
        ffn_hidden_size=4096,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        norm_layer="LayerNorm",
        **kwargs,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_drop=attn_dropout,
            norm_layer=norm_layer,
            **kwargs,
        )
        norm_cls = getattr(nn, norm_layer)  # 按字符串名取归一化类(如 nn.LayerNorm)
        self.attn_norm = norm_cls(hidden_size)
        self.ffn = Mlp(
            hidden_size, ffn_hidden_size, dropout=ffn_dropout, act_layer=nn.SiLU
        )
        self.ffn_norm = norm_cls(hidden_size)
        self.hidden_size = hidden_size

    def _build_causal_mask(self, T: int, device):
        """构造下三角因果 mask (T, T)；True=允许 attend。

        ``tril`` 保证位置 i 只能看见 j<=i，禁止看未来——这是把编码器变成因果
        (streaming-friendly)的关键。
        """
        return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    def _build_padding_mask(self, x_lens, max_len: int, device):
        """构造 padding mask (B, max_len)；True=有效 token、False=padding。

        把每个样本真实长度 ``x_lens`` 与位置索引比较，标出哪些位置是真实内容。
        """
        B = x_lens.size(0)
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        return positions < x_lens.unsqueeze(1)

    def _fuse_attn_mask(self, causal_mask, padding_mask):
        """把因果 mask 与 padding mask 融合成一个 (B, T, T) 的注意力 mask。

        约定 True=可 attend、False=屏蔽，交给下游 ``MultiHeadAttention`` 转成 -inf bias。
        padding 由 1D (B,T) 经 **外积** 升成 2D：``row & col`` 同时屏蔽 padding 的
        query 行与 key 列。返回值可能为 (B,T,T) 或广播用的 (1,T,T)，由 attention 内部
        再展开到 num_heads。

        Returns:
            None(无任何 mask) / (1, T, T)(仅因果) / (B, T, T)(含 padding 或两者)。
        """
        if causal_mask is None and padding_mask is None:
            return None
        if causal_mask is None:
            # 仅 padding：外积得到 (B,T,T)，行/列任一为 padding 即屏蔽
            row = padding_mask.unsqueeze(2)
            col = padding_mask.unsqueeze(1)
            return row & col
        if padding_mask is None:
            return causal_mask.unsqueeze(0)  # 仅因果：加 batch 维 (1,T,T) 供广播

        _B, _T = padding_mask.shape
        causal = causal_mask.unsqueeze(0)  # (1,T,T)，将与 (B,T,T) 广播相与
        row = padding_mask.unsqueeze(2)  # (B,T,1) query 维
        col = padding_mask.unsqueeze(1)  # (B,1,T) key 维
        pad_2d = row & col  # 外积 -> (B,T,T)
        return causal & pad_2d  # 因果约束 ∧ padding 约束

    def forward(
        self,
        x,
        x_lens=None,
        causal=True,
    ):
        """整段并行前向 / Full-sequence parallel forward.

        Args:
            x: (B, T, C) 输入序列，C 须等于 ``hidden_size``。
            x_lens: (B,) 各样本有效长度；None 表示无 padding。
            causal: 是否启用下三角因果 mask。
        Returns:
            (B, T, C) pre-norm 残差后的输出。
        """
        _B, T, C = x.shape
        assert self.hidden_size == C
        device = x.device

        causal_mask = self._build_causal_mask(T, device) if causal else None
        if x_lens is not None:
            padding_mask = self._build_padding_mask(x_lens, T, device)
        else:
            padding_mask = None
        fused_mask = self._fuse_attn_mask(causal_mask, padding_mask)

        h = self.attn_norm(x)  # pre-norm：norm 在子层前
        h = self.attn(
            q=h,
            mask=fused_mask,
        )
        x = x + h  # 注意力残差

        h = self.ffn_norm(x)
        h = self.ffn(h)
        return x + h  # FFN 残差

    def decode_step(
        self,
        x,
        *,
        cache: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
    ):
        """流式单步前向 / Streaming step using a KV cache.

        与 ``forward`` 等价但只处理新到的一小段(一个或几个 token)：因果约束与对
        padding 的屏蔽都在 ``attn.decode_step`` 内部基于 ``positions`` 与 cache 容量
        完成，故此处不再构造显式 mask。

        Args:
            x: (B, n, C) 本步新输入(n 通常等于 out_ds_rate)。
            cache: 本层 KV cache ``(key_cache, value_cache)``，会被原地更新。
            positions: (n,) 这批 token 在整段序列里的绝对位置(用于 RoPE 与因果判定)。
        Returns:
            ((B, n, C) 输出, 更新后的 cache)。
        """
        if x.size(1) <= 0:
            raise ValueError(
                "TransformerEncoderLayer.decode_step expects a non-empty input."
            )

        h = self.attn_norm(x)
        h, cache = self.attn.decode_step(h, cache=cache, positions=positions)
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        return x + h, cache


class SuperviseEncoder(nn.Module):
    """多层因果 transformer 堆叠 / Stack of ``TransformerEncoderLayer``.

    ``VAESemanticEncoder`` 内部用的 patch 级编码器主体。除整段 ``forward`` 外，还负责
    流式 KV cache 的生命周期：``init_decode_state`` 分配、``reset_decode_state`` 清零、
    ``decode_step`` 逐层增量推进。``causal`` 默认从 config 读，置 True 时编码器只看历史，
    使流式与并行结果一致。

    Args (via config dict):
        hidden_size, num_heads, ffn_hidden_size, norm_layer, num_layers, causal。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 1024)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=self.hidden_size,
                    num_heads=config.get("num_heads", 16),
                    ffn_hidden_size=config.get("ffn_hidden_size", 4096),
                    norm_layer=config.get("norm_layer", "LayerNorm"),
                )
                for _ in range(config.get("num_layers", 6))
            ]
        )
        self.causal = config.get("causal", False)

    def forward(self, x, x_lens=None):
        """整段并行编码 / Run all layers over the full sequence.

        Args:
            x: (B, T, D) 输入。
            x_lens: (B,) 有效长度；None 时按满长度填充(无 padding)。
        Returns:
            (B, T, D) 编码结果。
        """
        batch_size, seq_len, _ = x.shape
        if x_lens is None:
            # 未给长度 -> 视作整段都有效，构造全满 x_lens
            x_lens = torch.full(
                (batch_size,), seq_len, device=x.device, dtype=torch.long
            )
        for layer in self.layers:
            x = layer(x, x_lens=x_lens, causal=self.causal)
        return x

    def init_decode_state(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """为每层分配空 KV cache / Allocate per-layer KV caches for streaming.

        每层一对 (key_cache, value_cache)，形状 (B, num_heads, max_seq_len, head_dim)；
        ``max_seq_len`` 是这段流式会写入的最大 token 数(由上层据 patch 数推得)。

        Returns:
            ``tuple`` of ``(key_cache, value_cache)``，长度 = 层数。
        """
        layer_caches = []
        for layer in self.layers:
            cache_shape = (
                batch_size,
                layer.attn.num_heads,
                max_seq_len,
                layer.attn.head_dim,
            )
            layer_caches.append(
                (
                    torch.zeros(cache_shape, dtype=dtype, device=device),
                    torch.zeros(cache_shape, dtype=dtype, device=device),
                )
            )
        return tuple(layer_caches)

    def reset_decode_state(
        self,
        layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        """原地清零所有层的 KV cache，复用同一份显存开始新一段流式。"""
        if len(layer_caches) != len(self.layers):
            raise ValueError("Layer cache count does not match encoder depth.")
        for key_cache, value_cache in layer_caches:
            key_cache.zero_()
            value_cache.zero_()

    def decode_step(self, x, *, layer_caches, positions: torch.Tensor):
        """逐层流式推进一步 / One streaming step through every layer.

        各层共享同一组 ``positions``(本批 token 的绝对位置)，依次把 ``x`` 喂给每层的
        ``decode_step`` 并原地更新对应 cache。

        Args:
            x: (B, n, D) 本步输入。
            layer_caches: 与层数等长的 KV cache 列表。
            positions: (n,) 绝对位置索引。
        Returns:
            (B, n, D) 编码输出。
        """
        if len(layer_caches) != len(self.layers):
            raise ValueError("Layer cache count does not match encoder depth.")

        for layer, cache in zip(self.layers, layer_caches, strict=True):
            # 各层 cache 原地更新；逐层串联，x 流经整栈
            x, _ = layer.decode_step(x, cache=cache, positions=positions)
        return x


class VAESemanticEncoder(nn.Module):
    """AudioVAE latent -> patch 级语义嵌入 / Patch encoder over continuous latents.

    顶层模块：把 AudioVAE 的连续 latent 序列两级下采样并打包成 patch token，再经因果
    transformer 编码，供自回归 backbone 使用。两级下采样：

      1. ``ds_proj`` 因果 Conv1d(kernel=stride=in_ds_rate=2) 把帧率降到 1/2；
      2. ``encoder`` 在 1/2 帧率上做因果注意力；
      3. ``_project_embeddings`` 把每 ``out_ds_rate`` 个 token 沿通道拼接再线性投影，
         总下采样率达到 ``patch_size = in_ds_rate * out_ds_rate``。

    Args:
        in_dim: 输入 latent 通道数。
        out_dim: 输出 patch 嵌入维度(供 backbone 消费)。
        config: 含 ``patch_size`` 与 ``PatchEncoder``(子编码器配置)。
    """

    def __init__(self, in_dim, out_dim, config):
        super().__init__()
        in_ds_rate = 2  # 第一级(卷积)下采样率，固定为 2
        self.patch_size = int(config.patch_size)
        self.in_ds_rate = in_ds_rate
        self.ds_proj = Conv1d(
            in_dim, in_dim, kernel_size=in_ds_rate, stride=in_ds_rate, causal=True
        )
        self.in_proj = nn.Linear(in_dim, config.PatchEncoder.hidden_size)
        self.encoder = SuperviseEncoder(config.PatchEncoder)
        # 第二级下采样(通道拼接)率：patch_size 在卷积已降 1/2 之外还需再降的倍数
        self.out_ds_rate = self.patch_size // in_ds_rate
        # 输入维 = hidden_size × out_ds_rate，因为打包时把 out_ds_rate 个 token 拼到通道维
        self.out_proj = nn.Linear(
            config.PatchEncoder.hidden_size * self.out_ds_rate, out_dim
        )

    def forward(self, x, x_lens=None):
        """整段并行编码 / Full-sequence encode.

        Args:
            x: (B, T, in_dim) AudioVAE latent。
            x_lens: (B,) 各样本有效长度(在原始 latent 帧率上)；None 表示无 padding。
        Returns:
            (B, T // patch_size, out_dim) patch 级嵌入。
        """
        x = self._downsample(x)  # 卷积下采样 1/2 -> (B, T/2, in_dim)
        x = self.in_proj(x)  # 投到编码器隐藏维
        z = self.encoder(x, x_lens=x_lens)
        return self._project_embeddings(z)  # 再 1/out_ds_rate 打包成 patch

    def init_decode_state(
        self,
        *,
        max_audio_patch_count: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SemanticEncoderDecodeState:
        """初始化流式解码状态 / Build a fresh ``SemanticEncoderDecodeState``.

        分配卷积历史(``conv_tail``，长度 = ``ds_proj.left_padding``)与每层 KV cache。
        KV cache 容量按 ``patch 数 × out_ds_rate`` 计——每个 patch 在编码器层面对应
        ``out_ds_rate`` 个 token。

        Args:
            max_audio_patch_count: 本段流式预计最多生成的 patch 数(决定 cache 容量)。
        """
        return SemanticEncoderDecodeState(
            conv_tail=torch.zeros(
                # 卷积左侧历史：替代并行版的左 padding，保证 patch 边界处感受野连续
                (batch_size, self.ds_proj.in_channels, self.ds_proj.left_padding),
                dtype=dtype,
                device=device,
            ),
            layer_caches=self.encoder.init_decode_state(
                batch_size=batch_size,
                max_seq_len=max_audio_patch_count * self.out_ds_rate,
                device=device,
                dtype=dtype,
            ),
            seq_len=0,
        )

    def reset_decode_state(self, state: SemanticEncoderDecodeState) -> None:
        """原地清零流式状态(卷积历史 + KV cache + 已写长度)，以便复用。"""
        state.conv_tail.zero_()
        self.encoder.reset_decode_state(state.layer_caches)
        state.seq_len = 0

    def prefill(
        self,
        x,
        state: SemanticEncoderDecodeState,
    ) -> tuple[torch.Tensor, SemanticEncoderDecodeState]:
        """流式 prefill：一次性编码 prompt latent 并写入 KV cache。

        对应自回归 prefill 阶段——把已知的整段 prompt latent 编码出 patch 嵌入，同时把
        K/V 灌进 cache、并把卷积尾部存进 ``conv_tail``，使后续逐 patch 的 ``decode_patch``
        能无缝续上。输入长度必须是 ``patch_size`` 的整数倍。

        Args:
            x: (B, L, in_dim) prompt latent，L 须能被 ``patch_size`` 整除。
            state: 由 ``init_decode_state`` 创建的解码状态，会被原地更新。
        Returns:
            ((B, L//patch_size, out_dim) patch 嵌入, 更新后的 state)。
        """
        if x.ndim != 3:
            raise ValueError(
                f"VAESemanticEncoder.prefill expects rank-3 input, got {tuple(x.shape)}."
            )
        if x.size(1) % self.patch_size != 0:
            raise ValueError(
                f"Prompt latent length {x.size(1)} must be divisible by patch_size={self.patch_size}."
            )

        if x.size(1) == 0:
            # 空 prompt：直接返回 0 长度嵌入，state 不变
            return (
                x.new_zeros((x.size(0), 0, self.out_proj.out_features)),
                state,
            )
        if state.conv_tail.size(0) != x.size(0):
            raise ValueError(
                "VAESemanticEncoder.prefill batch size does not match decode state."
            )

        step_inputs = self.in_proj(self._downsample(x))  # (B, L/2, hidden)
        # 编码器层面 token 数 = patch 数 × out_ds_rate；用作完整性校验
        expected_token_count = (x.size(1) // self.patch_size) * self.out_ds_rate
        if step_inputs.size(1) != expected_token_count:
            raise RuntimeError(
                "Patch encoder prefill produced an unexpected token count: "
                f"expected={expected_token_count} actual={step_inputs.size(1)}."
            )

        current_seq_len = state.seq_len  # cache 里已有内容的长度(prefill 通常从 0 起)
        next_seq_len = current_seq_len + step_inputs.size(1)
        cache_capacity = state.layer_caches[0][0].size(2)  # KV cache 第 2 维 = 容量
        if next_seq_len > cache_capacity:
            raise ValueError(
                "Patch encoder prefill exceeds decode-state capacity: "
                f"required={next_seq_len} capacity={cache_capacity}."
            )

        # 绝对位置 = 局部索引 + 已写偏移，决定写入 cache 的槽位与 RoPE 相位
        positions = (
            torch.arange(step_inputs.size(1), device=x.device, dtype=torch.long)
            + current_seq_len
        )
        encoded = self.encoder.decode_step(
            step_inputs,
            layer_caches=state.layer_caches,
            positions=positions,
        )
        embedding = self._project_embeddings(encoded)
        raw = x.transpose(1, 2)  # (B, in_dim, L) 便于沿时间维取尾部
        # 存下最后 left_padding 帧作卷积历史，供下一步 decode_patch 跨边界续接
        state.conv_tail.copy_(raw[..., -self.ds_proj.left_padding :])
        state.seq_len = next_seq_len
        return embedding, state

    def decode_patch(
        self,
        latent_patch,
        conv_tail: torch.Tensor,
        layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """流式逐 patch 编码 / Encode exactly one latent patch incrementally.

        ``prefill`` 之后的增量主循环：每次喂入恰好一个 ``patch_size`` 长的 latent，
        借 ``conv_tail`` 续接卷积历史、借 KV cache 续接注意力历史，产出 1 个 patch 嵌入。
        与 ``forward`` 在数值上等价(因果 + 历史拼接保证)。

        Args:
            latent_patch: (B, patch_size, in_dim) 单个 patch 的原始 latent。
            conv_tail: (B, in_dim, left_padding) 上一步留下的卷积历史。
            layer_caches: 每层 KV cache，原地更新。
            positions: (out_ds_rate,) 本 patch 对应的 out_ds_rate 个 token 的绝对位置。
        Returns:
            ((B, 1, out_dim) patch 嵌入, 新的 conv_tail)。
        """
        if latent_patch.ndim != 3:
            raise ValueError(
                f"VAESemanticEncoder.decode_patch expects rank-3 input, got {tuple(latent_patch.shape)}."
            )
        if latent_patch.size(1) != self.patch_size:
            raise ValueError(
                f"decode_patch expects patch length {self.patch_size}, got {latent_patch.size(1)}."
            )
        if positions.ndim != 1 or positions.size(0) != self.out_ds_rate:
            raise ValueError(
                "decode_patch positions must be a rank-1 tensor matching out_ds_rate."
            )

        # 流式版下采样：手动拼 conv_tail 历史，得到本 patch 的 out_ds_rate 个 token
        step_inputs, conv_tail = self._downsample_step(
            latent_patch,
            conv_tail=conv_tail,
        )
        if step_inputs.size(1) != self.out_ds_rate:
            raise RuntimeError(
                f"Downsample step produced {step_inputs.size(1)} tokens, expected {self.out_ds_rate}."
            )

        encoded = self.encoder.decode_step(
            step_inputs,
            layer_caches=layer_caches,
            positions=positions,
        )
        embedding = self._project_embeddings(encoded)  # 打包成 1 个 patch 嵌入
        return embedding, conv_tail

    def _downsample(self, x):
        """并行版第一级下采样 / Parallel conv downsample.

        ``ds_proj`` 是 Conv1d，作用在 (B, C, T) 上；这里前后各转一次轴以适配
        (B, T, C) 的张量布局。因果 conv 在 ``forward`` 内部自动左 padding。
        """
        return self.ds_proj(x.transpose(1, 2)).transpose(1, 2)

    def _project_embeddings(self, z):
        """第二级下采样 + 投影 / Pack ``out_ds_rate`` tokens then project to out_dim.

        把序列维上每 ``out_ds_rate`` 个相邻 token 沿通道(h)拼接：
        ``(s d) h -> s (d h)``，序列长度降 ``out_ds_rate`` 倍、通道升同倍，再线性投影。
        即"通道拼接式"下采样，无信息丢弃。
        """
        if self.out_ds_rate > 1:
            z = rearrange(z, "b (s d) h -> b s (d h)", d=self.out_ds_rate)
        return self.out_proj(z)

    def _downsample_step(self, latent_patch, *, conv_tail):
        """流式版第一级下采样 / Streaming conv downsample with explicit history.

        不靠 conv 内部的 padding，而是手动把上一步存的 ``conv_tail`` 拼在当前 patch
        前面再做 padding=0 的 ``F.conv1d``——等价于在连续流上滑动卷积，从而保证跨 patch
        边界的输出与并行 ``_downsample`` 完全一致。

        Args:
            latent_patch: (B, patch_size, in_dim) 当前 patch。
            conv_tail: (B, in_dim, left_padding) 上一步留下的左侧历史。
        Returns:
            ((B, out_ds_rate, hidden) 投影后 token, 新的 conv_tail)。
        """
        raw = latent_patch.transpose(1, 2)  # (B, in_dim, patch_size)
        conv_input = torch.cat([conv_tail, raw], dim=-1)  # 前置历史 -> 滑窗连续

        projected = F.conv1d(
            conv_input,
            self.ds_proj.weight,
            self.ds_proj.bias,
            stride=self.ds_proj.stride[0],
            padding=0,  # 不再额外 padding，左侧上下文已由 conv_tail 提供
            dilation=self.ds_proj.dilation[0],
            groups=self.ds_proj.groups,
        ).transpose(1, 2)
        # 取本 patch 末尾 left_padding 帧作为下一步的卷积历史
        new_conv_tail = raw[..., -self.ds_proj.left_padding :]
        return self.in_proj(projected), new_conv_tail
