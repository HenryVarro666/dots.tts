"""通用神经网络基础层 / Common neural-network building blocks for the backbone.

本文件 (What this file does)
------------------------------
为 dots.tts 的声学头(flow-matching DiT, 见同目录 ``dit.py``)与相关网络提供一组
可复用的底层模块。它不实现完整模型，只是"积木":

- :class:`Dropout`       —— 可强制开启的 dropout(用于 CFG 等需要随机置零的场景)。
- :class:`Conv1d`        —— 支持因果(causal)左侧 padding 的 1D 卷积。
- :class:`ConvTranspose1d` —— 支持因果裁剪的 1D 转置卷积(上采样)。
- :class:`Mlp`           —— Transformer FFN 风格的两层 MLP。
- :func:`rotate_half` / :func:`apply_rotary_pos_emb` / :class:`RotaryEmbedding`
  —— RoPE 旋转位置编码(rotary position embedding)的三件套。
- :class:`MultiHeadAttention` —— 多头注意力,支持 RoPE、QK-Norm、self/cross-attn,
  以及流式推理用的 :meth:`MultiHeadAttention.decode_step` (带 KV cache)。

在数据流里的位置 (Where it sits)
--------------------------------
``dit.py`` 的 ``DiTBlock`` 直接 import 这里的 ``Mlp`` 与 ``MultiHeadAttention`` 来
堆叠 Transformer 层;因果 Conv 主要服务于 AudioVAE / 流式编码器这类需要严格时序因果
(causal,即第 t 帧只能看到 ≤t 的输入)的卷积网络。

张量约定 (Tensor conventions)
-----------------------------
- 卷积层用 ``(B, C, T)``: batch、channel、time。
- 注意力/MLP 用 ``(B, L, D)``: batch、序列长度 length、隐藏维度 hidden dim。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Dropout(nn.Module):
    """带 ``force_drop`` 开关的 Dropout / Dropout that can be forced on at inference.

    与原生 ``nn.Dropout`` 的唯一区别: 当 ``force_drop=True`` 时,即使模块处于
    ``eval()`` 模式也照样以概率 ``p`` 置零。这在需要"推理时仍引入随机性"的场景
    (例如某些 CFG / 采样技巧)里有用,普通层则保持训练/推理一致行为。

    Args:
        p:          丢弃概率 dropout probability,取值 [0, 1]。
        inplace:    是否原地修改输入张量。
        force_drop: 为真时无视 ``self.training``,强制启用 dropout。
    """

    def __init__(
        self, p: float = 0.5, inplace: bool = False, force_drop: bool = False, **_kwargs
    ):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.inplace = inplace
        self.force_drop = force_drop

    def forward(self, x, **_kwargs):
        # training 标志决定是否真正执行置零: force_drop 时恒为 True, 否则沿用 nn.Module
        # 的 self.training (train()/eval() 控制)。
        return F.dropout(
            x,
            p=self.p,
            training=True if self.force_drop else self.training,
            inplace=self.inplace,
        )


class Conv1d(nn.Conv1d):
    """支持因果 padding 的 1D 卷积 / Causal-capable 1D convolution.

    继承自 ``nn.Conv1d``,额外支持 ``causal=True``: 此时不在两侧对称补零,而是只在
    序列**左侧**补 ``dilation*(kernel_size-1)`` 个零,使输出第 t 帧只依赖输入的
    ≤t 帧,严格满足时序因果(causal),适合流式 / 自回归场景。

    非因果时则按 ``padding=(kernel*dilation - dilation)/2`` 做"same"式对称补零,
    保持输出时间长度与输入一致(stride=1 情形)。

    输入/输出张量形状均为 ``(B, C, T)``。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        **_kwargs,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                # 因果模式: 卷积本身不补零, 改为 forward 里只在左侧补零, 补的数量等于
                # 感受野左展宽度 dilation*(kernel_size-1)。
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                # 非因果: 对称 "same" padding, 让 stride=1 时输出时间长度不变。
                padding = int((kernel_size * dilation - dilation) / 2)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.in_channels = in_channels

    def forward(self, x):
        if self.causal:
            # 仅在时间维左侧补 left_padding 个零(右侧/前面不补), 实现因果卷积。
            # 借 unsqueeze(2)->4D 让 F.pad 的 (left, right, top, bottom) 只作用到时间维,
            # 再 squeeze 回 (B, C, T)。
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super().forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    """支持因果裁剪的 1D 转置卷积(上采样) / Causal-capable 1D transposed conv.

    转置卷积常用于在 vocoder / decoder 中按 ``stride`` 倍率对时间轴上采样。
    ``causal=True`` 时要求 ``kernel_size == 2*stride`` 且 ``padding==0``,并在
    ``forward`` 末尾裁掉输出右端的 ``stride`` 帧——这些尾帧依赖"未来"输入, 去掉后
    才能保证因果性(causal),适配流式生成。

    输入/输出张量形状为 ``(B, C, T)``,输出时间维约为输入的 ``stride`` 倍。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
        **_kwargs,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, (
                "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."
            )

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        self.causal = causal
        self.stride = stride

    def forward(self, x):
        x = super().forward(x)
        if self.causal:
            # 转置卷积会在右端多吐出依赖未来输入的 stride 帧, 裁掉以维持因果性。
            x = x[:, :, : -self.stride]
        return x


class Mlp(nn.Module):
    """Transformer FFN 风格的两层 MLP / Two-layer feed-forward network.

    结构: ``Linear(hidden -> ffn) -> 激活 -> dropout -> Linear(ffn -> hidden) -> dropout``。
    即先升维到 ``ffn_hidden_size`` 做非线性变换, 再降回 ``hidden_size``, 是
    Transformer block 里 attention 之后的逐位置(position-wise)前馈子层。

    Args:
        hidden_size:     输入/输出维度 D。
        ffn_hidden_size: 中间隐藏维度(通常为 D 的若干倍)。
        act_layer:       激活函数类(默认 GELU)。
        dropout:         两处 dropout 的丢弃概率。

    forward 输入/输出形状均为 ``(B, L, D)``;``_mask`` 参数保留接口一致性但未使用。
    """

    def __init__(
        self,
        hidden_size,
        ffn_hidden_size=4096,
        act_layer=nn.GELU,
        dropout=0.0,
        **_kwargs,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.act = act_layer()
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.drop = Dropout(dropout)

    def forward(self, x, _mask=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def rotate_half(x):
    """RoPE 的"旋转一半"辅助函数 / Rotate the second half of the last dim.

    把最后一维切成前后两半 ``(x1, x2)``, 返回 ``(-x2, x1)``。这正是把每一对维度看作
    复平面上的点、做 90° 旋转所需的分量重排, 配合 ``cos/sin`` 即可实现旋转位置编码
    (rotary position embedding, RoPE)。输入输出形状一致。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.autocast(enabled=False, device_type="cuda")
def apply_rotary_pos_emb(pos, t):
    """对 q/k 施加 RoPE 旋转 / Apply rotary position embedding to a tensor.

    公式 ``t*cos(pos) + rotate_half(t)*sin(pos)``, 等价于按位置角度旋转每对特征维,
    使注意力的 dot-product 天然编码相对位置(relative position)。

    用 ``@torch.autocast(enabled=False)`` 关闭混合精度: RoPE 的 cos/sin 旋转对数值精度
    敏感, 强制在 fp32 下计算更稳定。

    Args:
        pos: 预先算好的角度张量(由 :class:`RotaryEmbedding` 给出);若为 3D ``(B, L, d)``
             则补一个 head 维 ``(B, 1, L, d)`` 以便和 4D 的 ``t`` 广播。
        t:   待旋转的 query 或 key, 形状 ``(B, H, L, d)``。
    """
    if pos.dim() == 3:
        # 补出 head 维, 让按位置的旋转角对所有注意力头广播。
        pos = pos.unsqueeze(1)
    return t * pos.cos() + rotate_half(t) * pos.sin()


class RotaryEmbedding(nn.Module):
    """RoPE 角度生成器 / Produces the per-position rotation angles for RoPE.

    依据标准 RoPE 公式预存逆频率 ``inv_freq = 1/theta^(2i/dim)``(注册为非持久 buffer,
    不进 state_dict), forward 时把位置索引 ``t`` 与 ``inv_freq`` 外积得到各位置各频率的
    角度, 再 ``cat`` 复制一份以匹配 :func:`rotate_half` 的"前后两半"布局。

    Args:
        dim:   head_dim, 即每个注意力头的特征维度。
        theta: 旋转基频(base/period), 越大越能编码长程位置。
    """

    def __init__(self, dim, theta=50000):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self._theta = float(theta)

    def _apply(self, fn):
        # 重写 .to()/.cuda() 等迁移逻辑: 让 inv_freq 跟随模块换设备, 但始终保持 fp32,
        # 避免被整体 half()/bfloat16() 降精度而损害 RoPE 数值稳定性。
        inv_freq = self.inv_freq
        super()._apply(fn)
        self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    @torch.autocast(enabled=False, device_type="cuda")
    def forward(self, t):
        """由位置索引计算旋转角度 / Map position indices to RoPE angles.

        Args:
            t: 位置索引。1D ``(L,)`` 表示一条共享序列;2D ``(B, L)`` 表示每个 batch
               各自的位置(如带 KV cache 的流式解码用绝对位置)。

        Returns:
            角度张量, 末维已复制成两份: 1D 入 -> ``(L, dim)``, 2D 入 -> ``(B, L, dim)``。
        """
        inv_freq = self.inv_freq
        if inv_freq.device != t.device:
            raise RuntimeError(
                "RotaryEmbedding buffer device mismatch: "
                f"inv_freq={inv_freq.device} input={t.device}."
            )
        t = t.to(dtype=inv_freq.dtype)
        if t.dim() == 1:
            # 位置 i 与频率 j 的外积 -> (L, dim/2)
            freqs = torch.einsum("i , j -> i j", t, inv_freq)
        else:
            # 带 batch 维的位置外积 -> (B, L, dim/2)
            freqs = torch.einsum("bi, j -> bij", t, inv_freq)
        # 复制成两份拼到 dim, 与 rotate_half 的前/后半切分对齐 -> (..., dim)
        return torch.cat((freqs, freqs), dim=-1)


class MultiHeadAttention(nn.Module):
    """多头注意力 / Multi-head attention.

    标准 multi-head attention, 但额外支持 dots.tts backbone 需要的几项能力:

    - **self / cross attention**: ``forward(q, k, v)`` 中 ``k``/``v`` 缺省回退为 ``q`` 即
      自注意力;传入不同的 ``k``/``v``(且 L≠S) 则为交叉注意力, RoPE 会分别按各自序列长度施加。
    - **RoPE**: ``rotary_bias=True`` 时对 q/k 施加旋转位置编码(见上方 :class:`RotaryEmbedding`)。
    - **QK-Norm**: ``qk_norm=True`` 时对每个 head 的 q/k 做归一化, 提升训练稳定性。
    - **流式解码**: :meth:`decode_step` 配合外部 KV cache 做 double-streaming 增量推理。

    维度记号: ``B`` batch, ``L`` query 长度, ``S`` key/value 长度, ``H`` 头数,
    ``d`` 每头维度(``head_dim = hidden_size // num_heads``)。

    Args:
        hidden_size:  隐藏维度 D, 必须能被 ``num_heads`` 整除。
        num_heads:    注意力头数 H。
        qkv_bias:     q/k/v 投影是否带 bias。
        qk_norm:      是否对 q/k 做 per-head 归一化。
        attn_drop:    注意力权重的 dropout。
        dropout:      输出投影后的 dropout。
        norm_layer:   QK-Norm 用的归一化层名(取自 ``torch.nn``, 如 "LayerNorm")。
        rotary_bias:  是否启用 RoPE。
        rotary_theta: RoPE 基频 theta。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        dropout: float = 0.0,
        norm_layer: str = "LayerNorm",
        rotary_bias: bool = False,
        rotary_theta: float | None = 50000,
        **_kwargs,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            "hidden_size should be divisible by num_heads"
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5  # 1/sqrt(d) 缩放(此处由 SDPA 内部使用)
        self.rotary_bias = rotary_bias

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # 按名字从 nn 取归一化类; 不开 QK-Norm 时用 Identity 占位(无操作)。
        norm_layer = getattr(nn, norm_layer)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.o_dropout = Dropout(dropout)

        if self.rotary_bias:
            self.rotary = RotaryEmbedding(self.head_dim, theta=rotary_theta)

    def forward(self, q, k=None, v=None, mask=None, pos_ids=None, **_kwargs):
        """全序列注意力(训练 / 非流式推理) / Full-sequence attention.

        Args:
            q:       query, ``(B, L, D)``。
            k, v:    key/value, ``(B, S, D)``;为 ``None`` 时回退为 ``q`` 即自注意力。
            mask:    布尔有效位掩码(True=可见)。``(B, S)`` 按 key 维 padding mask, 或
                     ``(B, L, S)`` 任意成对 mask;内部会广播到 ``(B, H, L, S)``。
            pos_ids: 自定义位置索引(仅 L==S 自注意力时生效), 缺省用 ``arange(L)``。

        Returns:
            注意力输出, ``(B, L, D)``。
        """
        k = k or q  # k/v 缺省回退为 q -> 自注意力 self-attention
        v = v or q
        B, L, _ = q.shape
        _, S, _ = v.shape
        if mask is not None:
            # 把不同维度的 mask 统一广播成 SDPA 期望的 (B, H, L, S) 形状。
            if mask.ndim == 2:  # [B, L]
                assert L == S
                mask = rearrange(mask, "b j -> b 1 1 j")
                mask = mask.expand(-1, self.num_heads, L, -1)
            elif mask.ndim == 3:  # [B, L, S]
                assert mask.size(1) == L and mask.size(2) == S
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        # 拆头: (B, N, H*d) -> (B, H, N, d), 让每个 head 在独立子空间做注意力。
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)  # QK-Norm(未开启则为 Identity)

        # Apply rotary
        if self.rotary_bias:
            if L == S:
                # 自注意力: q/k 共用一套位置编码。
                if pos_ids is None:
                    rotary_emb = self.rotary(torch.arange(L, device=q.device))
                else:
                    rotary_emb = self.rotary(pos_ids)
                q, k = (apply_rotary_pos_emb(rotary_emb, tensor) for tensor in (q, k))
            else:
                # 交叉注意力: q 与 k 序列长度不同, 各按自身长度生成 RoPE 角度。
                q_rotary_emb = self.rotary(torch.arange(L, device=q.device))
                k_rotary_emb = self.rotary(torch.arange(S, device=k.device))
                q = apply_rotary_pos_emb(q_rotary_emb, q)
                k = apply_rotary_pos_emb(k_rotary_emb, k)

        # 用加性 attn_bias 表达 mask: 不可见位置加 -inf, softmax 后权重≈0。
        attn_bias = torch.zeros(B, self.num_heads, L, S, dtype=q.dtype, device=q.device)

        if mask is not None:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))

        # 走 PyTorch 融合 SDPA(内部含 1/sqrt(d) 缩放与 softmax); 推理时关 dropout。
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        out = rearrange(out, "b h n d -> b n (h d)")  # 合头还原 (B, L, D)
        return self.o_dropout(self.o_proj(out))

    def decode_step(self, x, *, cache, positions: torch.Tensor):
        """流式增量解码一步(带 KV cache) / One streaming decode step with a KV cache.

        double-streaming 推理时调用: 每步只把当前 block 的新 token 写入预分配的 KV cache,
        并对"截至本步已写入"的所有 key 做因果注意力, 从而避免重算历史。

        Args:
            x:         本次解码块的输入, ``(B, n, D)``, ``n`` 为 block 长度(可>1)。
            cache:     ``(cached_k, cached_v)`` 元组, 各形状 ``(B, H, cache_capacity, d)``,
                       为整段序列预分配好的 KV cache(原地更新)。
            positions: 长度为 ``n`` 的 1D 绝对位置索引, 指明本块各 token 写入 cache 的槽位
                       (同时作为 RoPE 位置)。

        Returns:
            ``(out, cache)`` —— 注意力输出 ``(B, n, D)`` 与(已原地更新的)cache。
        """
        if x.size(1) <= 0:
            raise ValueError("MultiHeadAttention.decode_step expects a non-empty input.")
        if positions.ndim != 1 or positions.size(0) != x.size(1):
            raise ValueError(
                "MultiHeadAttention.decode_step positions must match the decode block length."
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)
        block_len = q.size(2)

        if self.rotary_bias:
            # 用绝对位置 positions 施加 RoPE, 保证 cache 内历史 key 与新 query 位置对齐。
            rotary_emb = self.rotary(positions)
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        cached_k, cached_v = cache
        # 把本块新算的 k/v 原地写入 cache 的对应槽位(沿 time 维 dim=2 按 positions 散布)。
        cached_k.index_copy_(2, positions, k)
        cached_v.index_copy_(2, positions, v)

        cache_capacity = cached_k.size(2)
        # 对 cache 全容量上的每个槽位编一个 key 位置索引 (1, cache_capacity)。
        key_positions = torch.arange(
            cache_capacity,
            device=x.device,
            dtype=torch.long,
        ).unsqueeze(0)
        query_positions = positions.unsqueeze(1)  # (n, 1)
        # 因果掩码: query 只能看 ≤ 自身位置的 key (n, cache_capacity)。
        causal_mask = key_positions <= query_positions
        # 有效掩码: 屏蔽尚未写入的未来槽位(位置 > 本块最后一个 token 的都还是空的)。
        valid_mask = key_positions <= positions[-1]
        attn_bias = torch.zeros(
            q.size(0),
            self.num_heads,
            block_len,
            cache_capacity,
            dtype=q.dtype,
            device=q.device,
        )
        # 两掩码取交集再取反 -> 不可见处填 -inf; unsqueeze 两次广播到 (B, H, n, cap)。
        attn_bias.masked_fill_(
            (causal_mask & valid_mask).unsqueeze(0).unsqueeze(0).logical_not(),
            float("-inf"),
        )

        # query 用本块的 q, key/value 用完整 cache(含历史), 实现增量注意力。
        out = F.scaled_dot_product_attention(
            q,
            cached_k,
            cached_v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_dropout(self.o_proj(out)), cache
