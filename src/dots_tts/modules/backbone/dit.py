"""DiT (Diffusion Transformer) 声学头 / acoustic head for dots.tts.

本文件做什么 (What this file does)
----------------------------------
dots.tts 是连续潜在 (continuous-latent) 的自回归 TTS：Qwen2.5-1.5B 自回归主干预测
出每一帧的「上下文向量」，再由这个 DiT 充当 **flow-matching 的 velocity field 预测器**，
在 BigVGAN AudioVAE 的连续 latent 空间里把噪声逐步去噪成声学 latent；没有离散音频 token。

在数据流中的位置 (Position in the pipeline)
-------------------------------------------
    文本/参考音 → Qwen2.5 主干 → 每帧条件向量 g_cond
                                         │
        x_t (含噪 latent) ──┐           │
        timesteps t ────────┤           │
        (meanflow: duration)┤  +g_cond  ▼
                            └──────► DiT.forward ──► velocity v_θ(x_t, t, cond)
                                         │
                  ODE solver 多步积分 ───┘──► 干净声学 latent → AudioVAE 解码 → 24/48kHz 波形

DiT 不直接吃说话人/时间这些全局条件去拼到序列里，而是用 **adaLN (adaptive LayerNorm)
调制 + gating** 把它们注入每一层 —— 这是 DiT/U-ViT 系扩散模型的标准做法：条件只改变
归一化后的 scale/shift 和残差门控，不占用注意力的 token 预算。

两种模式 (Two modes)
--------------------
- ``flow_matching``：标准 flow-matching，条件 = 时间嵌入 (+ 可选 g_cond)。velocity field
  采到的是「瞬时速度」，推理时需要多步 ODE 积分。
- ``meanflow``：MeanFlow 蒸馏模式，额外用一个 ``duration_embedder`` 把「积分区间长度
  duration」编码进条件 c。MeanFlow 预测的是一段区间上的「平均速度」，从而支持极少步
  (甚至 1 步) 采样，用于流式/加速推理。

关键类/函数清单 (Key classes / functions)
-----------------------------------------
- ``modulate``         : adaLN 的 scale/shift 调制算子。
- ``TimestepEmbedder`` : 把标量 timestep (或 duration) 编成 sinusoidal + MLP 嵌入。
- ``FinalLayer``       : 输出层，adaLN 调制后线性投影回 latent 维度。
- ``DiTBlock``         : 单个 DiT block，adaLN 调制 + gating，注入说话人/时间条件。
- ``DiT``              : 顶层模型，串起输入投影、若干 DiTBlock、输出层。
"""

import math

import torch
import torch.nn as nn

from dots_tts.modules.backbone.layers import Mlp, MultiHeadAttention


def modulate(x, shift, scale, **_kwargs):
    """adaLN 调制算子 / adaptive LayerNorm modulation.

    对已做过 (无仿射的) LayerNorm 的 ``x`` 施加逐通道的缩放与平移：
        out = x * (1 + scale) + shift
    其中 ``scale``/``shift`` 由条件向量经 MLP 生成。写成 ``1 + scale`` 是 DiT 的惯例：
    初始化时让生成 scale/shift 的最后一层权重为 0，则 scale=shift=0，调制退化为恒等
    （out = x），保证训练初期 block 近似恒等映射、更稳定 (identity-at-init)。

    形状 (Shapes):
        x     : (B, T, D)  序列特征
        shift : (B, D)     ── unsqueeze(1) 广播到时间维 → (B, 1, D)
        scale : (B, D)
        return: (B, T, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """把标量 timestep 编成条件嵌入 / sinusoidal timestep embedding + MLP.

    扩散/flow-matching 里每个样本有一个连续标量时间 t (噪声水平)。这里先用
    Transformer 式的 **sinusoidal 频率嵌入** 把标量 t 升维成 ``frequency_embedding_size``
    维的向量 (不同频率的 cos/sin)，再过一个 2 层 MLP (Linear-SiLU-Linear) 映射到
    ``hidden_size``，得到送入 adaLN 的条件向量 c。

    在 ``meanflow`` 模式下，同一个类还被复用来编码 duration (积分区间长度)，
    见 ``DiT.__init__`` 里的 ``duration_embedder``。

    设计原因 (Why)：sinusoidal 嵌入对连续标量给出平滑、多尺度的表示，MLP 再做非线性
    变换 —— 这是 DiT/ADM 系扩散模型注入时间条件的标准组件。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """构造 sinusoidal timestep 频率嵌入 / Transformer-style sinusoidal embedding.

        参数 (Args):
            t          : (B,) 标量 timestep（每个样本一个）。
            dim        : 输出嵌入维度 = ``frequency_embedding_size``。
            max_period : 最低频率对应的周期，控制频率覆盖范围。

        返回 (Returns): (B, dim)，前半为 cos、后半为 sin。

        ``freqs`` 是 dim/2 个按对数等距递减的频率；``args`` = t × 频率，外积成 (B, dim/2)。
        """
        half = dim // 2
        # 对数空间等距的频率：exp(-log(max_period) * i/half)，i=0..half-1
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]  # (B,1)*(1,half) → (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:  # dim 为奇数时 cos/sin 拼出 dim-1 维，补一列 0 凑齐
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # 标量 t → sinusoidal 频率嵌入 → MLP → 条件向量 (B, hidden_size)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    """DiT 输出层 / final adaLN + linear projection back to latent space.

    最后一个 block 之后：条件 c 生成 (shift, scale) 两份调制参数（注意只有 2 份，没有
    gating），对无仿射 LayerNorm 后的特征做 adaLN 调制，再线性投影回声学 latent 的维度
    ``output_size``（即 velocity field 的输出）。

    形状 (Shapes):
        x : (B, T, hidden_size) → return (B, T, output_size)
        c : (B, hidden_size)    条件向量

    设计原因 (Why)：``initialize_weights`` 里把这里的 ``linear`` 权重/偏置都初始化为 0，
    使整个 DiT 在训练起点输出全 0 的 velocity，等价于恒等流，训练更稳。
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),  # 生成 shift+scale 两份
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, c, **_kwargs):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)  # 各 (B, hidden_size)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTBlock(nn.Module):
    """单个 DiT block / Transformer block with adaLN-Zero modulation + gating.

    结构是标准的 pre-norm Transformer block（self-attention + FFN，各带残差），但归一化
    用的是 **adaLN (adaptive LayerNorm)**：条件向量 c 经一个 MLP 产出 6 份参数 ——
    attention 与 FFN 各一组 (shift, scale, gate)：

        x = x + gate_attn * attn( modulate(norm1(x), shift_attn, scale_attn) )
        x = x + gate_ffn  * ffn(  modulate(norm2(x), shift_ffn,  scale_ffn ) )

    其中 ``gate`` 对残差分支做门控（adaLN-Zero：gate 初始为 0，block 起步即恒等映射）。
    这就是说话人 x-vector / timestep 等全局条件注入声学 latent 序列的方式。

    参数 (Args):
        attention  : 注入的 MultiHeadAttention 模块（带 RoPE、KV cache 等）。
        ffn        : 注入的 Mlp 模块。
        modulation : 是否启用 adaLN 调制。True ⇒ 用条件 c 调制、LayerNorm 不带仿射参数
                     （仿射改由 c 动态生成）；False ⇒ 普通带仿射的 LayerNorm、无条件。
                     ``forward`` 里用 assert 强制 modulation 与是否传入 condition 一致。
    """

    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        hidden_size: int = 1024,
        modulation: bool = False,
        eps: float = 1e-5,
        **_kwargs,
    ):
        super().__init__()
        # modulation=True 时关闭 LayerNorm 自带的 affine：scale/shift 改由条件 c 动态产生
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=not modulation, eps=eps
        )
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=not modulation, eps=eps
        )
        self.attn = attention
        self.ffn = ffn
        self.modulation = modulation
        if modulation:
            # 一次性产出 6 份调制参数：attn 的 shift/scale/gate + ffn 的 shift/scale/gate
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(self, x, condition=None, mask=None, **kwargs):
        """前向：pre-norm attention + FFN，可选 adaLN 条件调制。

        形状 (Shapes):
            x         : (B, T, hidden_size) 声学 latent 序列特征
            condition : (B, hidden_size) 全局条件 c（timestep / duration / 说话人之和）
            mask      : 注意力 mask，透传给 attention
            return    : (B, T, hidden_size)

        分两条路径：有 condition 走 adaLN 调制 + gating；无 condition 退化为普通
        Transformer block。两者必须与 ``self.modulation`` 一致（下方 assert 保证）。
        """
        if condition is None:
            assert not self.modulation, (
                "Without global condition, must set modulation to False"
            )
        else:
            assert self.modulation, "With global condition, must set modulation to True"
            # 一个线性层产 6 份，按通道切开；每份形状 (B, hidden_size)
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
                self.adaLN_modulation(condition).chunk(6, dim=1)
            )

        if condition is not None:
            # pack_indices：double-streaming / packed 推理下，不同 token 来自不同样本，
            # 用索引把每条样本的 gate 散射 (gather) 到打平后的 token 维上 → 形状贴合 x。
            pack_indices = kwargs.get("pack_indices")
            if pack_indices is not None:
                gate_attn = gate_attn[pack_indices]
                gate_ffn = gate_ffn[pack_indices]
            else:
                # 常规 batch：在时间维插一维，让 (B, D) gate 广播到 (B, T, D)
                gate_attn = gate_attn.unsqueeze(1)
                gate_ffn = gate_ffn.unsqueeze(1)

        if condition is not None:
            # 残差分支：先 adaLN 调制再 attention，最后用 gate 门控（adaLN-Zero）
            x = x + gate_attn * self.attn(
                modulate(self.norm1(x), shift_attn, scale_attn, **kwargs),
                mask=mask,
                **kwargs,
            )
        else:
            x = x + self.attn(self.norm1(x), mask=mask, **kwargs)

        if condition is not None:
            x = x + gate_ffn * self.ffn(
                modulate(self.norm2(x), shift_ffn, scale_ffn, **kwargs)
            )
        else:
            x = x + self.ffn(self.norm2(x), mask=mask)
        return x


class DiT(nn.Module):
    """flow-matching velocity field 的 DiT 主干 / Diffusion-Transformer acoustic head.

    职责 (Role)：给定含噪声学 latent ``x``、连续时间 ``timesteps``，以及（来自 Qwen2.5
    主干、说话人 x-vector 等的）全局条件 ``g_cond``，预测 velocity field
    v_θ(x_t, t, cond)。配 ODE solver 多步积分即可把噪声还原成干净声学 latent。

    结构：input_layer (Linear 升维) → N×DiTBlock (带 adaLN 调制) → FinalLayer (投影回
    out_dim)。条件 c = time_embedder(t) (+ duration_embedder(duration)) (+ g_cond)，
    在每个 block 内通过 adaLN 注入，而不是拼进 token 序列。

    参数 (Args):
        in_dim / out_dim   : 输入/输出 latent 维度（通常相等，velocity 与 latent 同维）。
        transformer_config : 提供 hidden_size / num_layers / num_heads 等，并能 to_dict()
                             把超参透传给每层的 attention / FFN。
        mode               : ``"flow_matching"`` 或 ``"meanflow"``。后者额外建一个
                             ``duration_embedder`` 把积分区间长度编码进条件，支持少步采样。
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        transformer_config,
        *,
        mode: str = "flow_matching",
    ):
        super().__init__()
        if mode not in {"flow_matching", "meanflow"}:
            raise ValueError(
                f"DiT mode must be 'flow_matching' or 'meanflow', got {mode!r}."
            )

        transformer_kwargs = transformer_config.to_dict()  # 透传给每层 attn/ffn 的超参
        model_dim = transformer_config.hidden_size
        self.mode = mode
        self.num_layers = transformer_config.num_layers

        self.input_layer = nn.Linear(in_dim, model_dim)
        self.time_embedder = TimestepEmbedder(model_dim)
        if mode == "meanflow":
            # 仅 MeanFlow 蒸馏模式：额外编码「积分区间长度」duration 进条件
            self.duration_embedder = TimestepEmbedder(model_dim)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            attn_block = MultiHeadAttention(**transformer_kwargs, name=f"layer_{i}")
            ffn_block = Mlp(
                act_layer=lambda: nn.GELU(approximate="tanh"), **transformer_kwargs
            )
            self.blocks.append(
                DiTBlock(attention=attn_block, ffn=ffn_block, **transformer_kwargs)
            )

        self.output_layer = FinalLayer(model_dim, out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """权重初始化 / adaLN-Zero init.

        Linear 用 Xavier、bias 置 0；关键是把所有 adaLN_modulation 的最后一层
        （以及 FinalLayer 的输出 linear）权重/偏置初始化为 **0**：这样起步时
        scale=shift=gate=0、velocity 输出为 0，每个 DiTBlock 近似恒等映射、整网输出全 0，
        是 DiT 论文证明能显著稳定训练的 adaLN-Zero 技巧。
        """

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.output_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output_layer.linear.weight, 0)
        nn.init.constant_(self.output_layer.linear.bias, 0)

    def forward(
        self,
        x,
        timesteps,
        duration: torch.Tensor | None = None,
        mask=None,
        attn_mask=None,
        g_cond: torch.Tensor | None = None,
        **kwargs,
    ):
        """预测 velocity field / forward pass.

        参数 (Args):
            x         : (B, T, in_dim) 含噪声学 latent x_t
            timesteps : (B,) 连续时间 t
            duration  : (B,) 仅 meanflow 用，积分区间长度；flow_matching 传 None
            mask      : 序列 padding mask（透传，当前实现未直接消费）
            attn_mask : 注意力 mask，作为 block 的 ``mask`` 传给 attention
            g_cond    : (B, model_dim) 全局条件（Qwen2.5 主干上下文 / 说话人 x-vector 等）
            return    : (B, T, out_dim) 预测的 velocity v_θ(x_t, t, cond)

        条件相加而非拼接：time / duration / g_cond 投到同一 model_dim 后逐元素求和得到
        统一的全局条件 c，再由每个 DiTBlock 内部用 adaLN 注入。
        """
        t = self.time_embedder(timesteps)
        c = t
        # getattr 取 duration_embedder：flow_matching 模式没建这个属性，故用默认 None 容错
        duration_embedder = getattr(self, "duration_embedder", None)
        if duration_embedder is not None and duration is not None:
            c = c + duration_embedder(duration)  # MeanFlow：叠加区间长度条件
        if g_cond is not None:
            c = c + g_cond  # 叠加主干/说话人全局条件

        x = self.input_layer(x)  # (B,T,in_dim) → (B,T,model_dim)
        for block in self.blocks:
            x = block(x, c, mask=attn_mask, **kwargs)
        return self.output_layer(x, c, **kwargs)
