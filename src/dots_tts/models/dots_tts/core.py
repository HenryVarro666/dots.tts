"""
dots.tts 神经核心 (neural core)  ——  连续潜在 (continuous-latent) 自回归 TTS 的中枢。

本文件做什么
------------
定义 `DotsTtsCore`：把三大子网络缝合成一个端到端的 TTS 模型，并实现训练 forward 与
逐步 (step-by-step) 推理所需的所有"装配"逻辑：
  1. LLM 主干 (Qwen2ForCausalLM / Qwen2.5-1.5B)：自回归地预测下一 token；其中 audio span
     token (`<|audio_gen_span|>` / `<|audio_comp_span|>`) 在喂进 LLM 前会被替换成由
     `VAESemanticEncoder` 算出的 patch embedding —— 这就是"用 latent 代替离散音频 token"。
  2. flow-matching DiT (`velocity_field_predictor`)：声学头 (acoustic head)，在连续 latent
     空间里预测 velocity field v_t，配合 ODE solver 把高斯噪声积分成目标 latent。
  3. CAM++ x-vector：经 `xvec_proj` 投影后作为说话人/音色的全局条件 (global condition)
     g_cond 注入 DiT (adaLN)。

在数据流里的位置
----------------
  text/音色 → DotsTtsCore.forward(训练) 或 step_llm+step_fm(推理) → 连续 latent
  → (本文件之外) BigVGAN AudioVAE 解码 → 24/48kHz 波形。
本文件只负责 token↔latent 的装配、mask 构造、flow-matching 目标/积分；不含波形编解码。

关键类/函数清单
----------------
  - DotsTtsCore.forward            训练前向：自回归 LLM + span→patch 替换 + flow-matching 目标
  - DotsTtsCore.step_llm           推理：单步 LLM 前向，带 KV cache
  - DotsTtsCore.fm_solver_step     单个 ODE 步；CFG 把 batch 翻倍，vt = vt_c + scale*(vt_c - vt_u)
  - DotsTtsCore._flow_matching_step_fm  标准 flow-matching：用 torchdiffeq.odeint 积分采样一个 patch
  - DotsTtsCore._meanflow_step_fm  MeanFlow 蒸馏：少步 (NFE 通常=2) 自定义积分
  - DotsTtsCore.step_fm            按 self.mode 路由到上面两者之一
  - FlowMatchingHelper             flow-matching 的 x_t / u_t / σ_t 计算 (Lipman 2022)
  - CausalHelper                   因果 mask 与块对角 (block-diagonal) chunk mask + 按 patch 的 position id
  - IOHelper                       latent 归一化、VAE 重参数采样、prepare_inputs_for_dit 的 span/latent 交错拼装

核心 ML 概念：flow-matching、velocity field、classifier-free guidance (CFG)、DiT + adaLN、
KV cache、x-vector、continuous latent、patch、MeanFlow 蒸馏。
"""

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from transformers import Qwen2Config, Qwen2ForCausalLM

from dots_tts.models.dots_tts.config import ModelConfig
from dots_tts.modules.backbone.dit import DiT
from dots_tts.modules.backbone.semantic_encoder import VAESemanticEncoder
from dots_tts.utils.tokenizer import (
    AUDIO_COMP_SPAN_TOKEN,
    AUDIO_GEN_SPAN_TOKEN,
    TEXT_COND_END_TOKEN,
    require_token_id,
)
from dots_tts.utils.util import get_mask_from_lengths, mask_data


@dataclass(frozen=True)
class DotsTtsForwardOutput:
    """训练 forward 的返回打包。各字段供 loss 函数分别取用。

    Attributes:
        llm_logits: (B, L, V) 自回归 LLM 在词表上的 logits，用于文本/控制 token 的交叉熵。
        pred:       (N, latent_patch_size, latent_dim) DiT 预测的 velocity field v_t；
                    N = 全 batch 内 latent patch 总数 (拉平后拼接)。
        target:     与 pred 同形，flow-matching 的真值 u_t = x1 - (1-σ)·x0。
        eos_out:    (B, L, 2) 每个时间步的 EOS/继续二分类 logits，决定何时停止自回归生成。
    """

    llm_logits: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    eos_out: torch.Tensor


class DotsTtsCore(nn.Module):
    """dots.tts 的神经核心：自回归 LLM 主干 + flow-matching DiT 声学头 + x-vector 条件。

    设计要点 (为什么这么搭)：
      - "无离散音频 token"：音频不被量化成码本 (codebook) id，而是停留在连续 latent 空间。
        LLM 序列里用占位 span token 标记音频位置，真正喂进 LLM 的是这些 token 处被替换成的
        patch embedding (由 VAESemanticEncoder 把 latent 编码而来)。
      - LLM 负责"语义/时序规划"(决定每个 patch 的上下文 hidden)，DiT 负责"声学细节生成"
        (在该 hidden 条件下用 flow-matching 把噪声积分成 latent)。两阶段解耦。
      - 支持 classifier-free guidance (CFG)：训练时随机丢弃 LLM hidden 与 x-vector 条件
        (cfg_droprate / xvec_drop_rate)，推理时再用 guidance_scale 外推。
      - 可选 MeanFlow 蒸馏：把多步 ODE 压成极少步 (NFE≈2) 推理，靠 duration embedding。

    子模块速览：
      llm                      Qwen2ForCausalLM，词表扩成 len(tokenizer) 以容纳特殊 token。
      patch_encoder            latent → LLM hidden 维度的 patch embedding (VAESemanticEncoder)。
      hidden_proj/latent_proj/coordinate_proj  分别把 LLM hidden / 历史 latent / 噪声 latent
                               投到 DiT 的 fm_hidden_size，是 DiT 输入序列的三种"槽位"来源。
      xvec_proj                CAM++ x-vector → DiT 全局条件 g_cond。
      velocity_field_predictor flow-matching / meanflow 的 DiT，预测 velocity field。
      eos_proj                 LLM hidden → 2 类 EOS logits。
      fm_helper/causal_helper/io_helper  见各自类的 docstring。
    """

    # region Module construction
    def __init__(
        self,
        config: ModelConfig,
        llm_config: Qwen2Config,
        tokenizer=None,
        *,
        latent_stats_path,
    ):
        """构建并连线所有子模块。

        Args:
            config: dots.tts 的 ModelConfig (含 DiT/PatchEncoder 子配置、patch_size、
                latent_dim、CFG/xvec drop rate、可选 meanflow 配置等)。
            llm_config: Qwen2 主干配置；会被深拷贝后把 vocab_size 改成 len(tokenizer)。
            tokenizer: 提供 audio span / text_cond_end 等特殊 token 的 id，必须传入。
            latent_stats_path: 预存的 latent 全局均值/方差，用于 IOHelper 的归一化。
        """
        super().__init__()
        self.config = config
        self.fm_hidden_size = config.DiT.hidden_size  # DiT 内部工作维度
        # 每个 LLM hidden 对应 1 个 DiT "槽位"；下游多处 assert 依赖此值恒为 1。
        self.hidden_patch_size = 1
        self.cfg_droprate = config.get("cfg_droprate", 0.2)  # 训练时丢弃 LLM hidden 条件的概率 (CFG)
        self.latent_patch_size = config.patch_size  # 一个 audio patch 含多少帧 latent
        self.latent_dim = config.latent_dim
        self.xvec_dim = config.campplus_embedding_size  # CAM++ x-vector 维度
        self.xvec_drop_rate = config.get("xvec_drop_rate", 0.2)  # 训练时丢弃音色条件的概率 (CFG)

        # Setup tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be provided before building the model.")
        if llm_config is None:
            raise RuntimeError("LLM config must be provided before building the model.")
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        # 两类 audio span 占位 token：gen=待生成音频, comp=已知/参考音频 (clone 条件)。
        # 它们在序列中的 hidden/embedding 会被换成 latent patch embedding。
        self.audio_gen_span_id = require_token_id(self.tokenizer, AUDIO_GEN_SPAN_TOKEN)
        self.audio_comp_span_id = require_token_id(
            self.tokenizer, AUDIO_COMP_SPAN_TOKEN
        )
        self.text_cond_end_id = require_token_id(self.tokenizer, TEXT_COND_END_TOKEN)

        # Setup LLM with language modeling head so we can obtain logits directly
        # 深拷贝避免污染外部 config；词表扩到 len(tokenizer) 以纳入新增的特殊 token。
        llm_config = copy.deepcopy(llm_config)
        llm_config.vocab_size = len(self.tokenizer)
        self.llm = Qwen2ForCausalLM._from_config(
            llm_config,
            dtype=torch.float32,
        )
        self.llm_hidden_size = self.llm.config.hidden_size

        # latent → LLM hidden 维的 patch embedding 编码器：把连续 latent 压成
        # 每 patch 一个 embedding，去顶替序列里的 audio span token embedding。
        self.patch_encoder = VAESemanticEncoder(
            in_dim=self.latent_dim,
            out_dim=self.llm_hidden_size,
            config=config,
        )

        # Setup Flow matching related modules
        # DiT 输入序列由三类"槽位"拼成，各用一条线性层投到 fm_hidden_size：
        #   hidden_proj    : LLM hidden (语义条件)
        self.hidden_proj = nn.Linear(self.llm_hidden_size, self.fm_hidden_size)
        #   latent_proj    : 历史/参考 latent (history latent，已知声学)
        self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        #   coordinate_proj: 噪声 latent x_t (待去噪的"坐标")
        self.coordinate_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.xvec_proj = nn.Sequential(
            nn.Linear(self.xvec_dim, self.fm_hidden_size),
            nn.LayerNorm(self.fm_hidden_size),
        )
        self.meanflow_config = config.meanflow if config.meanflow is not None else None
        # 整体推理模式：meanflow(少步蒸馏) 或 flow_matching(标准多步 ODE)。
        self.mode = (
            "meanflow"
            if self.meanflow_config is not None and self.meanflow_config.enabled
            else "flow_matching"
        )
        # DiT 结构模式略有不同：仅当 meanflow 且开启 duration embedding 时，DiT 才多一个
        # duration_embedder (吃步长 dt)。否则即便整体 meanflow，DiT 仍按 flow_matching 建。
        dit_mode = (
            "meanflow"
            if self.mode == "meanflow"
            and self.meanflow_config.use_duration_embedding
            else "flow_matching"
        )
        self.velocity_field_predictor = DiT(
            in_dim=self.fm_hidden_size,
            out_dim=self.latent_dim,
            transformer_config=config.DiT,
            mode=dit_mode,
        )

        # Setup eos predictor
        # 由 LLM hidden 预测 2 类 (停止/继续)，推理时据此决定是否结束自回归生成。
        self.eos_proj = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
            nn.SiLU(),
            nn.Linear(self.llm_hidden_size, 2),
        )

        # Helpers
        self.fm_helper = FlowMatchingHelper(sigma=config.get("fm_sigma", 0.0))
        self.causal_helper = CausalHelper()
        self.io_helper = IOHelper(latent_stats_path=latent_stats_path)
        # 这两个 span id 在 forward 里用来定位需要被 patch embedding 顶替的位置。
        self.audio_span_token_ids: list[int] = [
            self.audio_gen_span_id,
            self.audio_comp_span_id,
        ]
    # endregion Module construction

    # region Training forward path
    def forward(self, data: dict[str, Any]) -> DotsTtsForwardOutput:
        """训练前向：一次跑完 自回归 LLM + flow-matching 目标，返回三套监督信号。

        整体流程：
          1. 取 latent，VAE 重参数采样 → 编成 patch embedding；把序列里 audio span token
             的 embedding 就地替换为对应 patch embedding (latent 代替离散音频 token)。
          2. LLM 前向 (带 padding mask 的因果注意力) 得到 logits 与最后一层 hidden。
          3. EOS 头：用 detach 后的 hidden 预测停止/继续 (放在 CFG mask 之前，保证标签干净)。
          4. flow-matching：对 x-vector 做 CFG dropout 得 g_cond；用 io_helper 把
             LLM hidden / 历史 latent / 噪声 latent 交错拼成 DiT 输入序列，DiT 预测 velocity
             field v_t，与真值 u_t 对齐 (loss 在外部算)。
          5. 若本 batch 无任何 latent，则走 dummy 分支：仍调用 DiT 一次但乘 0，纯粹为
             DDP 保证所有参数都有梯度连接，避免 "unused parameters" 报错。

        Args:
            data: 训练 batch，关键键：
                input_ids (B, L), input_ids_lengths (B,),
                input_span_mask (B, L)  — 输入侧需被 patch embedding 顶替的位置,
                output_span_mask (B, L) — 需做 flow-matching 监督的 (待生成) 位置,
                latents 或 latents_sampled, latent_lengths, xvector, 可选 vocal_mask。

        Returns:
            DotsTtsForwardOutput(llm_logits, pred, target, eos_out)。
        """
        input_ids: torch.Tensor = data["input_ids"]
        input_ids_lengths: torch.Tensor = data["input_ids_lengths"]
        input_span_mask: torch.Tensor = data["input_span_mask"]
        output_span_mask: torch.Tensor = data["output_span_mask"]
        batch_size = input_ids.size(0)
        device = input_ids.device

        # latents      : VAE 输出的 (mean, log_std) 拼接，未采样的"分布参数"。
        # latents_sampled: 已重参数采样好的 latent；二者至少给一个。
        latents: torch.Tensor | None = data.get("latents")
        latents_sampled: torch.Tensor | None = data.get("latents_sampled")
        latent_lengths: torch.Tensor | None = data.get("latent_lengths")
        has_latents = latents is not None or latents_sampled is not None

        patch_embeddings: torch.Tensor | None
        valid_patch_counts: torch.Tensor | None
        if has_latents:
            if latents_sampled is None:
                # 重参数采样 (reparameterization): z = mean + randn*exp(log_std)。
                latents_sampled = self.io_helper.sample_from_latent(latents)
            # patch_embeddings: (B, num_patch, llm_hidden) —— 顶替 span token 用。
            patch_embeddings = self.patch_encoder(
                latents_sampled, x_lens=latent_lengths
            )
            # 每条样本的有效 patch 数 = latent 帧数 // 每 patch 帧数。
            valid_patch_counts = latent_lengths // self.latent_patch_size
            # 注意先编 patch_embeddings 再归一化：patch_encoder 吃原始尺度，
            # 而 flow-matching 目标用归一化后的 latent (零均值单位方差，利于训练稳定)。
            latents_sampled = self.io_helper.normalize(latents_sampled)
        else:
            latents_sampled = None
            patch_embeddings = None
            valid_patch_counts = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )

        # 每条样本含多少个 input 侧 audio span token (待被 patch embedding 顶替)。
        input_span_counts = input_span_mask.sum(dim=1)
        if input_span_counts.sum() > 0 and patch_embeddings is None:
            raise RuntimeError(
                "Found audio span tokens but no latents provided to compute patch embeddings."
            )

        # Token embeddings with audio span replacement
        # 先按词表查 embedding，再把 span 位置就地换成 latent 的 patch embedding。
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        if patch_embeddings is not None:
            inputs_embeds = inputs_embeds.clone()  # clone 以免原 embedding 表被原地写
            patch_embeddings = patch_embeddings.to(inputs_embeds.dtype)
            for b in range(batch_size):
                span_num = input_span_counts[b].item()
                if span_num == 0:
                    continue
                expected = valid_patch_counts[b].item()
                # span token 数必须与 latent patch 数严格一致，否则交错装配会错位。
                if expected != span_num:
                    raise RuntimeError(
                        f"Mismatch between span tokens ({span_num}) and latent patches ({expected}) for sample {b}."
                    )
                # indices: 该样本所有 span token 的列位置 (1D)。
                indices = input_span_mask[b].nonzero(as_tuple=False).squeeze(-1)
                # 按顺序把第 i 个 span 的 embedding 换成第 i 个 patch embedding。
                inputs_embeds[b, indices, :] = patch_embeddings[b, :span_num, :]

        # LLM forward pass to obtain logits & hidden states
        # 这里只取 padding 的 seq_mask 喂给 HF (HF 内部自带因果 mask)；因果 attn_mask 不用。
        _llm_attn_mask, llm_seq_mask, _ = self.causal_helper.create_causal_mask_and_pos(
            seq_lens=input_ids_lengths, max_len=input_ids.size(1)
        )
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=llm_seq_mask.long(),
            use_cache=False,  # 训练整段并行前向，无需 KV cache
            output_hidden_states=True,
            return_dict=True,
        )
        llm_logits = llm_outputs.logits  # [B, L, V]
        llm_hidden = llm_outputs.hidden_states[-1]  # [B, L, H]

        # eos prediction, before cfg masking
        # detach：EOS 头的梯度不回传 LLM (它只是一个旁路分类器)；且必须在 CFG dropout
        # 之前算，否则被丢弃的 hidden 会污染 EOS 标签语义。
        eos = self.eos_proj(llm_hidden.detach())

        # Flow matching forward
        total_patches = int(output_span_mask.sum().item())  # 全 batch 待生成 patch 总数
        if total_patches > 0 and latents_sampled is None:
            raise RuntimeError("Flow matching requested but latents are missing.")
        if total_patches > 0:
            xvec_cond = self.xvec_proj(data["xvector"])  # x-vector → DiT 全局条件 g_cond
            # vocal_mask 标记"含人声/可用音色"的样本；无则全 True。
            vocal_mask = data.get("vocal_mask")
            if vocal_mask is None:
                vocal_mask = torch.ones((batch_size,), device=device, dtype=torch.bool)
            # CFG: 以 xvec_drop_rate 概率随机丢弃音色条件，让模型学会"无条件"分支。
            xvec_drop_mask = (
                torch.empty((batch_size,), device=device, dtype=torch.float32).uniform_(
                    0, 1
                )
                < self.xvec_drop_rate
            )
            # 只对真有 vocal 的样本做丢弃 (无人声样本本就没有有意义的音色，不参与 CFG)。
            xvec_drop_mask = xvec_drop_mask & vocal_mask
            xvec_cond = mask_data(xvec_cond, xvec_drop_mask)  # 被选中的位置置 0

            # DiT 条件序列里 span(待生成)位置用 LLM hidden，其余位置用原 token embedding：
            # 即"语义条件来自 LLM 顶层 hidden，非语义位置保留输入嵌入"。
            hiddens_for_fm = torch.where(
                output_span_mask.unsqueeze(-1), llm_hidden, inputs_embeds
            )

            # Prepare DiT inputs
            # 把 LLM hidden / 历史 latent / 噪声 latent 交错拼成 DiT 的输入序列 fm_seq，
            # 并同步产出 flow-matching 真值 target=u_t、采样时刻 times、块对角 attn_mask、
            # 以及切回 per-patch 预测所需的长度元信息。详见 IOHelper.prepare_inputs_for_dit。
            (
                fm_seq,
                target,
                fm_attn_mask,
                fm_seq_mask,
                fm_pos_ids,
                times,
                fm_prefix_lengths,
                fm_gen_lengths,
                fm_gen_patch_size,
            ) = self.io_helper.prepare_inputs_for_dit(
                hiddens=hiddens_for_fm,
                hidden_lens=input_ids_lengths,
                latents=latents_sampled,
                latent_lens=latent_lengths,
                hidden_proj=self.hidden_proj,
                latent_proj=self.latent_proj,
                noisy_proj=self.coordinate_proj,
                span_mask=output_span_mask,
                hidden_patch_size=self.hidden_patch_size,
                latent_patch_size=self.latent_patch_size,
                fm_helper=self.fm_helper,
                cfg_droprate=self.cfg_droprate,
            )

            # Predict velocity field
            # DiT 对整条交错序列预测 velocity field；g_cond=音色, times=每样本采样时刻。
            vt = self.velocity_field_predictor(
                x=fm_seq,
                timesteps=times,
                pos_ids=fm_pos_ids,
                mask=fm_seq_mask,
                attn_mask=fm_attn_mask,
                return_hidden_stats=False,
                g_cond=xvec_cond,
            )

            # Get predictions and targets
            # 从交错序列里抠出"纯 latent"部分并拉平成 (N, latent_patch_size, latent_dim)，
            # 与同样拉平的 target 一一对应，供外部算 MSE flow-matching loss。
            pred = self.io_helper.get_dit_outputs(
                pred_v=vt,
                fm_prefix_lengths=fm_prefix_lengths,
                fm_gen_lengths=fm_gen_lengths,
                fm_gen_patch_size=fm_gen_patch_size,
                latent_patch_size=self.latent_patch_size,
            )
        else:
            # Dummy forward for velocity_field_predictor to keep gradients connected in DDP
            # 本 batch 没有任何待生成 latent，但 DDP 要求每个 rank 用到所有参数，否则报
            # "unused parameters"。故构造零输入跑一遍 DiT 与三条投影层，乘 0 不影响 loss，
            # 只为让这些参数出现在反向图里。
            dummy_length = self.latent_patch_size
            dummy_seq_h = llm_hidden.new_zeros((1, dummy_length, self.llm_hidden_size))
            dummy_seq_h = self.hidden_proj(dummy_seq_h) * 0.0  # dummy op for ddp
            dummy_seq_l = llm_hidden.new_zeros((1, dummy_length, self.latent_dim))
            dummy_seq_l = self.latent_proj(dummy_seq_l) * 0.0  # dummy op for ddp
            dummy_seq_c = llm_hidden.new_zeros((1, dummy_length, self.latent_dim))
            dummy_seq_c = self.coordinate_proj(dummy_seq_c) * 0.0  # dummy op for ddp
            dummy_seq = dummy_seq_h + dummy_seq_l + dummy_seq_c
            dummy_times = torch.zeros((1,), device=device, dtype=torch.float32)
            dummy_attn_mask = torch.ones(
                (1, dummy_length, dummy_length), device=device, dtype=torch.bool
            )
            dummy_out = self.velocity_field_predictor(
                x=dummy_seq,
                timesteps=dummy_times,
                attn_mask=dummy_attn_mask,
            )
            pred = dummy_out[:, -self.latent_patch_size :, :]
            target = pred.detach()  # pred==target ⇒ flow-matching loss 恒为 0

        return DotsTtsForwardOutput(
            llm_logits=llm_logits,
            pred=pred,
            target=target,
            eos_out=eos,
        )
    # endregion Training forward path

    # region Autoregressive and flow-matching inference steps
    @torch.no_grad()
    def fm_solver_step(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None,
        guidance_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        """ODE solver 的单步右端项 (RHS)，带 classifier-free guidance (CFG)。

        被 torchdiffeq.odeint 在每个积分子步反复调用，返回当前 latent 坐标 z 处的
        引导后 velocity field，用于把噪声朝目标 latent 推进一步。

        CFG 实现：把 batch 在第 0 维翻倍 —— 上半"有条件"(用真实 LLM hidden + 音色),
        下半"无条件"(用 cfg_sequence 的占位 hidden + g_cond 置零)。DiT 一次前向同时算出
        两支 v_t，再外推：vt = vt_c + scale·(vt_c − vt_u)。scale 越大越贴合条件。

        Args:
            t: 标量时间步 (会被 reshape+repeat 成两支)。
            z: (B, patch_size, latent_dim) 当前 latent 坐标。
            input_sequence / cfg_sequence: (B, S, fm_hidden) 已拼好的条件 / 无条件序列，
                末尾 patch_size 个槽位是留给当前噪声 latent 的占位 (会被 z 覆盖)。
            hidden_size / patch_size: 传给 DiT 用于内部分块的尺寸提示。
            g_cond: (B, fm_hidden) 音色条件；无条件支用其 zeros_like。
            guidance_scale: CFG 强度，float 或张量均可。

        Returns:
            (B, patch_size, latent_dim) 引导后的 velocity field。
        """
        batch_size = input_sequence.size(0)
        if input_sequence.shape != cfg_sequence.shape:
            raise ValueError(
                "FM input_sequence and cfg_sequence must share the same shape."
            )
        if input_sequence.size(1) < patch_size:
            raise ValueError(
                "FM input sequence must reserve at least one latent patch slot."
            )
        # 当前噪声 latent 占据序列最后 patch_size 个槽位。
        latent_start = input_sequence.size(1) - patch_size
        z = self.coordinate_proj(z)  # latent → fm_hidden，写回序列末尾占位
        # ——有条件支——
        z_c = input_sequence.clone()
        z_c[:, latent_start:] = z
        z_branches = [z_c]
        g_cond_t = (
            None if g_cond is None else g_cond.to(device=z_c.device, dtype=z_c.dtype)
        )
        g_cond_branches = None if g_cond_t is None else [g_cond_t]

        # ——无条件支—— 用 cfg_sequence(占位/丢弃了 LLM hidden 的序列)，并把音色条件清零。
        z_cfg = cfg_sequence.clone()
        z_cfg[:, latent_start:] = z
        z_branches.append(z_cfg)
        if g_cond_branches is not None:
            g_cond_branches.append(torch.zeros_like(g_cond_t))

        # batch 翻倍：[有条件; 无条件] 一次前向，省一半 DiT 调用。
        z_z = torch.cat(z_branches, dim=0)
        t_t = t.reshape(1).repeat(len(z_branches))  # 同一时间步广播到两支
        if g_cond_branches is not None:
            g_cond_t = torch.cat(g_cond_branches, dim=0)
        vt = self.velocity_field_predictor(
            x=z_z,
            timesteps=t_t,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond_t,
            # 推理 KV/分块提示：1 个 hidden + patch_size 历史 + patch_size 噪声 = 2*patch+hidden。
            hidden_size=patch_size * 2 + hidden_size,
            patch_size=patch_size + 1,
        )
        vt = vt[:, latent_start:]  # 只取末尾噪声 latent 对应的 velocity
        vt_c = vt[:batch_size]  # 前半 = 有条件
        vt_u = vt[batch_size:]  # 后半 = 无条件
        if not torch.is_tensor(guidance_scale):
            guidance_scale = vt_c.new_tensor(float(guidance_scale))
        else:
            guidance_scale = guidance_scale.to(device=vt_c.device, dtype=vt_c.dtype)
        # CFG 外推：在无条件方向反向、有条件方向加强。
        return vt_c + guidance_scale * (vt_c - vt_u)

    @torch.no_grad()
    def step_llm(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any | None]:
        """自回归推理的单步 LLM 前向，带 KV cache。

        既可喂 input_ids(普通 token)，也可直接喂 inputs_embeds(例如 audio span 位置已被
        patch embedding 顶替的那一步)，二者必须且只能给一个。传入上一步的 past_key_values
        以复用 KV cache、只对新增 token 算注意力，返回更新后的 cache 供下一步续传。

        Args:
            inputs_embeds: (B, T, H) 预算好的输入嵌入，或
            input_ids:     (B, T)    token id (内部转成嵌入)。
            past_key_values: 上一步的 KV cache，首步为 None。

        Returns:
            (inputs_embeds, hidden(B,T,H), logits(B,T,V), past_key_values)。
        """
        # 异或校验：恰好提供其一。
        provided = int(inputs_embeds is not None) + int(input_ids is not None)
        if provided != 1:
            raise ValueError(
                "Exactly one of inputs_embeds or input_ids must be provided to step_llm()."
            )

        if inputs_embeds is not None:
            pass  # 已是嵌入，直接用
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,  # 推理开 KV cache，增量解码
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]
        logits = outputs.logits
        past_key_values = outputs.past_key_values  # 透传给调用方续传

        return inputs_embeds, hidden, logits, past_key_values

    @torch.no_grad()
    def _meanflow_step_fm(
        self,
        *,
        input_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        nfe: int = 2,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """MeanFlow 蒸馏推理：用极少步 (NFE 默认 2) 把噪声积分成一个 latent patch。

        与标准 flow-matching 不同，MeanFlow 的 DiT 预测的是区间 [t, t+dt] 上的"平均速度"
        (mean velocity)，故每步直接 z ← z + v·dt 即可，无需 torchdiffeq 的自适应积分。
        把 [0,1] 均匀切成 nfe 段，逐段调用 meanflow_solver_step。注意此分支无 CFG batch 翻倍。

        Args:
            input_sequence: (B, S, fm_hidden) 已拼好的条件序列，末尾留 patch_size 个噪声槽位。
            patch_size: 本次生成的 latent patch 长度。
            nfe: number of function evaluations，即积分步数 (蒸馏后通常很小)。

        Returns:
            (B, patch_size, latent_dim) 去噪得到的 latent。
        """
        if nfe <= 0:
            raise ValueError(f"MeanFlow nfe must be positive, got {nfe}.")
        batch_size = input_sequence.size(0)
        device = input_sequence.device
        dtype = input_sequence.dtype
        solver_step = self.meanflow_solver_step if solver_step is None else solver_step
        # 初始坐标 = 标准高斯噪声 (t=0 处)。
        z = (
            torch.randn(
                (batch_size, patch_size, self.latent_dim),
                device=device,
                dtype=dtype,
            )
        )
        # nfe+1 个等距节点 → nfe 个步长。
        times = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=dtype)

        for step in range(nfe):
            t = times[step].expand(batch_size)  # 区间左端
            dt = (times[step + 1] - times[step]).expand(batch_size)  # 该步步长
            z = solver_step(
                z,
                t=t,
                dt=dt,
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                patch_size=patch_size,
                g_cond=g_cond,
            ).clone()  # clone 切断步间别名，避免原地写回引发的视图问题
        return z

    @torch.no_grad()
    def meanflow_solver_step(
        self,
        z: torch.Tensor,
        *,
        t: torch.Tensor,
        dt: torch.Tensor,
        input_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        patch_size: int,
        g_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        """MeanFlow 单步显式更新：z ← z + v(z,t,dt)·dt。

        DiT 这里多吃一个 duration=dt (经 duration_embedder 注入 adaLN 条件)，预测的是区间
        平均速度，故无需 CFG batch 翻倍，直接前向欧拉式推进一步。

        Returns:
            (B, patch_size, latent_dim) 推进 dt 后的 latent。
        """
        if input_sequence.size(1) < patch_size:
            raise ValueError(
                "MeanFlow input sequence must reserve at least one latent patch slot."
            )
        latent_start = input_sequence.size(1) - patch_size
        z_proj = self.coordinate_proj(z)  # latent → fm_hidden
        z_c = input_sequence.clone()
        z_c[:, latent_start:] = z_proj  # 写回末尾噪声槽位
        vt = self.velocity_field_predictor(
            x=z_c,
            timesteps=t,
            duration=dt,  # meanflow 专有：把步长作为条件
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond,
        )
        velocity = vt[:, latent_start:]  # 只取噪声 latent 段的速度
        # dt.view(-1,1,1)：把 (B,) 步长广播到 (B, patch, dim) 做逐样本缩放。
        return z + velocity * dt.view(-1, 1, 1)

    @torch.no_grad()
    def _flow_matching_step_fm(
        self,
        *,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 3.0,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """标准 flow-matching 推理：torchdiffeq.odeint 在 t∈[0,1] 上积分采样一个 latent patch。

        从高斯噪声 (t=0) 出发，沿 fm_solver_step 给出的 (带 CFG 的) velocity field 积分到
        t=1，得到目标 latent。固定步长方法 (euler/midpoint/rk4) 用 num_steps 控制 NFE；
        自适应方法则 NFE 不可控 (会告警)。

        Args:
            input_sequence / cfg_sequence: 条件 / 无条件序列 (CFG 用)。
            num_steps: 固定步长法的步数 → step_size = 1/num_steps。
            guidance_scale: CFG 强度。
            ode_method: torchdiffeq 求解器名。

        Returns:
            (B, patch_size, latent_dim) 积分终点 (t=1) 的 latent。
        """
        batch_size = input_sequence.size(0)
        num_evals = 0  # 实际 RHS 调用次数 (调试用，验证 NFE)
        solver_step = self.fm_solver_step if solver_step is None else solver_step
        guidance_scale_tensor = input_sequence.new_tensor(float(guidance_scale))

        # Prepare ODE solver
        # odeint 要求 RHS 签名为 f(t, z)；这里闭包补齐其余条件参数。
        def solver(t, z):
            nonlocal num_evals
            num_evals += 1
            return solver_step(
                t,
                z,
                input_sequence=input_sequence,
                cfg_sequence=cfg_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                hidden_size=hidden_size,
                patch_size=patch_size,
                g_cond=g_cond,
                guidance_scale=guidance_scale_tensor,
            )

        # Prepare noise as initial coordinate
        # 初值 y0 = 高斯噪声 (flow-matching 约定 t=0 为噪声端、t=1 为数据端)。
        noise = torch.randn(
            (batch_size, patch_size, self.latent_dim),
            dtype=input_sequence.dtype,
            device=input_sequence.device,
        )
        # Solve
        # 只需首尾两个评估时刻；轨迹中间点由 solver 内部按 step_size 细分。
        times = torch.tensor(
            [0.0, 1.0], dtype=input_sequence.dtype, device=input_sequence.device
        )
        if ode_method in ["euler", "midpoint", "rk4"]:  # fixed step size methods
            options = {"step_size": 1.0 / num_steps}  # 固定步长 ⇒ NFE 可控
        else:
            logger.warning(
                "Using adaptive step size ODE solver for FM, NFE is not guaranteed: "
                "ode_method={}",
                ode_method,
            )
            options = {}
        trajectory = odeint(
            func=solver,
            y0=noise,
            t=times,
            atol=1e-5,
            rtol=1e-5,
            method=ode_method,
            options=options,
        )
        # print(f"Expected NFE: {num_steps}, Actual NFE: {num_evals}")
        return trajectory[-1]  # t=1 处的 latent

    @torch.no_grad()
    def step_fm(
        self,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 3.0,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """flow-matching 采样的统一入口：按 self.mode 路由到 meanflow 或标准 flow-matching。

        外部推理循环 (每生成一个 audio patch) 调用本方法一次，得到该 patch 的 latent。
        meanflow 分支不需要 cfg_sequence/guidance_scale (那两个仅标准 CFG 分支使用)。
        """
        if self.mode == "meanflow":
            return self._meanflow_step_fm(
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                patch_size=patch_size,
                g_cond=g_cond,
                nfe=num_steps,
                solver_step=solver_step,
            )

        return self._flow_matching_step_fm(
            input_sequence=input_sequence,
            cfg_sequence=cfg_sequence,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            hidden_size=hidden_size,
            patch_size=patch_size,
            g_cond=g_cond,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            solver_step=solver_step,
        )
    # endregion Autoregressive and flow-matching inference steps


class FlowMatchingHelper:
    """
    Base helper for computing x_t and u_t, given target x_1 and noise x_0
    ref:  Flow matching for generative modeling, Lipman

    给定噪声端 x0 (t=0) 与数据端 x1 (t=1)，按 Lipman 的条件最优传输 (conditional OT)
    路径定义插值点 x_t 与目标 velocity u_t：
        x_t = t·x1 + (1 - (1-σ)·t)·x0      (随 t→1 越来越像 x1，噪声方差线性收缩)
        u_t = x1 - (1-σ)·x0                  (该路径的恒定速度场，DiT 的回归目标)
    σ 是数据端残余噪声标度；本仓库常取 σ=0 (compute_mu_t/compute_u_t 即简化形式)。
    """

    def __init__(self, sigma=1e-5):
        self.sigma = sigma

    def compute_mu_t(self, x1, t):
        # 插值均值 μ_t = t·x1。
        return t * x1

    def compute_sigma_t(self, t):
        # 噪声系数 σ_t 随 t 从 1 线性降到 σ。
        return 1 - (1 - self.sigma) * t

    def sample_x_t(self, x0, x1, t):
        # 路径上的样本点：x_t = μ_t + σ_t·x0。
        mu_t = self.compute_mu_t(x1, t)
        sigma_t = self.compute_sigma_t(t)
        return mu_t + sigma_t * x0

    def compute_u_t(self, x0, x1):
        # 目标 velocity field：u_t = x1 - (1-σ)·x0 (与 t 无关，是直线路径的恒定速度)。
        return x1 - (1 - self.sigma) * x0

    def compute_xt_ut(self, x1, t=None, x0=None):
        """采样一个训练对 (x_t, u_t) 及其时刻 times。

        Args:
            x1: (B, T, D) 数据端目标 latent。
            t:  (B,) 采样时刻；None 则每样本独立均匀采。
            x0: 噪声；None 则采标准高斯。

        Returns:
            xt (同 x1 形), ut (同 x1 形), times (B,)。
        """
        if x0 is None:
            x0 = torch.randn_like(x1, device=x1.device)
        if t is None:
            t = torch.rand(x1.size(0), dtype=x1.dtype, device=x1.device)
        times = t  # 保留 (B,) 形原始时刻给 DiT 的 time embedder
        # 把 t reshape 成 (B,1,1,...) 以便与 x1 逐元素广播。
        t = t.reshape(-1, *([1] * (x1.dim() - 1)))
        xt = self.sample_x_t(x0, x1, t)
        ut = self.compute_u_t(x0, x1)
        return xt, ut, times


class CausalHelper:
    """注意力 mask 构造工具：标准因果 mask，以及 DiT 用的块对角 (block-diagonal) chunk mask。

    两套 mask 对应两种注意力结构：
      - create_causal_mask_and_pos: 普通自回归三角因果 + padding，用于 LLM 段。
      - create_causal_chunk_mask_and_pos: 给 flow-matching DiT 序列 (前缀 C + 生成 Z) 的
        特殊 mask —— C 内部因果、Z 内部按 patch 块对角 (同一 patch 内全连)、Z→C 只看到
        当前 patch 之前的条件 token，并按 patch 给出 position id。
    """

    def create_causal_mask_and_pos(self, seq_lens, max_len):
        """构造标准因果 + padding 注意力 mask。

        Args:
            seq_lens: (B,) 各样本有效长度。
            max_len: 序列对齐长度。

        Returns:
            attn_mask (B, max_len, max_len) bool: True=可注意；
            seq_mask  (B, max_len) bool: padding 有效位；
            None: 占位 (此函数不产 position id)。
        """
        seq_mask = get_mask_from_lengths(seq_lens, max_len=max_len).unsqueeze(1)
        # triu(1)=严格上三角(未来位置)；取反得到下三角(含对角)的因果可见区。
        causal_mask = (
            torch.ones((max_len, max_len), device=seq_lens.device).triu(1).bool()
        )
        causal_mask = ~causal_mask.unsqueeze(0)
        attn_mask = seq_mask & causal_mask  # 因果 ∧ 非 padding
        return attn_mask, seq_mask.squeeze(1), None

    def create_causal_chunk_mask_and_pos(
        self,
        batch_size,
        C_lens,
        Z_lens,
        span_mask,
        patch_size=8,
    ):
        """为 flow-matching DiT 的"前缀 C + 生成 Z"序列构造分块注意力 mask 与 position id。

        序列布局：[ C(长 C_len) | Z(长 Z_len) ]。C 是条件前缀 (交错的 LLM hidden + 历史
        latent)，Z 是待去噪的噪声 latent，按 patch_size 切成若干 patch。注意力 mask 分四象限：

            | C2C |  -  |     行=query, 列=key
            | Z2C | Z2Z |

          - C2C: 标准因果 (前缀内部自回归可见)。
          - Z2Z: 块对角 —— 同一 patch 内的 latent 互相可见，跨 patch 不可见 (每个 patch 独立去噪)。
          - Z2C: 第 k 个 Z patch 只能看到 C 中其对应条件 token 之前的位置 (因果地暴露条件)。
          - 右上 C2Z: 恒 False (前缀不看未来的生成内容)。

        position id 设计：C 段为 0..C_len-1；每个 Z patch 的 pos 从它在 C 中对应的条件
        token 索引起算、再加 patch 内偏移 —— 让 Z 的 latent 在位置上"对齐"其条件 token。

        Args:
            C_lens / Z_lens: (B,) 前缀 / 生成段长度。
            span_mask: (B, C_len) bool，标记 C 中"历史 latent 起始"位置 (每 patch 一个 True)。
            patch_size: 一个 chunk (= hidden_patch + latent_patch) 的长度。

        Returns:
            attn_mask (B, L, L) bool, seq_mask (B, L) bool, pos_ids (B, L) float。
        """
        device = C_lens.device
        total_lens = C_lens + Z_lens
        attn_mask = torch.zeros(
            (batch_size, total_lens.max(), total_lens.max()),
            device=device,
            dtype=torch.bool,
        )
        pos_ids = []
        # | C2C |     |
        # | Z2C | Z2Z |
        for i in range(batch_size):
            C_len = C_lens[i]
            Z_len = Z_lens[i]

            # C2C parts are standard causal attention
            # 取严格上三角的逻辑非 = 下三角(含对角)，即标准因果可见区。
            attn_mask[i, :C_len, :C_len] = (
                torch.ones((C_len, C_len), device=device, dtype=torch.bool)
                .triu(1)
                .logical_not()
            )
            # Position ids in C parts are 0, 1, 2, ..., n
            c_pos = torch.arange(C_len, device=device, dtype=torch.float32)

            # Z2Z parts are block diag attention
            # 块对角：把 (Z_len//patch_size) 个 patch_size×patch_size 全 1 块拼到对角线上，
            # 实现"同 patch 内全连、跨 patch 屏蔽"。
            assert Z_len % patch_size == 0, "Z_len must be multiple of patch_size"
            attn_mask[i, C_len : C_len + Z_len, C_len : C_len + Z_len] = (
                torch.block_diag(
                    *[
                        torch.ones(
                            (patch_size, patch_size), device=device, dtype=torch.bool
                        )
                    ]
                    * (Z_len // patch_size)
                )
            )

            # Z2C parts is full attention before current patch latents
            # build according to span_mask
            j_indices = torch.arange(Z_len, device=device)  # Z 内每个位置 0..Z_len-1
            patch_indices = j_indices // patch_size  # 它属于第几个 Z patch
            # span_mask[i] 中第 k 个 True 的列位置 = 第 k 个 patch 在 C 里的条件 token 索引。
            patch_in_c_indices = torch.where(span_mask[i])[0][patch_indices]
            # 广播比较：Z 的第 j 行可见 C 的第 c 列 当且仅当 c < 该 patch 的条件 token 索引,
            # 即"只看到自己条件 token 之前的所有 C"。
            attn_mask[
                i,
                C_len + j_indices.unsqueeze(1),
                torch.arange(C_len, device=device).unsqueeze(0),
            ] = torch.arange(C_len, device=device).unsqueeze(
                0
            ) < patch_in_c_indices.unsqueeze(1)
            # Position ids in Z parts start from current patch latents index in C parts
            # Z pos = 条件 token 在 C 中的索引 + patch 内偏移 (j % patch_size)。
            z_pos = (patch_in_c_indices + j_indices % patch_size).to(torch.float32)
            pos_ids.append(torch.cat([c_pos, z_pos]))
        seq_mask = get_mask_from_lengths(total_lens, max_len=total_lens.max().item())
        # 各样本 pos_ids 长度不一，右侧补 0 对齐成 (B, L)。
        pos_ids = pad_sequence(pos_ids, batch_first=True, padding_value=0.0).to(
            C_lens.device
        )
        return attn_mask, seq_mask, pos_ids


class IOHelper:
    """latent 的输入/输出装配工具：归一化、VAE 重参数采样，以及 DiT 输入序列的拼装/拆解。

    核心方法 prepare_inputs_for_dit 把"LLM hidden + 历史 latent + 噪声 latent"按 patch
    交错成一条长序列 (供 DiT 用块对角 mask 做并行 flow-matching)，get_dit_outputs 再从
    DiT 输出里抠回纯 latent。归一化用预存的全局均值/方差，让 flow-matching 在标准化空间训练。
    """

    def __init__(self, latent_stats_path=None):
        """加载 latent 全局统计量 (mean/var)，缺省则不做归一化。"""
        if latent_stats_path is not None:
            latent_stats = torch.load(latent_stats_path, weights_only=False)
            self.global_mean = torch.as_tensor(latent_stats["mean"])
            self.global_var = torch.as_tensor(latent_stats["var"])
        else:
            self.global_mean = None
            self.global_var = None

    def normalize(self, x):
        """latent 标准化：(x - mean) / sqrt(var)，把 latent 拉到零均值单位方差。"""
        if self.global_mean is not None and self.global_var is not None:
            x = (x - self.global_mean.to(x.device)) / torch.sqrt(
                self.global_var.to(x.device)
            )
        return x

    def denormalize(self, x):
        """normalize 的逆操作：x * sqrt(var) + mean，把模型输出还原回原始 latent 尺度。"""
        if self.global_mean is not None and self.global_var is not None:
            x = x * torch.sqrt(self.global_var.to(x.device)) + self.global_mean.to(
                x.device
            )
        return x

    @staticmethod
    def sample_from_latent(latent):
        """VAE 重参数采样 (reparameterization trick)。

        输入 latent 沿通道维拼了 (mean, log_std) 两半，采 z = mean + ε·exp(log_std)，
        ε~N(0,I)。最后 transpose 把 (B, C, T) 调成 (B, T, C) 以匹配下游时间在前的约定。
        """
        mean, log_std = latent.chunk(2, 1)  # 沿 dim=1(通道) 切成均值与对数标准差
        z = mean + torch.randn_like(mean) * torch.exp(log_std)
        return z.transpose(1, 2)  # (B, C, T) → (B, T, C)

    @staticmethod
    def prepare_inputs_for_dit(
        hiddens,
        hidden_lens,
        latents,
        latent_lens,
        hidden_proj,
        latent_proj,
        noisy_proj,
        span_mask,
        hidden_patch_size,
        latent_patch_size,
        fm_helper,
        cfg_droprate=-1,
    ):
        """把 LLM hidden / 历史 latent / 噪声 latent 交错拼成 DiT 的训练输入序列。

        这是训练时 flow-matching 分支的"装配中枢"。最终 fm_seq 结构按样本是：

            [ 前缀 prefix(交错的 hidden+历史 latent，按 chunk) | 生成段 gen(交错的 hidden+噪声 latent) ]

        每个 chunk = 1 个 hidden token + latent_history_patch_size 个历史 latent；生成段每个
        chunk = 1 个 hidden + latent_patch_size 个噪声 latent。这样排布配合 CausalHelper 的块
        对角 mask，就能让 DiT 一次并行地对所有 patch 做 flow-matching，而每个 patch 仍只看到
        自己之前的条件。同时对 hidden 做 CFG dropout，对噪声 latent 算 flow-matching 真值 u_t。

        关键参数：
            hiddens (B, L, H): forward 里算好的 hiddens_for_fm (span 处=LLM hidden)。
            span_mask (B, L) bool: 哪些位置是要做 flow-matching 的 audio span。
            *_proj: 三条投影层 (hidden/latent/noisy → fm_hidden)。
            fm_helper: 用于采 (x_t, u_t)。
            cfg_droprate: 训练 CFG 丢 hidden 条件的概率。

        返回 (元信息见 forward 的解包注释)：
            fm_seq (B, S, fm_dim), fm_target(=u_t 拉平, N, latent_patch, latent_dim),
            fm_attn_mask, fm_seq_mask, fm_pos_ids, times(B,),
            fm_prefix_lengths(B,1), fm_gen_lengths(B,1), fm_gen_patch_size(int)。
        """
        assert hidden_patch_size == 1, "Hidden patch size > 1 is not supported."

        B, _, _, device = *hiddens.shape, hiddens.device

        # Gather span hidden states for flow matching using span_mask
        # 先把每条样本里散落在序列各处的 span hidden 抽出来左对齐，去掉文本/控制 token，
        # 后续只围绕这些 audio span 做装配。
        span_hidden_list = []
        for b in range(B):
            indices = span_mask[b].nonzero(as_tuple=False).squeeze(-1)
            span_hidden_list.append(hiddens[b, indices, :])
        hiddens = pad_sequence(span_hidden_list, batch_first=True, padding_value=0.0)
        hidden_lens = torch.tensor(
            [t.size(0) for t in span_hidden_list], device=device, dtype=torch.long
        )

        # Update span_mask to be all True for the new lengths
        # 抽取后 hiddens 已是"纯 span"，新 span_mask 即各样本前 hidden_lens 个位置为 True。
        max_len = hiddens.size(1)
        span_mask = torch.arange(max_len, device=device).expand(
            B, max_len
        ) < hidden_lens.unsqueeze(1)

        # Prepare history latents
        history_latents = latent_proj(latents)  # 历史/参考 latent → fm_dim
        fm_dim = history_latents.shape[-1]
        # patch_encoder 可能对 latent 做了时间下采样，故一个 latent_patch 对应到 history
        # 序列里是 latent_history_patch_size 个 token (二者长度比)。
        assert (latent_patch_size * history_latents.size(1) % latents.size(1)) == 0
        latent_history_patch_size = (
            latent_patch_size * history_latents.size(1) // latents.size(1)
        )

        # Prepare llm hidden with cfg masking
        # CFG: 以 cfg_droprate 概率整条丢掉 hidden 条件 (置 0)，训练无条件分支。
        cfg_mask = (
            torch.empty((B,), dtype=torch.float, device=latents.device).uniform_(0, 1)
            < cfg_droprate
        )
        hiddens = hidden_proj(mask_data(hiddens, cfg_mask))

        # Prepare noise latents
        # 采插值点 x_t 与目标速度 u_t；x_t 投到 fm_dim 作为生成段的噪声槽位，u_t 是回归目标。
        xt, ut, times = fm_helper.compute_xt_ut(latents)
        projected_noise = noisy_proj(xt)

        # Initialize empty fm_seq
        # 各段长度推导：
        #   hist_chunk_size  : 前缀里一个 chunk 的长度 = 1 hidden + 历史 latent 段。
        #   valid_patch_counts: 每样本 patch 数。
        #   fm_prefix_lengths: 前缀总长 = hidden 数 + 每 patch 额外携带的历史 latent。
        #   fm_gen_lengths   : 生成段总长 = 噪声 latent 数 + 每 patch 1 个 hidden。
        #   fm_gen_patch_size: 生成段一个 chunk = 1 hidden + latent_patch_size 噪声。
        hist_chunk_size = hidden_patch_size + latent_history_patch_size
        valid_patch_counts = latent_lens // latent_patch_size
        fm_prefix_lengths = hidden_lens + valid_patch_counts * (
            hist_chunk_size - hidden_patch_size
        )
        fm_gen_lengths = latent_lens + valid_patch_counts * hidden_patch_size
        fm_gen_patch_size = hidden_patch_size + latent_patch_size
        fm_seq_lengths = fm_prefix_lengths + fm_gen_lengths
        fm_seq = torch.zeros(
            (B, fm_seq_lengths.max().item(), fm_dim),
            dtype=history_latents.dtype,
            device=device,
        )
        fm_target = []
        patch_context_lengths = []
        history_latent_span_mask = torch.zeros(
            (B, fm_seq_lengths.max().item()), dtype=torch.bool, device=device
        )  # to mark start positions of each history latents
        # ↑ 标记每个历史 latent chunk 的起始位置，供 chunk mask 的 Z2C 定位条件 token。

        # Fill fm_seq
        for b in range(B):
            # Step 1: Interleave hiddens at span positions with patched_latents
            # 把"每个 span 的 1 个 hidden"与"对应历史 latent 段"沿 patch 维拼接，
            # 再拉平成 (num_spans*hist_chunk_size, D)，即前缀里交错排好的内容。
            interleaved = []
            span_mask_b = span_mask[b, : hidden_lens[b]]
            interleaved.append(
                hiddens[b, : hidden_lens[b]][span_mask_b].reshape(
                    valid_patch_counts[b], hidden_patch_size, fm_dim
                )
            )
            interleaved.append(
                history_latents[
                    b, : valid_patch_counts[b] * latent_history_patch_size, :
                ].reshape(valid_patch_counts[b], latent_history_patch_size, fm_dim)
            )
            interleaved = torch.cat(interleaved, dim=1)
            interleaved = rearrange(
                interleaved, "n h d -> (n h) d"
            )  # [num_spans*hist_chunk_size, D]

            # Step 2: Build mapping from input positions to fm positions
            # 每个输入位置在前缀里占多宽：span 占 hist_chunk_size，非 span 占 1。
            # 前缀和-自身 = 该位置在 fm_seq 中的起始下标 (exclusive prefix sum)。
            position_increment = torch.where(
                span_mask_b, hist_chunk_size, 1
            )  # span->hist_chunk_size, non-span->1
            fm_seq_positions = (
                torch.cumsum(position_increment, dim=0) - position_increment
            )

            # Step 3: Scatter non-span hiddens
            # 非 span 的 hidden (文本/控制 token) 直接散射到各自单格位置。
            non_span_mask = ~span_mask_b
            non_span_indices = fm_seq_positions[non_span_mask]  # [num_non_spans]
            fm_seq[b, non_span_indices, :] = hiddens[b, : hidden_lens[b]][
                non_span_mask, :
            ]

            # Step 4: Scatter interleaved span tokens
            # 每个 span 起点连续展开 hist_chunk_size 个下标，把 Step1 的交错块写进去。
            span_indices = fm_seq_positions[span_mask_b]  # [num_spans]
            span_indices_expanded = torch.stack(
                [span_indices + i for i in range(hist_chunk_size)], dim=1
            )  # [num_spans, hist_chunk_size]
            span_indices_flat = span_indices_expanded.reshape(
                -1
            )  # [num_spans*hist_chunk_size]
            fm_seq[b, span_indices_flat, :] = interleaved
            # span 起点即历史 latent chunk 的起点，标 True 供 chunk mask 用。
            history_latent_span_mask[b, span_indices] = True
            patch_context_lengths.append(span_indices.clone())

            # Step 5: Fill with noise latents at the end
            # 生成段紧接前缀之后：同样按 (1 hidden + latent_patch_size 噪声) 交错。
            noise_part = []
            span_mask_b = span_mask[b, : hidden_lens[b]]
            noise_part.append(
                hiddens[b, : hidden_lens[b]][span_mask_b].reshape(
                    valid_patch_counts[b], hidden_patch_size, fm_dim
                )
            )
            noise_part.append(
                projected_noise[b, : latent_lens[b], :].reshape(
                    valid_patch_counts[b], latent_patch_size, fm_dim
                )
            )
            noise_part = torch.cat(noise_part, dim=1)
            noise_part = rearrange(noise_part, "n h d -> (n h) d")
            # 生成段起点 = 前缀最后一个位置的起始 + 它的宽度 = 前缀总长。
            noise_start = fm_seq_positions[-1] + position_increment[-1]
            noise_end = noise_start + fm_gen_lengths[b]
            fm_seq[b, noise_start:noise_end, :] = noise_part

            # Step 6: prepare fm_target
            # 真值 u_t 也按 patch 切块，最终与 DiT 抠出的 pred 一一对应。
            ut_b = ut[b, : latent_lens[b], :]
            fm_target.append(rearrange(ut_b, "(n p) d -> n p d", p=latent_patch_size))

        # Construct fm_attn_mask and fm_pos_ids
        # 前缀=C, 生成段=Z；用块对角 chunk mask 让每个 patch 独立去噪、只看自己之前的条件。
        fm_attn_mask, fm_seq_mask, fm_pos_ids = (
            CausalHelper().create_causal_chunk_mask_and_pos(
                batch_size=B,
                C_lens=fm_prefix_lengths,
                Z_lens=fm_gen_lengths,
                span_mask=history_latent_span_mask,
                patch_size=fm_gen_patch_size,
            )
        )
        fm_prefix_lengths = fm_prefix_lengths.unsqueeze(1)  # (B,) → (B,1) 给下游切片用
        fm_gen_lengths = fm_gen_lengths.unsqueeze(1)
        fm_target = torch.cat(fm_target, dim=0)  # 拼成 (N, latent_patch_size, latent_dim)
        results = [
            fm_seq,
            fm_target,
            fm_attn_mask,
            fm_seq_mask,
            fm_pos_ids,
            times,
            fm_prefix_lengths,
            fm_gen_lengths,
            fm_gen_patch_size,
        ]
        return tuple(results)

    @staticmethod
    def prepare_meanflow_inputs_for_dit(
        *,
        hiddens: torch.Tensor,
        latents: torch.Tensor,
        latent_lens: torch.Tensor,
        hidden_proj,
        latent_proj,
        noisy_proj,
        span_mask: torch.Tensor,
        hidden_patch_size: int,
        latent_patch_size: int,
        cfg_mask: torch.Tensor,
        noise_latents: torch.Tensor,
    ) -> dict[str, Any]:
        """MeanFlow 训练版的 DiT 输入拼装 (prepare_inputs_for_dit 的精简/可重采版)。

        与标准版的差异：
          - 噪声 latent 由外部 noise_latents 传入 (而非内部 fm_helper 采样)，且 cfg_mask 也
            由外部给定 —— 这样 MeanFlow 训练能多次替换噪声而复用同一前缀 (见
            replace_noise_latents_in_fm_seq)。
          - 假设每样本所有 span 相邻 (span 起点 = patch_idx * hist_chunk_size)，拼装更直接。
          - 额外返回 noise_region_starts / noise_chunk_size / noise_inner_offset 等元信息，
            供后续就地替换噪声槽位。

        返回 dict，键含义见 return 块；其中 fm_seq (B, S, fm_dim) 即拼好的 DiT 输入序列。
        """
        if hidden_patch_size != 1:
            raise ValueError("MeanFlow training only supports hidden_patch_size=1.")

        batch_size = hiddens.size(0)
        device = hiddens.device

        span_hidden_list = []
        for batch_idx in range(batch_size):
            indices = span_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            span_hidden_list.append(hiddens[batch_idx, indices, :])
        hiddens = pad_sequence(span_hidden_list, batch_first=True, padding_value=0.0)
        hidden_lens = torch.tensor(
            [item.size(0) for item in span_hidden_list],
            device=device,
            dtype=torch.long,
        )

        history_latents = latent_proj(latents)
        fm_dim = history_latents.shape[-1]
        latent_history_patch_size = (
            latent_patch_size * history_latents.size(1) // latents.size(1)
        )

        hiddens = hidden_proj(mask_data(hiddens, cfg_mask))
        projected_noise = noisy_proj(noise_latents)

        hist_chunk_size = hidden_patch_size + latent_history_patch_size
        valid_patch_counts = latent_lens // latent_patch_size
        fm_prefix_lengths = hidden_lens + valid_patch_counts * (
            hist_chunk_size - hidden_patch_size
        )
        fm_gen_lengths = latent_lens + valid_patch_counts * hidden_patch_size
        fm_gen_patch_size = hidden_patch_size + latent_patch_size
        fm_seq_lengths = fm_prefix_lengths + fm_gen_lengths
        fm_seq = torch.zeros(
            (batch_size, int(fm_seq_lengths.max().item()), fm_dim),
            dtype=history_latents.dtype,
            device=device,
        )
        history_latent_span_mask = torch.zeros(
            (batch_size, int(fm_seq_lengths.max().item())),
            dtype=torch.bool,
            device=device,
        )
        noise_region_starts = []

        for batch_idx in range(batch_size):
            patch_count = int(valid_patch_counts[batch_idx].item())
            hidden_len = int(hidden_lens[batch_idx].item())
            if patch_count <= 0 or hidden_len <= 0:
                noise_region_starts.append(0)
                continue

            # 前缀交错块：每 patch = 1 hidden + 历史 latent 段，拼接后拉平。
            hidden_block = hiddens[batch_idx, :hidden_len].reshape(
                patch_count,
                hidden_patch_size,
                fm_dim,
            )
            history_block = history_latents[
                batch_idx,
                : patch_count * latent_history_patch_size,
                :,
            ].reshape(patch_count, latent_history_patch_size, fm_dim)
            interleaved = rearrange(
                torch.cat([hidden_block, history_block], dim=1),
                "n h d -> (n h) d",
            )

            # MeanFlow 版假设 span 等距排布：第 k 个 chunk 起点 = k * hist_chunk_size。
            span_indices = torch.arange(patch_count, device=device) * hist_chunk_size
            span_indices_expanded = torch.stack(
                [span_indices + idx for idx in range(hist_chunk_size)],
                dim=1,
            )
            fm_seq[batch_idx, span_indices_expanded.reshape(-1), :] = interleaved
            history_latent_span_mask[batch_idx, span_indices] = True

            # 生成段紧接前缀，记录其起点供后续替换噪声 latent。
            noise_start = patch_count * hist_chunk_size
            noise_region_starts.append(noise_start)
            noise_part = torch.cat(
                [
                    hidden_block,
                    projected_noise[
                        batch_idx,
                        : patch_count * latent_patch_size,
                        :,
                    ].reshape(patch_count, latent_patch_size, fm_dim),
                ],
                dim=1,
            )
            noise_part = rearrange(noise_part, "n h d -> (n h) d")
            noise_end = noise_start + int(fm_gen_lengths[batch_idx].item())
            fm_seq[batch_idx, noise_start:noise_end, :] = noise_part

        fm_attn_mask, fm_seq_mask, fm_pos_ids = (
            CausalHelper().create_causal_chunk_mask_and_pos(
                batch_size=batch_size,
                C_lens=fm_prefix_lengths,
                Z_lens=fm_gen_lengths,
                span_mask=history_latent_span_mask,
                patch_size=fm_gen_patch_size,
            )
        )
        return {
            "fm_seq": fm_seq,
            "fm_attn_mask": fm_attn_mask,
            "fm_seq_mask": fm_seq_mask,
            "fm_pos_ids": fm_pos_ids,
            "fm_prefix_lengths": fm_prefix_lengths.unsqueeze(1),
            "fm_gen_lengths": fm_gen_lengths.unsqueeze(1),
            "fm_gen_patch_size": fm_gen_patch_size,
            "noise_region_starts": torch.tensor(
                noise_region_starts,
                device=device,
                dtype=torch.long,
            ),
            "noise_chunk_size": fm_gen_patch_size,
            "noise_inner_offset": hidden_patch_size,
            "valid_patch_counts": valid_patch_counts,
            "latent_lens": latent_lens,
            "latent_patch_size": latent_patch_size,
        }

    @staticmethod
    def replace_noise_latents_in_fm_seq(
        prefix_data: dict[str, Any],
        new_noise_latents: torch.Tensor,
        noisy_proj,
    ) -> torch.Tensor:
        """在已拼好的 fm_seq 里就地替换噪声 latent 槽位 (复用前缀，仅换噪声)。

        MeanFlow 训练常对同一前缀重采若干份噪声/时间步；此函数据 prepare_meanflow_inputs
        留下的 noise_region_starts / chunk 元信息，把每个 patch 的 latent 槽位换成新噪声，
        避免重复拼装前缀。返回替换后的 fm_seq 副本 (不改原张量)。
        """
        projected = noisy_proj(new_noise_latents)
        fm_seq = prefix_data["fm_seq"].clone()
        starts = prefix_data["noise_region_starts"]
        chunk_size = int(prefix_data["noise_chunk_size"])
        inner_offset = int(prefix_data["noise_inner_offset"])
        latent_patch_size = int(prefix_data["latent_patch_size"])
        valid_patch_counts = prefix_data["valid_patch_counts"]

        for batch_idx in range(fm_seq.size(0)):
            patch_count = int(valid_patch_counts[batch_idx].item())
            if patch_count <= 0:
                continue
            base = int(starts[batch_idx].item())  # 生成段起点
            for patch_idx in range(patch_count):
                src_start = patch_idx * latent_patch_size  # 新噪声里该 patch 的起点
                # 目标位置 = 生成段起点 + patch 偏移 + chunk 内跳过 hidden 的 inner_offset。
                dst_start = base + patch_idx * chunk_size + inner_offset
                fm_seq[
                    batch_idx,
                    dst_start : dst_start + latent_patch_size,
                    :,
                ] = projected[
                    batch_idx,
                    src_start : src_start + latent_patch_size,
                    :,
                ]
        return fm_seq

    @staticmethod
    def get_dit_outputs(
        pred_v,
        fm_prefix_lengths,
        fm_gen_lengths,
        fm_gen_patch_size,
        latent_patch_size,
    ):
        """从 DiT 对整条交错序列的输出里抠出"纯 latent"的 velocity 预测。

        prepare_inputs_for_dit 把序列排成 [prefix | gen]，gen 段又是 (1 hidden +
        latent_patch_size latent) 的 chunk。这里先跳过 prefix 取 gen 段，按 chunk 切块，
        再丢掉每块开头的 hidden、只留尾部 latent_patch_size 个，最后全 batch 拼成
        (N, latent_patch_size, latent_dim)，与 fm_target 对齐算 loss。

        Args:
            pred_v: (B, S, latent_dim) DiT 对整条序列的输出。
            fm_prefix_lengths / fm_gen_lengths: (B, P) 各段长度 (P 一般为 1)。

        Returns:
            (N, latent_patch_size, latent_dim) 拼接后的 latent 预测。
        """
        B, P = fm_prefix_lengths.shape
        fm_pred = []
        for b in range(B):
            p_offset = 0  # 当前样本内已消费的序列偏移
            for p in range(P):
                # 跳过 prefix，切出本段 gen。
                latents_b = pred_v[
                    b,
                    p_offset + fm_prefix_lengths[b][p] : p_offset
                    + fm_prefix_lengths[b][p]
                    + fm_gen_lengths[b][p],
                ]
                # 按 chunk 切：每 chunk = 1 hidden + latent_patch_size 个 latent。
                latents_b = rearrange(
                    latents_b, "(n p) d -> n p d", p=fm_gen_patch_size
                )
                # extract only the latent parts
                # 丢掉 chunk 头部的 hidden，只保留尾部 latent_patch_size 个 latent。
                latents_b = latents_b[:, -latent_patch_size:, :]
                fm_pred.append(latents_b)
                p_offset += fm_prefix_lengths[b][p] + fm_gen_lengths[b][p]
        return torch.cat(fm_pred, dim=0)
