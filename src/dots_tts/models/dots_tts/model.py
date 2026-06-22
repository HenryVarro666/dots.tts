"""顶层模型装配与推理编排 / Top-level model assembly and inference orchestration.

本文件做什么 (What this file does)
----------------------------------
定义 ``DotsTtsModel`` —— dots.tts 的**最外层 nn.Module**。它把三大子系统组装成一个端到端
TTS 系统，并实现训练 forward 与流式/整段推理:

1. ``DotsTtsCore`` (``core.py``): Qwen2.5-1.5B 自回归主干 + flow-matching / MeanFlow 声学头
   (velocity field predictor / DiT)。负责文本→hidden、hidden+历史 latent→下一段连续 latent patch。
2. ``AudioVAE`` (BigVGAN 声码器, ``vocoder/bigvgan.py``): 在**连续潜在空间 (continuous latent)**
   做编/解码——把波形编成 latent (训练取条件)、把生成的 latent 解码回 24/48 kHz 波形 (支持流式)。
3. ``SpeakerXVectorFeatures`` (CAM++ x-vector, ``speaker/encoder.py``): 从 prompt audio 抽说话人
   嵌入 (x-vector)，经 ``xvec_proj`` 得到全局条件 ``g_cond``，给声学头提供音色/说话人条件。

在推理数据流里的位置 (Position in the inference pipeline)
------------------------------------------------------
generation_schedule (文本 + audio span 占位符 token)
  → ``_prefill``: 一次性把 prompt 文本/音频喂进 LLM，发现 prompt patch 边界
  → ``_decode`` 主循环: 文本段走 ``_consume_text_schedule`` (step LLM)，
     audio span 走 ``_decode_next_audio`` (flow-matching / MeanFlow 解 ODE 得一个 latent patch)
     再 ``_consume_audio_patch`` (patch 状态机: 把刚生成的 latent 反馈回 LLM 与 patch encoder)
  → latent patch 流式喂给 ``AudioVAE.stream_step`` 解成音频块 → yield 给调用方。

关键类/函数清单 (Key classes / functions)
----------------------------------------
- ``DotsTtsModel``: 主装配类。
- ``_GenerateState`` / ``_PromptConditioning`` / ``_PromptFeatureCacheEntry``:
  推理过程中的可变状态、prompt 条件、prompt 特征缓存项 (dataclass)。
- ``_GenerateLengthBucket``: 把"最大生成长度"离散成若干 bucket，配合 ``torch.compile`` 缓存
  与静态 buffer 复用 (避免 CUDA graph / 编译产物随长度爆炸)。
- 推理核心: ``_prefill`` / ``_consume_text_schedule`` / ``_decode_next_audio`` /
  ``_consume_audio_patch`` / ``_decode`` / ``generate_audio`` / ``generate_audio_stream``。
- 训练侧: ``prepare_training_inputs`` / ``_prepare_loss_metadata`` / ``forward``
  (CE / flow-matching / EOS 三路损失装配)。
- warmup / 编译: ``run_warmup`` / ``_warmup_fm_bucket`` / ``_get_compiled_model``。

设计要点 (Design notes)
----------------------
dots.tts 走**连续潜在自回归**——没有离散音频 token。每个 "audio span" 占位符对应一个
latent patch；LLM 提供该 patch 的上下文 hidden，声学头用 flow-matching 在连续空间里把
高斯噪声沿 velocity field 积分 (ODE solver) 成一个 latent patch。CFG (classifier-free
guidance) 通过维护一条并行的 "null 条件" 序列 (``fm_cfg_sequence``) 实现。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen2Config

from dots_tts.models.dots_tts.config import ModelConfig
from dots_tts.models.dots_tts.core import DotsTtsCore, DotsTtsForwardOutput
from dots_tts.modules.speaker.encoder import SpeakerXVectorFeatures
from dots_tts.modules.vocoder.bigvgan import AudioVAE
from dots_tts.training.losses import LossMasks, LossTerm, LossTerms
from dots_tts.utils.profiling import measure_inference
from dots_tts.utils.tokenizer import AUDIO_GEN_START_TOKEN, require_token_id
from dots_tts.utils.util import get_dtype


@dataclass
class _GenerateState:
    """单次自回归生成过程中的全部可变状态 / Mutable state threaded through one generation.

    在 ``_generate_latents_stream`` 里创建一次，随 prefill→decode 主循环不断被原地更新。

    字段含义:
    - ``llm_cache``: LLM 的 KV cache (past_key_values)，自回归增量解码的关键。
    - ``llm_hiddens``: 上一步 LLM 输出的最后一层 hidden，形状约 (B, n, D_llm)；既用作
      下一个 audio patch 的条件，也用于 EOS 判定。
    - ``patch_encoder_state``: patch encoder 的流式解码状态 (conv_tail + 各层 KV cache +
      seq_len)，把已生成的 latent patch 编回 LLM 输入嵌入。
    - ``fm_seq_len``: flow-matching 输入序列当前有效长度 (历史 hidden+latent 已填多少)。
    - ``fm_capacity``: ``fm_sequence`` 静态 buffer 的总容量 (按 bucket 预分配)。
    - ``fm_sequence``: flow-matching 条件序列 (条件分支)，形状 (1, fm_capacity, fm_hidden_size)。
    - ``fm_cfg_sequence``: 与上面并行的 **null 条件** 序列，供 classifier-free guidance (CFG)
      的无条件分支使用。
    - ``fm_null_g_cond``: 全零的 g_cond 占位 (无说话人条件时的兜底 / CFG 无条件全局向量)。
    - ``end_flag``: EOS 触发标志；一旦置位，当前 audio patch 产出后即终止生成。
    """

    llm_cache: Any | None = None
    llm_hiddens: torch.Tensor | None = None
    patch_encoder_state: Any | None = None
    fm_seq_len: int = 0
    fm_capacity: int = 0
    fm_sequence: torch.Tensor | None = None
    fm_cfg_sequence: torch.Tensor | None = None
    fm_null_g_cond: torch.Tensor | None = None
    end_flag: bool = False


@dataclass(frozen=True)
class _PromptConditioning:
    """从 prompt audio 派生出的全部条件 / Conditioning derived from the prompt audio.

    - ``prompt_patches``: 归一化后的 prompt latent，按 patch 切好，形状 (B, S, patch_size, D)。
      用于把参考音频的内容/韵律"喂回去"做 prompt-continuation (语音克隆的 prefill)。
    - ``prompt_latents``: prompt 的连续 latent (已采样、去掉末尾一个 patch)，形状 (B, T, D)，
      喂给 patch encoder 生成对应的 LLM 输入嵌入。
    - ``g_cond``: 经 ``xvec_proj`` 投影后的说话人全局条件向量，形状 (B, fm_hidden_size)，
      贯穿整个声学头解码，决定输出音色。

    三者非全有非全无: 只给参考音频 (不做 prefill) 时只有 ``g_cond``。
    """

    prompt_patches: torch.Tensor | None = None
    prompt_latents: torch.Tensor | None = None
    g_cond: torch.Tensor | None = None


@dataclass
class _PromptFeatureCacheEntry:
    """prompt 特征 LRU 缓存的一项 / One entry in the prompt-feature LRU cache.

    同一段参考音频常被复用 (批量合成同一说话人)，缓存这两项重特征以省去重复前向:
    - ``speaker_embedding``: CAM++ x-vector 抽出的说话人嵌入。
    - ``prompt_latent_distribution``: AudioVAE 抽出的 prompt latent 分布 (采样前)。
    缓存键见 ``_prepare_prompt_audio_for_conditioning`` 的 SHA1 摘要。
    """

    speaker_embedding: torch.Tensor | None = None
    prompt_latent_distribution: torch.Tensor | None = None


@dataclass(frozen=True)
class _GenerateLengthBucket:
    """生成长度分桶 / A discrete bucket of max generation length (in audio patches).

    为什么要分桶: ``torch.compile`` 与静态 buffer / CUDA graph 对张量形状敏感，若每个长度都
    重新编译会极慢。把"最大生成长度"向上取整到固定档位 (32/64/.../1024)，同一档共享一份编译
    产物和静态工作区，是典型的 bucketing / padding-to-bucket 优化。
    """

    size: int

    def run_warmup(
        self,
        model: "DotsTtsModel",
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
    ) -> None:
        """对本 bucket 触发一次完整 warmup / Warm up all compiled paths for this bucket.

        逐一编译/预热 flow-matching 头、patch encoder 与声码器，使首个真实请求不必承担
        ``torch.compile`` 的冷启动开销。构造一个"纯 audio span"的 dummy schedule (开头一个
        AUDIO_GEN_START，其余全是 audio_gen_span 占位符) 跑通完整流式管线一步即可。
        """
        model._warmup_fm_bucket(
            max_audio_patch_count=self.size,
            precision=precision,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        model._warmup_patch_encoder_bucket(
            max_audio_patch_count=self.size,
            precision=precision,
        )
        device = next(model.core.parameters()).device
        # dummy schedule: 第 0 位是生成起始 token，其余 size 个都是 audio span 占位符，
        # 让 warmup 走满 prefill + 一段 decode 的代码路径。
        generation_schedule = torch.full(
            (1, self.size + 1),
            fill_value=model.core.audio_gen_span_id,
            dtype=torch.long,
            device=device,
        )
        generation_schedule[0, 0] = model.audio_gen_start_id
        warmup_inputs = {"generation_schedule": generation_schedule}

        # 只需驱动出第一个 chunk 即说明整条管线已被编译/预热，随即提前 return。
        for _ in model.generate_audio_stream(
            warmup_inputs,
            precision=precision,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        ):
            return
        raise RuntimeError(
            f"Warmup produced no audio chunk for generate bucket {self.size}."
        )


class DotsTtsModel(nn.Module):
    """Full train/infer model assembly around the dots.tts core network.

    最外层装配类。组合 ``core`` (LLM + 声学头)、``vocoder`` (AudioVAE)、``xvector_extractor``
    (说话人编码) 三部分，并对外提供:
    - 训练: ``forward`` / ``prepare_training_batch`` / ``prepare_training_inputs``;
    - 推理: ``generate_audio`` (整段) / ``generate_audio_stream`` (流式) / ``run_warmup``;
    - 权重 IO: ``save_pretrained`` / ``from_pretrained`` / ``load_pretrained_weights``。

    类常量约定 (class-level constants)
    - ``_GENERATE_LENGTH_BUCKETS``: 支持的生成长度档位 (见 ``_GenerateLengthBucket``)。
    - ``_COMPILE_TARGETS``: 哪些子模块允许 ``torch.compile`` (FM 头 / patch encoder / 声码器)。
    - ``_ARTIFACT_ALIASES``: 保存权重时去重的 tied-weight 别名 (lm_head 与 embed_tokens 共享存储)。
    - ``*_FILENAME`` / ``REQUIRED_ARTIFACT_FILES``: HF 风格目录的产物文件名与必需文件清单。
    """

    _GENERATE_LENGTH_BUCKETS = (
        _GenerateLengthBucket(32),
        _GenerateLengthBucket(64),
        _GenerateLengthBucket(128),
        _GenerateLengthBucket(256),
        _GenerateLengthBucket(512),
        _GenerateLengthBucket(1024),
    )
    _COMPILE_TARGETS = frozenset(
        {
            "FM",
            "patch_encoder",
            "vocoder",
        }
    )
    _optimize_enabled = True
    _PROMPT_FEATURE_CACHE_MAX_ENTRIES = 256
    CONFIG_FILENAME = "config.json"
    HF_MODEL_TYPE = "dots_tts"
    HF_ARCHITECTURES = ["DotsTTSForConditionalGeneration"]
    LATENT_STATS_FILENAME = "latent_stats.pt"
    LLM_CONFIG_FILENAME = "llm_config.json"
    MODEL_FILENAME = "model.safetensors"
    VOCODER_FILENAME = "vocoder.safetensors"
    SPEAKER_ENCODER_FILENAME = "speaker_encoder.safetensors"
    _ARTIFACT_ALIASES = (("llm.lm_head.weight", "llm.model.embed_tokens.weight"),)
    REQUIRED_ARTIFACT_FILES = (
        CONFIG_FILENAME,
        LATENT_STATS_FILENAME,
        LLM_CONFIG_FILENAME,
        MODEL_FILENAME,
        VOCODER_FILENAME,
        SPEAKER_ENCODER_FILENAME,
    )

    # region Module assembly and checkpoint IO
    def __init__(
        self,
        config: ModelConfig,
        tokenizer,
        latent_stats_path: str | Path,
        llm_config: Qwen2Config,
    ):
        """装配整套模型 / Assemble core + vocoder + speaker encoder.

        参数:
        - ``config``: dots.tts 自身的 ModelConfig (patch_size、vocoder、cfg_droprate 等)。
        - ``tokenizer``: 文本 tokenizer，含 audio span 等特殊 token。
        - ``latent_stats_path``: latent 归一化统计 (mean/std) 的文件路径，供 io_helper
          做 normalize/denormalize。
        - ``llm_config``: Qwen2 主干的配置。

        声码器与说话人编码器置 eval 且冻结梯度 (推理用，不参与训练)；声码器去掉 weight_norm
        以便流式/编译。同时初始化各类编译缓存与静态工作区字典 (推理期才真正分配)。
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.latent_stats_path = Path(latent_stats_path)
        self.audio_gen_start_id = require_token_id(
            self.tokenizer, AUDIO_GEN_START_TOKEN
        )

        self.core = DotsTtsCore(
            config,
            llm_config=llm_config,
            tokenizer=tokenizer,
            latent_stats_path=self.latent_stats_path,
        )
        self.vocoder = AudioVAE(config.vocoder).eval()
        self.vocoder.remove_weight_norm()  # 推理/编译前移除 weight_norm 的参数重参数化
        self.hop_size = self.vocoder.hop_size  # 每个 latent 帧对应的波形采样点数
        self.xvector_extractor = SpeakerXVectorFeatures(
            sample_rate=self.vocoder.sample_rate,
            campplus_embedding_size=config.campplus_embedding_size,
            max_audio_seconds=config.xvec_max_audio_seconds,
        ).eval()

        for param in self.vocoder.parameters():
            param.requires_grad = False
        for param in self.xvector_extractor.parameters():
            param.requires_grad = False
        self._optimize_enabled = True
        # (key, signature) -> 已编译的 callable;signature 区分不同 bucket/dtype 的编译变体。
        self._compiled_models: dict[
            tuple[str, tuple[Any, ...] | None], Callable[..., Any]
        ] = {}
        # prompt 特征 LRU 缓存 (OrderedDict 实现 move_to_end / popitem 淘汰)。
        self._prompt_feature_cache: OrderedDict[
            str, _PromptFeatureCacheEntry
        ] = OrderedDict()
        # 按 (bucket, device, dtype) 复用的 flow-matching 静态 buffer 工作区。
        self._static_generate_workspaces: dict[tuple[Any, ...], dict[str, Any]] = {}
        # 按 (total_len, device, dtype) 复用的单步 FM 解码工作区 (input/cfg/mask/pos)。
        self._fm_decode_workspaces: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}

    def set_optimize(self, optimize: bool) -> None:
        """开/关推理优化 (torch.compile + 静态 buffer) / Toggle inference optimizations.

        关闭时清空编译缓存，回退到 eager + 动态分配状态 (便于调试、规避编译相关 bug)。
        """
        self._optimize_enabled = bool(optimize)
        if not self._optimize_enabled:
            self._compiled_models.clear()

    def set_cfg_droprate(
        self,
        cfg_droprate: float | None = None,
        xvec_drop_rate: float | None = None,
    ) -> None:
        """调整训练期 dropout 率 / Override CFG and speaker-conditioning drop rates.

        - ``cfg_droprate``: 训练时丢弃条件以学习无条件分支的概率，classifier-free
          guidance (CFG) 之所以能在推理期工作就靠这个。
        - ``xvec_drop_rate``: 丢弃说话人 x-vector 条件的概率。
        三处 (model.config / core.config / core 字段) 同步更新，保持一致。
        """
        if cfg_droprate is not None:
            self.config.cfg_droprate = cfg_droprate
            self.core.config.cfg_droprate = cfg_droprate
            self.core.cfg_droprate = cfg_droprate

        if xvec_drop_rate is not None:
            self.config.xvec_drop_rate = xvec_drop_rate
            self.core.config.xvec_drop_rate = xvec_drop_rate
            self.core.xvec_drop_rate = xvec_drop_rate

    @classmethod
    def _resolve_generate_length_bucket(
        cls,
        max_generate_length: int,
    ) -> _GenerateLengthBucket:
        """把请求长度向上取整到最小的可容纳 bucket / Round up to the smallest fitting bucket.

        超过最大档位 (1024) 即报错——这是编译/静态 buffer 支持的硬上限。
        """
        requested = int(max_generate_length)
        if requested <= 0:
            raise ValueError("max_generate_length must be positive.")
        for bucket in cls._GENERATE_LENGTH_BUCKETS:
            if requested <= bucket.size:
                return bucket
        raise ValueError(
            "max_generate_length exceeds the largest supported compile bucket: "
            f"max_generate_length={requested} "
            f"max_supported={cls._GENERATE_LENGTH_BUCKETS[-1].size}."
        )

    @torch.no_grad()
    def run_warmup(
        self,
        *,
        max_generate_length: int,
        precision: str = "bfloat16",
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
    ) -> None:
        """提前预热到指定最大长度 / Eagerly warm up all buckets up to a ceiling.

        服务启动时调用: 把 ``<= max_generate_length`` 的所有 bucket 逐一编译预热，使后续真实
        请求免去首次编译延迟。在 ``no_grad`` 下进行，纯副作用。
        """
        ceiling_bucket = self._resolve_generate_length_bucket(max_generate_length)
        warmup_buckets = tuple(
            bucket
            for bucket in self._GENERATE_LENGTH_BUCKETS
            if bucket.size <= ceiling_bucket.size
        )
        bucket_sizes = [bucket.size for bucket in warmup_buckets]
        logger.info(
            "Inference warmup started: requested_max_generate_length={} bucket_sizes={}",
            int(max_generate_length),
            bucket_sizes,
        )
        for bucket in warmup_buckets:
            bucket.run_warmup(
                self,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            )
        logger.info(
            "Inference warmup completed: requested_max_generate_length={} bucket_sizes={}",
            int(max_generate_length),
            bucket_sizes,
        )

    def _resolve_state_audio_patch_count(self, max_audio_patch_count: int) -> int:
        """决定静态状态按多少 patch 分配 / Resolve the patch count to size static buffers.

        优化开启时按 bucket 取整 (利于复用/编译)；关闭时按实际请求精确分配。
        """
        requested = int(max_audio_patch_count)
        if requested <= 0:
            raise ValueError("max_audio_patch_count must be positive.")
        if not self._optimize_enabled:
            return requested
        return self._resolve_generate_length_bucket(requested).size

    def _warmup_fm_bucket(
        self,
        *,
        max_audio_patch_count: int,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
    ) -> None:
        """编译/预热 flow-matching 头 / Compile & warm the FM (acoustic) head for a bucket.

        手动把 ``fm_seq_len`` 拉满到 ``fm_capacity``，强制走最长历史路径，从而触发该 bucket 对应
        签名下的 ``torch.compile``。仅副作用，结果丢弃。
        """
        dtype = get_dtype(precision)
        device = next(self.core.parameters()).device
        use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            state = self._allocate_generate_state(
                max_audio_patch_count=max_audio_patch_count,
                device=device,
                dtype=dtype,
            )
            # 拉满有效长度以编译"最大历史"分支 (覆盖该 bucket 的最坏情况)。
            state.fm_seq_len = state.fm_capacity
            self._decode_next_audio(
                state,
                device=device,
                g_cond=None,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            )

    def _warmup_patch_encoder_bucket(
        self,
        *,
        max_audio_patch_count: int,
        precision: str,
    ) -> None:
        """编译/预热 patch encoder 的单步解码 / Compile & warm patch encoder decode.

        构造一个零 latent patch，跑一次 ``decode_patch`` 触发编译。CPU 上状态用 fp32
        (autocast 仅在 CUDA + 半精度时启用)。
        """
        dtype = get_dtype(precision)
        device = next(self.core.parameters()).device
        state_dtype = dtype if device.type == "cuda" else torch.float32
        use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            state_audio_patch_count = self._resolve_state_audio_patch_count(
                max_audio_patch_count
            )
            patch_encoder_state = self.core.patch_encoder.init_decode_state(
                max_audio_patch_count=state_audio_patch_count,
                batch_size=1,
                device=device,
                dtype=state_dtype,
            )
            audio_patch = torch.zeros(
                (
                    1,
                    self.core.patch_encoder.patch_size,
                    self.core.latent_dim,
                ),
                dtype=state_dtype,
                device=device,
            )
            # patch encoder 期望"原始尺度"的 latent，故对零张量也走一遍 denormalize。
            audio_patch = self.core.io_helper.denormalize(audio_patch)
            patch_encoder_decode = self._get_compiled_model(
                "patch_encoder.decode_patch",
                self.core.patch_encoder.decode_patch,
                signature=self._patch_encoder_compile_signature(patch_encoder_state),
            )
            positions = torch.arange(
                self.core.patch_encoder.out_ds_rate,
                device=device,
                dtype=torch.long,
            )
            with measure_inference("patch_encoder"):
                patch_encoder_decode(
                    audio_patch,
                    patch_encoder_state.conv_tail,
                    patch_encoder_state.layer_caches,
                    positions,
                )

    def _compile_callable(
        self,
        key: str,
        model: Callable[..., Any],
        *,
        signature: tuple[Any, ...] | None = None,
    ) -> Callable[..., Any]:
        """按 (key, signature) 缓存 torch.compile 产物 / Compile once, cache per signature.

        ``signature`` 编码了形状/dtype 等会触发重编译的因素 (如 bucket 容量)。``fullgraph=True,
        dynamic=False`` 要求整图静态编译以拿到最佳性能;patch encoder 用 "default" 模式，其余
        用 "reduce-overhead" (CUDA graph，减少 kernel launch 开销)。
        """
        compile_target = key.split(".", maxsplit=1)[0]
        cache_key = (key, signature)
        compiled = self._compiled_models.get(cache_key)
        if compiled is None:
            mode = (
                "default"
                if key == "patch_encoder.decode_patch"
                else "reduce-overhead"
            )
            compiled = torch.compile(
                model,
                mode=mode,
                fullgraph=True,
                dynamic=False,
            )
            self._compiled_models[cache_key] = compiled
            logger.info(
                "Compiled inference target: key={} target={} signature={}",
                key,
                compile_target,
                signature,
            )
        return compiled

    def _get_compiled_model(
        self,
        key: str,
        model: Callable[..., Any],
        *,
        signature: tuple[Any, ...] | None = None,
    ) -> Callable[..., Any]:
        """取一个 (可能已编译的) callable / Return compiled callable, or the raw one.

        当优化关闭、或该 target 不在 ``_COMPILE_TARGETS`` 白名单内时，原样返回，不做编译。
        """
        compile_target = key.split(".", maxsplit=1)[0]
        if not self._optimize_enabled or compile_target not in self._COMPILE_TARGETS:
            return model
        return self._compile_callable(
            key,
            model,
            signature=signature,
        )

    def _get_compiled_method(
        self,
        key: str,
        owner: Any,
        method_name: str,
        *,
        signature: tuple[Any, ...] | None = None,
    ) -> Callable[..., Any]:
        """编译一个绑定方法 / Compile a bound method on ``owner``.

        与 ``_get_compiled_model`` 类似，但针对实例方法: 取**未绑定**的原函数 (透过
        ``__wrapped__`` 穿过 ``@torch.no_grad`` 等装饰器) 编译，再用 ``partial`` 把 ``owner``
        作为 self 重新绑定。这样编译图里不含闭包/装饰器层，利于 ``fullgraph``。
        """
        bound_method = getattr(owner, method_name)
        compile_target = key.split(".", maxsplit=1)[0]
        if not self._optimize_enabled or compile_target not in self._COMPILE_TARGETS:
            return bound_method

        raw_method = getattr(type(owner), method_name)
        # 透过 functools.wraps 链拿到真正的原函数，避免把装饰器逻辑编进图里。
        raw_callable = getattr(raw_method, "__wrapped__", raw_method)
        compiled = self._compile_callable(
            key,
            raw_callable,
            signature=signature,
        )
        return partial(compiled, owner)

    def _allocate_generate_state(
        self,
        *,
        max_audio_patch_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _GenerateState:
        """分配/复用一次生成所需的全部静态状态 / Allocate (or reuse) static generate state.

        关键: flow-matching 历史序列按 ``patch 数 × (hidden_patch_size + latent_patch_size)``
        预留容量——每个 audio patch 在历史里占 1 个 hidden 槽 + ``latent_patch_size`` 个 latent
        槽。静态 buffer 按 (bucket, device, dtype) 缓存复用并清零，避免每次生成重新 malloc，
        也是 CUDA graph 能稳定捕获的前提。优化关闭时才即时初始化 patch encoder 状态。
        """
        state_dtype = dtype if device.type == "cuda" else torch.float32
        state_audio_patch_count = self._resolve_state_audio_patch_count(
            max_audio_patch_count
        )
        # 历史序列容量 = patch 数 × 每 patch 在 FM 序列里占的槽位 (1 个 hidden + latent_patch_size 个 latent)。
        fm_capacity = state_audio_patch_count * (
            self.core.hidden_patch_size + self.core.latent_patch_size
        )
        workspace_key = (
            state_audio_patch_count,
            str(device),
            state_dtype,
        )
        workspace = self._static_generate_workspaces.get(workspace_key)
        if workspace is None:
            workspace = {
                "fm_sequence": torch.zeros(
                    (1, fm_capacity, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
                "fm_cfg_sequence": torch.zeros(
                    (1, fm_capacity, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
                "fm_null_g_cond": torch.zeros(
                    (1, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
            }
            self._static_generate_workspaces[workspace_key] = workspace
        else:
            # 复用旧 buffer：只清零 (null_g_cond 恒为零，无需重置)。
            workspace["fm_sequence"].zero_()
            workspace["fm_cfg_sequence"].zero_()

        patch_encoder_state = None
        if not self._optimize_enabled:
            patch_encoder_state = self.core.patch_encoder.init_decode_state(
                max_audio_patch_count=state_audio_patch_count,
                batch_size=1,
                device=device,
                dtype=state_dtype,
            )

        return _GenerateState(
            patch_encoder_state=patch_encoder_state,
            fm_seq_len=0,
            fm_capacity=fm_capacity,
            fm_sequence=workspace["fm_sequence"],
            fm_cfg_sequence=workspace["fm_cfg_sequence"],
            fm_null_g_cond=workspace["fm_null_g_cond"],
        )

    @staticmethod
    def _tensor_storage_signature(tensor: torch.Tensor) -> tuple:
        """张量底层存储的去重指纹 / Identity of a tensor's underlying storage.

        两个张量若 data_ptr/offset/size/stride/dtype 全同，则它们其实共享同一块显存
        (tied weight)。保存时据此去重，避免把同一权重写两份。
        """
        return (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.size()),
            tuple(tensor.stride()),
            tensor.dtype,
        )

    @classmethod
    def _build_artifact_state_dict(cls, module) -> dict[str, torch.Tensor]:
        """构造去重后的可保存 state_dict / Build a dedup'd, CPU state dict for saving.

        safetensors 不允许多个 key 共享同一存储，故这里:
        1) 删掉 ``_ARTIFACT_ALIASES`` 里与规范键共享存储的冗余别名键 (如 lm_head ↔ embed_tokens);
        2) 跳过任何与已见存储重复的张量;
        3) 其余张量 ``detach().cpu().contiguous()`` 后落盘。
        加载时再由 ``_restore_artifact_state_dict`` 把别名补回。
        """
        state_dict = module.state_dict()
        skip_keys = set()

        for redundant_key, canonical_key in cls._ARTIFACT_ALIASES:
            redundant_tensor = state_dict.get(redundant_key)
            canonical_tensor = state_dict.get(canonical_key)
            if (
                redundant_tensor is not None
                and canonical_tensor is not None
                and cls._tensor_storage_signature(redundant_tensor)
                == cls._tensor_storage_signature(canonical_tensor)
            ):
                skip_keys.add(redundant_key)

        cleaned_state_dict = {}
        seen_storage = set()
        for key, value in state_dict.items():
            if key in skip_keys:
                continue

            storage_signature = cls._tensor_storage_signature(value)
            if storage_signature in seen_storage:
                continue

            seen_storage.add(storage_signature)
            cleaned_state_dict[key] = value.detach().cpu().contiguous()

        return cleaned_state_dict

    @classmethod
    def _restore_artifact_state_dict(cls, state_dict: dict, module) -> dict:
        """把保存时去重掉的别名键补回 / Re-tie alias keys dropped during saving.

        若规范键在、别名键缺、且模块确实需要该别名，就让别名指向规范键的张量 (恢复 weight tying)。
        """
        restored_state_dict = dict(state_dict)
        for redundant_key, canonical_key in cls._ARTIFACT_ALIASES:
            if (
                canonical_key in restored_state_dict
                and redundant_key not in restored_state_dict
                and redundant_key in module.state_dict()
            ):
                restored_state_dict[redundant_key] = restored_state_dict[canonical_key]
        return restored_state_dict

    @classmethod
    def _save_artifact_module(cls, module, path: Path) -> None:
        """把单个子模块去重后存为 safetensors / Save one submodule to a safetensors file."""
        save_file(cls._build_artifact_state_dict(module), path)

    @classmethod
    def _load_artifact_module(cls, module, path: Path):
        """加载子模块权重并补回别名 / Load a submodule, re-tie aliases, fail on key mismatch.

        ``strict=False`` 配合显式检查: 允许别名通过 ``_restore_artifact_state_dict`` 补齐, 但若仍有
        缺失/多余键则报错 (防止静默加载错权重)。
        """
        state_dict = load_file(path, device="cpu")
        restored_state_dict = cls._restore_artifact_state_dict(state_dict, module)
        mismatch = module.load_state_dict(restored_state_dict, strict=False)
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"Failed to load {path}: {mismatch}")
        return module

    @classmethod
    def _validate_pretrained_directory(
        cls, pretrained_model_name_or_path: str | Path
    ) -> Path:
        """校验预训练目录含全部必需产物文件 / Validate the pretrained directory layout."""
        pretrained_path = Path(pretrained_model_name_or_path).expanduser().resolve()
        missing_files = [
            name
            for name in cls.REQUIRED_ARTIFACT_FILES
            if not (pretrained_path / name).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Pretrained path {pretrained_path} is missing required files: {missing_files}"
            )
        return pretrained_path

    @classmethod
    def _load_pretrained_config(cls, pretrained_path: Path) -> ModelConfig:
        """读并校验 dots.tts 的 ModelConfig / Load & validate the dots.tts ModelConfig."""
        return ModelConfig.model_validate(
            json.loads(
                (pretrained_path / cls.CONFIG_FILENAME).read_text(encoding="utf-8")
            )
        )

    @staticmethod
    def _save_llm_config(llm_config: Qwen2Config, path: Path) -> None:
        """把 Qwen2 主干配置写成 JSON / Persist the Qwen2 backbone config to JSON."""
        path.write_text(
            json.dumps(llm_config.to_dict(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _load_llm_config(path: Path) -> Qwen2Config:
        """从 JSON 还原 Qwen2 主干配置 / Load the Qwen2 backbone config from JSON."""
        return Qwen2Config.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _tie_llm_weights(self) -> None:
        """重建 LLM 的权重绑定 / Re-tie LLM input/output embeddings if supported.

        加载去重后的 state_dict 后, lm_head 与 embed_tokens 的共享需重新建立 (见 ``_ARTIFACT_ALIASES``)。
        """
        if hasattr(self.core.llm, "tie_weights"):
            self.core.llm.tie_weights()

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """以 HF 风格目录保存整套模型 / Save the full model as an HF-style directory.

        写出: config.json (含 model_type/architectures)、llm_config.json、tokenizer、
        latent_stats、以及 core/vocoder/speaker_encoder 三个 safetensors 权重。
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config_payload = self.config.to_declared_dict()
        config_payload["model_type"] = self.HF_MODEL_TYPE
        config_payload["architectures"] = list(self.HF_ARCHITECTURES)
        (save_directory / self.CONFIG_FILENAME).write_text(
            json.dumps(config_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self._save_llm_config(
            self.core.llm.config,
            save_directory / self.LLM_CONFIG_FILENAME,
        )
        self.tokenizer.save_pretrained(save_directory)
        shutil.copy2(
            self.latent_stats_path,
            save_directory / self.LATENT_STATS_FILENAME,
        )
        self._save_artifact_module(self.core, save_directory / self.MODEL_FILENAME)
        self._save_artifact_module(self.vocoder, save_directory / self.VOCODER_FILENAME)
        self._save_artifact_module(
            self.xvector_extractor,
            save_directory / self.SPEAKER_ENCODER_FILENAME,
        )
        return save_directory

    def _load_pretrained_artifacts(self, pretrained_path: Path) -> None:
        """加载 core/vocoder/speaker 权重并切到 eval / Load weights and set eval mode.

        先重建 io_helper 以指向新目录里的 latent_stats，再分别加载三套权重，重新 tie LLM
        权重，最后把三个子模块置 eval (推理不带 dropout / 不更新 BN 统计)。
        """
        self.latent_stats_path = pretrained_path / self.LATENT_STATS_FILENAME
        self.core.io_helper = type(self.core.io_helper)(
            latent_stats_path=self.latent_stats_path
        )
        self._load_artifact_module(self.core, pretrained_path / self.MODEL_FILENAME)
        self._tie_llm_weights()
        self._load_artifact_module(
            self.vocoder, pretrained_path / self.VOCODER_FILENAME
        )
        self._load_artifact_module(
            self.xvector_extractor,
            pretrained_path / self.SPEAKER_ENCODER_FILENAME,
        )
        self.core.eval()
        self.vocoder.eval()
        self.xvector_extractor.eval()

    def load_pretrained_weights(
        self, pretrained_model_name_or_path: str | Path
    ) -> None:
        """把权重加载进已构造好的实例 / Load weights into an already-built instance.

        与 ``from_pretrained`` 区别: 此处不新建模型，而是要求磁盘上的 config / llm_config 与
        当前实例完全一致 (否则报错)，再加载权重——用于"已知结构、只换权重"的场景。
        """
        pretrained_path = self._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        saved_config = self._load_pretrained_config(pretrained_path)
        if saved_config.to_declared_dict() != self.config.to_declared_dict():
            raise ValueError(
                f"Pretrained config at {pretrained_path} does not match the current model."
            )
        saved_llm_config = self._load_llm_config(
            pretrained_path / self.LLM_CONFIG_FILENAME
        )
        if saved_llm_config.to_dict() != self.core.llm.config.to_dict():
            raise ValueError(
                f"Pretrained LLM config at {pretrained_path} does not match the current model."
            )
        self._load_pretrained_artifacts(pretrained_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path):
        """从目录构造并加载完整模型 / Build and load a model from a pretrained directory.

        HF 风格入口: 读 config/llm_config/tokenizer → ``__init__`` 装配 → 加载三套权重 →
        返回 ``.eval()`` 后的模型。``local_files_only=True`` 强制离线 (不联网拉 tokenizer)。
        """
        logger.info(
            "DotsTtsModel load started: pretrained_path={}",
            pretrained_model_name_or_path,
        )
        pretrained_model_name_or_path = cls._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        config = cls._load_pretrained_config(pretrained_model_name_or_path)
        llm_config = cls._load_llm_config(
            pretrained_model_name_or_path / cls.LLM_CONFIG_FILENAME
        )
        logger.info(
            "DotsTtsModel config loaded: pretrained_path={} sample_rate={} patch_size={}",
            pretrained_model_name_or_path,
            config.vocoder.sample_rate,
            config.patch_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(pretrained_model_name_or_path),
            local_files_only=True,
        )
        model = cls(
            config,
            tokenizer=tokenizer,
            latent_stats_path=pretrained_model_name_or_path / cls.LATENT_STATS_FILENAME,
            llm_config=llm_config,
        )
        model._load_pretrained_artifacts(pretrained_model_name_or_path)
        logger.info(
            "DotsTtsModel load completed: pretrained_path={}",
            pretrained_model_name_or_path,
        )
        return model.eval()

    # endregion Module assembly and checkpoint IO

    # region Training batch preparation
    @torch.no_grad()
    def prepare_training_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        """把原始波形批转成训练所需的 latent / 说话人特征 / Turn waveforms into model inputs.

        在 ``no_grad`` 下用冻结的 vocoder/xvector 抽:
        - ``latents``: AudioVAE 编出的连续 latent (训练目标的来源)，形状约 (B, D, T)。
        - ``latents_sampled``: 从 latent 分布采样得到的连续 latent (供自回归监督)。
        - ``latent_lengths``: 由样本长度 // hop_size 推出的有效帧数。
        - ``xvector``: CAM++ 说话人嵌入 (条件)。
        无波形时 latent 置 None (例如纯文本/无音频 batch)。
        """
        self.vocoder.eval()
        self.xvector_extractor.eval()
        processed = dict(data)
        sample: torch.Tensor | None = data.get("sample")
        sample_lengths: torch.Tensor | None = data.get("sample_lengths")

        if sample is not None:
            latents = self.vocoder.extract_latents(sample)
            processed["latents"] = latents
            if sample_lengths is not None:
                processed["latent_lengths"] = sample_lengths // self.hop_size
            else:
                processed["latent_lengths"] = torch.full(
                    (latents.size(0),),
                    latents.size(-1),
                    dtype=torch.long,
                    device=latents.device,
                )
            processed["latents_sampled"] = self.core.io_helper.sample_from_latent(
                latents
            )
            fbank = data.get("fbank")
            fbank_lengths = data.get("fbank_lengths")
            processed["xvector"] = self.xvector_extractor(
                sample,
                audio_lengths=sample_lengths,
                fbank=fbank,
                fbank_lengths=fbank_lengths,
            )
        else:
            processed["latents"] = None
            processed["latent_lengths"] = None

        return processed

    def _build_audio_span_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """标出哪些位置是 audio span 占位符 / Boolean mask of audio-span placeholder tokens.

        返回与 ``token_ids`` 同形的 bool mask;True 处对应一个待生成的 latent patch (走 FM 头)，
        False 处是普通文本 token (走 LLM 的 CE 损失)。
        """
        span_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in self.core.audio_span_token_ids:
            span_mask = span_mask | (token_ids == token_id)
        return span_mask

    def _prepare_loss_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """从 token 序列推出三路损失的 mask / Derive per-loss masks from the token layout.

        把整条 loss_mask 按是否落在 audio span 上拆成两路:
        - ``ce_loss`` mask: 文本位置 → LLM 交叉熵 (cross-entropy)。
        - ``fm_loss`` mask: audio span 位置 → flow-matching 回归损失;再把这些位置按 batch
          压成 (B, max_patch_count) 的紧凑 patch 级 mask (因为 FM 损失是按 patch 算的)。
        - ``eos_loss`` mask: 由 fm_loss_mask 派生 (见 ``_build_eos_loss_mask``)。
        """
        input_ids: torch.Tensor = data["input_ids"]
        labels: torch.Tensor = data["labels"]
        loss_mask: torch.Tensor = data["loss_mask"]
        input_span_mask = self._build_audio_span_mask(input_ids)
        output_span_mask = self._build_audio_span_mask(labels)
        output_span_mask_float = output_span_mask.to(loss_mask.dtype)
        # 文本损失只在非 audio 位置生效;FM 损失只在 audio span 位置生效。
        llm_loss_mask = loss_mask * (1.0 - output_span_mask_float)
        fm_loss_mask = loss_mask * output_span_mask_float
        patch_counts = output_span_mask.sum(dim=1)
        max_patch_count = max(1, int(patch_counts.max().item()))
        # 把每条样本里散落的 audio-span 位置, 紧凑到前 count 列 → (B, max_patch_count)。
        fm_patch_mask = loss_mask.new_zeros((loss_mask.size(0), max_patch_count))
        for batch_idx in range(loss_mask.size(0)):
            count = int(patch_counts[batch_idx].item())
            if count <= 0:
                continue
            fm_patch_mask[batch_idx, :count] = fm_loss_mask[batch_idx].masked_select(
                output_span_mask[batch_idx]
            )

        return {
            "input_span_mask": input_span_mask,
            "output_span_mask": output_span_mask,
            "loss_masks": {
                "ce_loss": llm_loss_mask,
                "fm_loss": fm_patch_mask,
                "eos_loss": self._build_eos_loss_mask(fm_loss_mask),
            },
        }

    @staticmethod
    def _build_eos_loss_mask(eos_loss_mask: torch.Tensor) -> torch.Tensor:
        """构造 EOS 二分类的加权 mask / Build class-balanced weights for EOS detection.

        EOS 头要在每个 audio patch 上判"是否最后一个 patch"。正样本极少 (每条序列仅 1 个),
        故做类平衡: 最后一个 patch (cumsum == 总数) 记正样本，其余 audio patch 记负样本;
        正样本权重 0.5、负样本权重平摊 ``0.5/负样本数``，使正负对损失贡献均衡。返回 (B, T) 权重。
        """
        batch_size, seq_len = eos_loss_mask.shape
        mask = eos_loss_mask.to(dtype=torch.bool)
        target = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=mask.device)
        mask_counts = mask.sum(dim=1, keepdim=True)
        cumulative = mask.long().cumsum(dim=1)
        # 命中"累计计数 == 该样本 audio patch 总数"的那一处, 即序列最后一个 patch → 正样本。
        target[mask & (cumulative == mask_counts)] = True

        mask_counts_flat = mask_counts.squeeze(1)
        neg_counts = (mask_counts_flat - 1).clamp_min(0).to(eos_loss_mask.dtype)
        pos_weight = torch.where(
            neg_counts > 0,
            torch.full_like(neg_counts, 0.5),
            torch.ones_like(neg_counts),
        ).unsqueeze(1)
        neg_weight = torch.where(
            neg_counts > 0,
            0.5 / neg_counts,
            torch.zeros_like(neg_counts),
        ).unsqueeze(1)

        positive_mask = target & mask
        negative_mask = mask & ~positive_mask
        return torch.where(
            positive_mask,
            pos_weight,
            negative_mask.to(eos_loss_mask.dtype) * neg_weight,
        )
    # endregion Training batch preparation

    # region Training loss assembly and forward
    @staticmethod
    def _compute_ce_loss_term(
        llm_logits: torch.Tensor,
        llm_labels: torch.Tensor,
        llm_loss_mask: torch.Tensor,
    ) -> LossTerm:
        """文本 token 的交叉熵损失项 / Cross-entropy loss term over text tokens.

        ``reduction="none"`` 得到逐 token 损失再 reshape 回 (B, T)，配 mask 由上层做加权归约。
        """
        vocab_size = llm_logits.size(-1)
        ce_loss = F.cross_entropy(
            llm_logits.view(-1, vocab_size),
            llm_labels.view(-1),
            reduction="none",
        ).view_as(llm_labels)
        return LossTerm(loss=ce_loss, mask=llm_loss_mask.to(ce_loss.dtype))

    @staticmethod
    def _compute_fm_loss_term(
        pred: torch.Tensor,
        target: torch.Tensor,
        fm_patch_mask: torch.Tensor,
    ) -> LossTerm:
        """flow-matching 回归损失项 / Flow-matching (velocity) regression loss term.

        ``pred``/``target`` 是声学头预测/真值的 velocity (或位移)，形如 (N_patches, patch_size, D)。
        逐 patch 取 MSE (对 patch_size 与 D 两维求 mean)，再按各 batch 的有效 patch 数 scatter 回
        (B, max_patch_count) 的紧凑布局，对齐 ``_prepare_loss_metadata`` 里压平过的 mask。
        中途用 numel 校验展平后的总 patch 数与 mask 期望一致, 防止上下游对齐错位。
        """
        batch_size, max_patch_count = fm_patch_mask.shape
        fm_loss = (pred - target).pow(2).mean(dim=2).mean(dim=1)
        loss = fm_loss.new_zeros((batch_size, max_patch_count))
        patch_counts = fm_patch_mask.gt(0).sum(dim=1).tolist()
        expected_count = int(sum(patch_counts))
        if expected_count > 0 and int(fm_loss.numel()) != expected_count:
            raise RuntimeError(
                "Flow-matching loss count mismatch: "
                f"expected {expected_count}, got {int(fm_loss.numel())}."
            )

        offset = 0
        for batch_idx, patch_count in enumerate(patch_counts):
            if patch_count <= 0:
                continue
            next_offset = offset + int(patch_count)
            loss[batch_idx, :patch_count] = fm_loss[offset:next_offset]
            offset = next_offset
        return LossTerm(loss=loss, mask=fm_patch_mask.to(loss.dtype))

    @staticmethod
    def _compute_eos_loss_term(
        eos_out: torch.Tensor,
        eos_loss_mask: torch.Tensor,
    ) -> LossTerm:
        """EOS 二分类损失项 / EOS (stop) binary classification loss term.

        ``eos_out`` 是 EOS 头 logits (B, T, 2)。target 同样把"序列最后一个 audio patch"标 1、
        其余标 0。``rearrange`` 到 (B, C, T) 以喂 ``F.cross_entropy``，逐 token 损失再配权重归约。
        """
        batch_size, seq_len, _ = eos_out.shape
        weights = eos_loss_mask.to(device=eos_out.device)
        mask = weights.gt(0)
        target = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.long,
            device=eos_out.device,
        )
        mask_counts = mask.sum(dim=1, keepdim=True)
        cumulative = mask.long().cumsum(dim=1)
        target[mask & (cumulative == mask_counts)] = 1

        logits = rearrange(eos_out, "b n c -> b c n")
        ce_per_token = F.cross_entropy(logits, target, reduction="none")
        return LossTerm(loss=ce_per_token, mask=weights.to(ce_per_token.dtype))

    @staticmethod
    def _compute_eos_loss_stats(
        eos_out: torch.Tensor,
        eos_loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """逐样本 EOS 损失统计 / Per-sample EOS loss sum and presence count.

        给日志/监控用: 返回 (每条样本的加权 EOS 损失之和, 该样本是否含 EOS 监督的 0/1 计数)。
        """
        weights = DotsTtsModel._build_eos_loss_mask(eos_loss_mask)
        term = DotsTtsModel._compute_eos_loss_term(eos_out, weights)
        mask = term.mask.to(device=term.loss.device, dtype=term.loss.dtype)
        eos_loss_sum = (term.loss * mask).sum(dim=1)
        eos_sample_count = eos_loss_mask.to(device=term.loss.device).gt(0).any(
            dim=1
        ).to(term.loss.dtype)
        return eos_loss_sum, eos_sample_count

    def _compute_loss_terms(
        self,
        outputs: DotsTtsForwardOutput,
        *,
        labels: torch.Tensor,
        loss_masks: LossMasks,
    ) -> LossTerms:
        """组装三路损失 / Assemble the three loss terms from a core forward output.

        CE (文本) + flow-matching (声学 latent) + EOS (停止) 三者并行返回, 由训练器加权汇总。
        """
        return {
            "ce_loss": self._compute_ce_loss_term(
                outputs.llm_logits,
                labels,
                loss_masks["ce_loss"],
            ),
            "fm_loss": self._compute_fm_loss_term(
                outputs.pred,
                outputs.target,
                loss_masks["fm_loss"],
            ),
            "eos_loss": self._compute_eos_loss_term(
                outputs.eos_out,
                loss_masks["eos_loss"],
            ),
        }

    def prepare_training_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        """在 DataLoader/collate 侧补齐 loss 元数据 / Attach loss masks to a batch (CPU-side)."""
        prepared = dict(data)
        prepared.update(self._prepare_loss_metadata(prepared))
        return prepared

    def forward(self, data: dict[str, Any]) -> LossTerms:
        """训练前向 / Training forward: inputs → core → three loss terms.

        先抽 latent/说话人特征 (``prepare_training_inputs``)，把 span mask 透传给 core,
        再用 ``self.core`` 一次前向产出 logits / FM pred-target / eos logits, 最后装配损失。
        """
        loss_masks: LossMasks = data["loss_masks"]
        processed = self.prepare_training_inputs(data)
        processed["input_span_mask"] = data["input_span_mask"]
        processed["output_span_mask"] = data["output_span_mask"]
        return self._compute_loss_terms(
            self.core(processed),
            labels=processed["labels"],
            loss_masks=loss_masks,
        )
    # endregion Training loss assembly and forward

    # region Prompt conditioning and decode state helpers
    def _prepare_prompt_audio_for_conditioning(
        self,
        prompt_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        """规范化 prompt 波形并算缓存键 / Pad prompt audio to patch boundary, return cache key.

        把波形长度向上补零到 ``patch_size * hop_size`` 的整数倍 (保证能整切成 patch)，再对最终
        张量算 SHA1 摘要作为特征缓存键 (形状+dtype+原始字节)。返回 (规范化波形 (1, T_pad), 摘要)。
        """
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        prompt_audio = prompt_audio.detach().cpu().contiguous()

        # 每个 patch 覆盖的波形采样点数 = patch_size 个 latent 帧 × 每帧 hop_size 个采样点。
        samples_per_patch = self.config.patch_size * self.hop_size
        target_len = (
            math.ceil(prompt_audio.size(1) / samples_per_patch) * samples_per_patch
        )
        pad_len = target_len - prompt_audio.size(1)
        if pad_len > 0:
            prompt_audio = F.pad(prompt_audio, (0, pad_len))

        digest = hashlib.sha1()
        digest.update(str(tuple(prompt_audio.shape)).encode("ascii"))
        digest.update(str(prompt_audio.dtype).encode("ascii"))
        digest.update(prompt_audio.numpy().tobytes())
        return prompt_audio, digest.hexdigest()

    def _get_prompt_feature_cache_entry(
        self,
        cache_key: str,
    ) -> _PromptFeatureCacheEntry | None:
        """查 prompt 特征缓存并刷新 LRU 顺序 / LRU-get a prompt feature cache entry."""
        entry = self._prompt_feature_cache.get(cache_key)
        if entry is not None:
            self._prompt_feature_cache.move_to_end(cache_key)  # 命中即移到队尾 (最近使用)
        return entry

    def _store_prompt_feature_cache_entry(
        self,
        cache_key: str,
        entry: _PromptFeatureCacheEntry,
    ) -> None:
        """写入 prompt 特征缓存并按容量淘汰 / LRU-put, evicting oldest beyond capacity."""
        if (
            entry.speaker_embedding is None
            and entry.prompt_latent_distribution is None
        ):
            return  # 空条目不入缓存
        self._prompt_feature_cache[cache_key] = entry
        self._prompt_feature_cache.move_to_end(cache_key)
        while (
            len(self._prompt_feature_cache) > self._PROMPT_FEATURE_CACHE_MAX_ENTRIES
        ):
            self._prompt_feature_cache.popitem(last=False)  # 淘汰最久未用项 (队首)

    def _can_cache_speaker_embedding(self, prompt_sample_count: int) -> bool:
        """判断 x-vector 是否可缓存 / Whether the speaker embedding is reuse-safe.

        x-vector 提取器对超过 ``max_audio_seconds`` 的输入会截断，截断后的嵌入与不同长度的
        prompt 不可互换, 故只有在输入未超限时才允许缓存。``<=0`` 表示无限制, 恒可缓存。
        """
        max_audio_seconds = self.xvector_extractor.max_audio_seconds
        if max_audio_seconds <= 0:
            return True
        max_input_length = round(
            self.xvector_extractor.sample_rate * max_audio_seconds
        )
        return int(prompt_sample_count) <= max_input_length

    @torch.no_grad()
    def _prepare_prompt_conditioning(
        self,
        prompt_audio: torch.Tensor | None,
        *,
        use_prompt_prefill: bool,
        speaker_scale: float = 1.5,
    ) -> _PromptConditioning:
        """从参考音频派生条件 / Derive speaker + prompt-latent conditioning from prompt audio.

        两种模式 (对应两套 schema):
        - ``use_prompt_prefill=False`` (仅参考音色): 只抽 x-vector → ``g_cond``，不喂 prompt 内容。
        - ``use_prompt_prefill=True`` (语音克隆/续写): 额外用 AudioVAE 抽 prompt latent，切成
          ``prompt_patches`` 并保留 ``prompt_latents``，prefill 阶段把参考内容喂回模型。

        ``speaker_scale`` 放大 x-vector 后再投影, 经验上能增强音色一致性。重特征 (x-vector /
        prompt latent) 走 LRU 缓存复用。注意末尾会去掉一个 patch (``[:, :-patch_size]``):
        最后一段留给模型自己续生成, 避免直接复制 prompt 尾巴。
        """
        if prompt_audio is None:
            logger.info("Prompt conditioning skipped: no prompt audio provided.")
            return _PromptConditioning()

        self.vocoder.eval()
        self.xvector_extractor.eval()
        device = next(self.core.parameters()).device
        prompt_audio, cache_key = self._prepare_prompt_audio_for_conditioning(
            prompt_audio
        )
        prompt_sample_count = int(prompt_audio.shape[-1])
        cache_entry = self._get_prompt_feature_cache_entry(cache_key)
        if cache_entry is None:
            cache_entry = _PromptFeatureCacheEntry()
        prompt_audio = prompt_audio.to(device=device)

        can_cache_speaker = self._can_cache_speaker_embedding(prompt_sample_count)
        speaker_embedding = (
            cache_entry.speaker_embedding if can_cache_speaker else None
        )
        if speaker_embedding is None:
            speaker_encoder = self._get_compiled_model(
                "speaker_encoder",
                self.xvector_extractor,
            )
            with measure_inference("speaker_encoder"):
                speaker_embedding = speaker_encoder(prompt_audio[None, :])
            if can_cache_speaker:
                cache_entry.speaker_embedding = speaker_embedding.detach()
        else:
            logger.info(
                "Prompt speaker cache hit: key={} prompt_samples={}",
                cache_key[:12],
                prompt_sample_count,
            )
        # 放大 x-vector 后投影成全局条件 g_cond (B, fm_hidden_size), 贯穿整个声学头解码。
        g_cond = self.core.xvec_proj(speaker_embedding * float(speaker_scale))
        if not use_prompt_prefill:
            self._store_prompt_feature_cache_entry(cache_key, cache_entry)
            logger.info(
                "Reference-audio-only conditioning prepared: prompt_samples={} speaker_scale={} device={}",
                prompt_sample_count,
                speaker_scale,
                device,
            )
            return _PromptConditioning(g_cond=g_cond)

        prompt_latents = cache_entry.prompt_latent_distribution
        if prompt_latents is None:
            latent_encoder = self._get_compiled_model(
                "latent_encoder",
                self.vocoder.extract_latents,
            )
            with measure_inference("latent_encoder"):
                prompt_latents = latent_encoder(prompt_audio[None, :])
            cache_entry.prompt_latent_distribution = prompt_latents.detach()
        else:
            logger.info(
                "Prompt latent cache hit: key={} prompt_samples={}",
                cache_key[:12],
                prompt_sample_count,
            )
        self._store_prompt_feature_cache_entry(cache_key, cache_entry)
        prompt_latents_sampled = self.core.io_helper.sample_from_latent(prompt_latents)
        # 砍掉最后一个 patch: 该尾段留给模型续生成, 避免照搬 prompt 结尾。
        prompt_latents_sampled = prompt_latents_sampled[:, : -self.config.patch_size]
        # 归一化后按 patch_size 切成 (B, S, patch_size, D), 作为 prefill 的 prompt 内容嵌入。
        prompt_patches = rearrange(
            self.core.io_helper.normalize(prompt_latents_sampled),
            "b (s p) d -> b s p d",
            p=self.config.patch_size,
        )
        logger.info(
            "Prompt conditioning prepared: prompt_samples={} prompt_patch_count={} "
            "speaker_scale={} device={}",
            prompt_sample_count,
            prompt_patches.size(1),
            speaker_scale,
            device,
        )
        return _PromptConditioning(
            prompt_patches=prompt_patches,
            prompt_latents=prompt_latents_sampled,
            g_cond=g_cond,
        )

    @staticmethod
    def _patch_encoder_compile_signature(
        patch_encoder_state: Any,
    ) -> tuple[int, torch.dtype]:
        """patch encoder 编译签名 / Compile signature: (kv-cache capacity, dtype).

        以第 0 层 KV cache 的容量 (dim=2 是 seq 维) 与 dtype 作为编译 key, 同容量/同精度共享产物。
        """
        key_cache, _ = patch_encoder_state.layer_caches[0]
        return int(key_cache.size(2)), key_cache.dtype

    def _resolve_patch_encoder_audio_bucket(self, required_seq_len: int) -> int:
        """把 patch encoder 所需序列长换算成 bucket patch 数 / Bucketize encoder capacity.

        encoder 内部以 ``out_ds_rate`` 下采样, 故先把所需 token 数换成 patch 数再取 bucket。
        """
        requested = int(required_seq_len)
        if requested <= 0:
            raise ValueError("required_seq_len must be positive.")
        requested_patch_count = math.ceil(
            requested / self.core.patch_encoder.out_ds_rate
        )
        if not self._optimize_enabled:
            return requested_patch_count
        return self._resolve_generate_length_bucket(requested_patch_count).size

    def _copy_patch_encoder_state(self, source: Any, target: Any) -> None:
        """把 patch encoder 流式状态搬到更大容量的新状态 / Copy encoder state into a larger one.

        扩容时 (序列变长超过当前 bucket) 需要新建更大的 KV cache, 这里把已有 ``seq_len`` 范围内的
        conv_tail 与每层 K/V 原样拷过去, 保证流式连续性。仅拷有效前缀, 未填部分留零。
        """
        seq_len = source.seq_len
        target_capacity = int(target.layer_caches[0][0].size(2))
        if seq_len > target_capacity:
            raise ValueError(
                "Patch encoder state copy exceeds target capacity: "
                f"seq_len={seq_len} capacity={target_capacity}."
            )

        target.conv_tail.copy_(source.conv_tail)
        target.seq_len = seq_len
        for (source_key, source_value), (target_key, target_value) in zip(
            source.layer_caches,
            target.layer_caches,
            strict=True,
        ):
            if seq_len > 0:
                target_key[:, :, :seq_len, :].copy_(source_key[:, :, :seq_len, :])
                target_value[:, :, :seq_len, :].copy_(source_value[:, :, :seq_len, :])

    def _ensure_patch_encoder_state_capacity(
        self,
        state: _GenerateState,
        *,
        required_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """确保 patch encoder 状态容量足够, 不够则扩容并迁移 / Grow encoder state on demand.

        当前容量够就直接返回;否则按 bucket 新建更大状态并把旧状态拷过去 (见 ``_copy_...``)。
        """
        current_state = state.patch_encoder_state
        if current_state is not None:
            current_capacity = int(current_state.layer_caches[0][0].size(2))
            if current_capacity >= required_seq_len:
                return

        target_audio_patch_count = self._resolve_patch_encoder_audio_bucket(
            required_seq_len
        )
        next_state = self.core.patch_encoder.init_decode_state(
            max_audio_patch_count=target_audio_patch_count,
            batch_size=1,
            device=device,
            dtype=dtype,
        )
        if current_state is not None:
            self._copy_patch_encoder_state(current_state, next_state)
        state.patch_encoder_state = next_state

    def _prefill_prompt_latents(
        self,
        prompt_latents: torch.Tensor | None,
        *,
        state: _GenerateState,
    ) -> torch.Tensor | None:
        """把 prompt latent 编成 LLM 输入嵌入 / Prefill prompt latents into LLM-space embeddings.

        对语音克隆的 prefill: 先确保 patch encoder 状态容量足够, 再调用其 ``prefill`` 把 prompt
        latent (B, T, D) 编成可塞回 LLM 序列的 patch 嵌入 (B, S, D_llm)。空 prompt 返回零长张量。
        """
        if prompt_latents is None:
            return None
        if prompt_latents.size(1) == 0:
            return prompt_latents.new_zeros(
                (prompt_latents.size(0), 0, self.core.llm_hidden_size)
            )
        self._ensure_patch_encoder_state_capacity(
            state,
            required_seq_len=(
                (prompt_latents.size(1) // self.core.patch_encoder.patch_size)
                * self.core.patch_encoder.out_ds_rate
            ),
            device=prompt_latents.device,
            dtype=(
                state.fm_sequence.dtype
                if state.fm_sequence is not None
                else prompt_latents.dtype
            ),
        )
        with measure_inference("patch_encoder"):
            prompt_patch_embeddings, state.patch_encoder_state = (
                self.core.patch_encoder.prefill(
                    prompt_latents,
                    state.patch_encoder_state,
                )
            )
        return prompt_patch_embeddings

    def _get_fm_decode_workspace(
        self,
        *,
        total_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """取/建单步 FM 解码工作区 / Get a reusable single-step FM decode workspace.

        预分配 4 个静态张量并按 (total_len, device, dtype) 缓存复用:
        - ``input_sequence`` / ``cfg_sequence``: 条件分支 / CFG 无条件分支的输入, (1, total_len, fm_hidden)。
        - ``attn_mask``: (1, total_len, total_len) 的注意力 mask (本步现填)。
        - ``pos_ids``: (1, total_len) 的位置编号 (本步现填)。
        复用是为让 ``torch.compile`` 看到稳定的输入地址/形状 (CUDA graph 友好)。
        """
        workspace_key = (total_len, str(device), dtype)
        workspace = self._fm_decode_workspaces.get(workspace_key)
        if workspace is None:
            workspace = {
                "input_sequence": torch.zeros(
                    (1, total_len, self.core.fm_hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "cfg_sequence": torch.zeros(
                    (1, total_len, self.core.fm_hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "attn_mask": torch.zeros(
                    (1, total_len, total_len),
                    dtype=torch.bool,
                    device=device,
                ),
                "pos_ids": torch.zeros(
                    (1, total_len),
                    dtype=torch.float32,
                    device=device,
                ),
            }
            self._fm_decode_workspaces[workspace_key] = workspace
        else:
            workspace["input_sequence"].zero_()
            workspace["cfg_sequence"].zero_()
        return workspace

    def _resolve_fm_history_bucket_capacity(self, fm_seq_len: int) -> int:
        """把当前 FM 历史长度取整到 bucket 容量 / Bucketize the FM history length.

        历史每个 patch 占 ``hidden_patch_size + latent_patch_size`` 个槽 (history_stride)。先换成
        patch 数取 bucket, 再乘回 stride, 得到对齐 bucket 的历史容量, 使编译签名只在档位间变化。
        """
        requested = int(fm_seq_len)
        if requested <= 0:
            raise ValueError("fm_seq_len must be positive.")
        if not self._optimize_enabled:
            return requested
        history_stride = self.core.hidden_patch_size + self.core.latent_patch_size
        requested_patch_count = math.ceil(requested / history_stride)
        return self._resolve_generate_length_bucket(
            requested_patch_count
        ).size * history_stride

    def _build_fm_attn_mask(
        self,
        *,
        state: _GenerateState,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """构造声学头单步解码的注意力 mask / Build the attention mask for one FM decode step.

        声学头的输入序列 = [历史 (前 fm_seq_len 个槽)] + [padding] + [当前待解码 latent patch (末尾
        latent_patch_size 个槽)]。这里就地把 (1, L, L) 的 bool mask (True=可见) 填成:
        - 历史的"过去块" (block_start 之前) 走标准 causal mask (下三角可见);
        - 历史的"当前块"行 (最近一个 hidden patch) 可看全部历史 + 末尾 latent;
        - 末尾 latent 行 (要预测的 patch) 可看全部历史 + 自身块;
        - 中间 padding 位置只对角线自连 (防止 NaN, 不影响有效计算)。
        采用就地 (in-place) 填充以复用静态 buffer。
        """
        if state.fm_seq_len <= 0:
            raise RuntimeError("FM sequence length must be positive before decode.")
        hidden_patch_size = self.core.hidden_patch_size
        latent_start = attn_mask.size(-1) - self.core.latent_patch_size  # 末尾 latent patch 起点
        attn_mask.zero_()
        block_start = state.fm_seq_len - hidden_patch_size  # 最近一个 hidden patch 块的起点
        if block_start > 0:
            # 过去块: 标准 causal——triu(1) 取严格上三角再取反, 得到下三角 (含对角线) 可见。
            causal_mask = torch.ones(
                (block_start, block_start),
                device=attn_mask.device,
                dtype=torch.bool,
            ).triu(1).logical_not()
            attn_mask[:, :block_start, :block_start] = causal_mask

        # 当前历史块 (最近 hidden patch) 既能看全部历史, 也能看末尾待解码 latent。
        attn_mask[:, block_start : state.fm_seq_len, : state.fm_seq_len] = True
        attn_mask[:, block_start : state.fm_seq_len, latent_start:] = True
        # 末尾待解码 latent patch: 看全部历史 + 自身块 (块内双向 / full attention)。
        attn_mask[:, latent_start:, : state.fm_seq_len] = True
        attn_mask[:, latent_start:, latent_start:] = True
        if latent_start > state.fm_seq_len:
            # 历史与末尾 latent 之间的 padding 槽: 仅对角线自连, 避免某行全 mask 导致 softmax NaN。
            padding_indices = torch.arange(
                state.fm_seq_len,
                latent_start,
                device=attn_mask.device,
            )
            attn_mask[:, padding_indices, padding_indices] = True
        return attn_mask

    def _build_fm_pos_ids(
        self,
        *,
        state: _GenerateState,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        """构造声学头单步的位置编号 / Build position ids for one FM decode step.

        历史占 ``[0, fm_seq_len)``;末尾待解码 latent patch 紧接历史, 占
        ``[fm_seq_len, fm_seq_len + latent_patch_size)``。中间 padding 槽位置保持 0 (被 mask 掉,
        位置值无意义)。这样 RoPE/位置编码对"末尾 latent 接在历史之后"是连续正确的。
        """
        if state.fm_seq_len <= 0:
            raise RuntimeError("FM sequence length must be positive before decode.")
        pos_ids.zero_()
        latent_start = pos_ids.size(-1) - self.core.latent_patch_size
        pos_ids[:, : state.fm_seq_len] = torch.arange(
            state.fm_seq_len,
            device=pos_ids.device,
            dtype=pos_ids.dtype,
        )
        # 末尾 latent 的位置紧接历史之后, 而非从其物理 latent_start 处编号。
        pos_ids[:, latent_start:] = torch.arange(
            state.fm_seq_len,
            state.fm_seq_len + self.core.latent_patch_size,
            device=pos_ids.device,
            dtype=pos_ids.dtype,
        )
        return pos_ids

    def _prepare_fm_decode_inputs(
        self,
        state: _GenerateState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """把历史拷进按 bucket 对齐的工作区 / Stage history into a bucket-sized workspace.

        总长 = bucket 对齐后的历史容量 + 一个待解码 latent patch。从静态历史 buffer 里把有效前缀
        (前 fm_seq_len 个槽) 拷到工作区, 返回 (输入序列, cfg 序列, attn_mask, pos_ids, 历史容量)。
        """
        sequence = state.fm_sequence
        cfg_sequence = state.fm_cfg_sequence
        if sequence is None or cfg_sequence is None:
            raise RuntimeError("FM static buffers are not initialized.")
        history_bucket_capacity = self._resolve_fm_history_bucket_capacity(
            state.fm_seq_len
        )
        total_len = history_bucket_capacity + self.core.latent_patch_size
        workspace = self._get_fm_decode_workspace(
            total_len=total_len,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        workspace["input_sequence"][:, : state.fm_seq_len].copy_(
            sequence[:, : state.fm_seq_len]
        )
        workspace["cfg_sequence"][:, : state.fm_seq_len].copy_(
            cfg_sequence[:, : state.fm_seq_len]
        )
        return (
            workspace["input_sequence"],
            workspace["cfg_sequence"],
            workspace["attn_mask"],
            workspace["pos_ids"],
            history_bucket_capacity,
        )

    def _append_to_fm_buffer(
        self,
        buffer: torch.Tensor | None,
        state: _GenerateState,
        chunk: torch.Tensor,
    ) -> tuple[int, int]:
        """把一个 chunk 写入 FM 静态历史 buffer 的尾部 / Append a chunk to the FM history buffer.

        返回写入的 ``[start, end)`` 区间;超出预分配容量即报错 (说明 bucket 选小了)。注意: 这里**不**
        推进 ``state.fm_seq_len``——由调用方在写完 cfg buffer 后统一推进, 保证两条序列长度同步。
        """
        if buffer is None:
            raise RuntimeError("FM static buffer is not initialized.")
        start = state.fm_seq_len
        end = start + chunk.size(1)
        if end > state.fm_capacity:
            raise RuntimeError(
                "FM StaticBuffer capacity exceeded: "
                f"next_length={end} capacity={state.fm_capacity}."
            )
        buffer[:, start:end].copy_(chunk.to(buffer.dtype))
        return start, end

    def _append_hidden_chunk(
        self, state: _GenerateState, hidden_chunk: torch.Tensor
    ) -> None:
        """把一段 LLM hidden 投影后追加进 FM 历史 / Append projected LLM hidden into FM history.

        取最近 ``hidden_patch_size`` 个 hidden, 经 ``hidden_proj`` 投到 FM 空间作为条件分支;
        同时把"全零 hidden"投影后写入 cfg 分支——这就是 classifier-free guidance (CFG) 的无条件
        分支: 推理时它见不到真实文本条件, 只见说话人 g_cond, 从而 guidance 能放大条件贡献。
        """
        last_hidden = hidden_chunk[:, -self.core.hidden_patch_size :, :]
        projected = self.core.hidden_proj(last_hidden)
        # CFG 无条件分支: 用零输入过同一投影, 作为 cfg 序列里对应位置的内容。
        null_projected = self.core.hidden_proj(torch.zeros_like(last_hidden))
        _start, end = self._append_to_fm_buffer(
            state.fm_sequence,
            state,
            projected,
        )
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len : end].copy_(null_projected.to(cfg_buffer.dtype))
        state.fm_seq_len = end  # 两条序列同步推进

    def _append_history_chunk(
        self, state: _GenerateState, latent_chunk: torch.Tensor
    ) -> None:
        """把一个 latent patch 投影后追加进 FM 历史 / Append a projected latent patch to history.

        与 ``_append_hidden_chunk`` 不同: latent 是声学内容本身, 两条分支 (条件 / cfg 无条件) 写
        **相同**的 history_latent——CFG 只在文本/语义条件上做有/无之分, 已生成的声学历史对两支都可见。
        """
        history_latent = self.core.latent_proj(latent_chunk)
        _start, end = self._append_to_fm_buffer(
            state.fm_sequence,
            state,
            history_latent,
        )
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        # 两支共享同一份声学历史 (latent), 只有 hidden/文本条件才区分条件与无条件。
        cfg_buffer[:, state.fm_seq_len : end].copy_(history_latent.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    def _consume_text_schedule(
        self,
        generation_schedule: torch.Tensor,
        *,
        position: int,
        next_audio_position: int,
        state: _GenerateState,
    ) -> int:
        """消费一段纯文本 token / Run LLM over a text run up to the next audio span.

        把 ``[position, next_audio_position)`` 的文本一次喂进 LLM (增量更新 KV cache), 取末端
        hidden 投影后追加进 FM 历史 (作为接下来要生成的 audio patch 的语义条件)。返回新游标。
        """
        with measure_inference("LLM"):
            text_chunk = generation_schedule[:, position:next_audio_position]
            _, state.llm_hiddens, _, state.llm_cache = self.core.step_llm(
                input_ids=text_chunk,
                past_key_values=state.llm_cache,
            )
        self._append_hidden_chunk(state, state.llm_hiddens)
        return next_audio_position

    def _locate_prefill_boundary(
        self,
        *,
        span_positions: torch.Tensor,
        prompt_patch_count: int,
    ) -> tuple[int, torch.Tensor]:
        """定位 prefill 边界 / Locate where prompt prefill ends and free decode begins.

        前 ``prompt_patch_count`` 个 audio span 属于 prompt (要喂参考内容);第
        ``prompt_patch_count`` 个 span 的位置就是 prefill 截止点 (从这里开始自由生成)。
        返回 (prefill_end 位置, 属于 prompt 的那些 span 位置)。
        """
        if span_positions.numel() > prompt_patch_count:
            return int(span_positions[prompt_patch_count].item()), span_positions[
                :prompt_patch_count
            ]
        raise RuntimeError(
            "Prefill boundary discovery failed despite prior schedule validation."
        )

    @staticmethod
    def _find_audio_span_positions(
        generation_schedule: torch.Tensor,
        *,
        audio_placeholder_ids: set[int],
    ) -> torch.Tensor:
        """找出所有 audio span 占位符的位置 / Indices of all audio-span placeholder tokens.

        用 ``torch.isin`` 在 schedule 上匹配占位符 id 集合, 返回升序的位置一维张量。每个位置对应
        一个待生成的 latent patch。
        """
        schedule = generation_schedule[0]
        placeholder_ids = torch.tensor(
            sorted(audio_placeholder_ids),
            device=schedule.device,
            dtype=schedule.dtype,
        )
        return torch.nonzero(
            torch.isin(schedule, placeholder_ids),
            as_tuple=False,
        ).squeeze(-1)

    @staticmethod
    def _next_token_is_audio_span(
        generation_schedule: torch.Tensor,
        *,
        position: int,
        audio_placeholder_ids: set[int],
    ) -> bool:
        """下一个 token 是否仍是 audio span / Is the next scheduled token another audio span?

        用于决定连续 audio patch 之间是否需要再追加一段 hidden 条件 (相邻 audio 直接续接时, 当前
        patch 的 LLM hidden 也要作为下一个 patch 的条件)。越界 (序列尾) 返回 False。
        """
        next_position = position + 1
        if next_position >= generation_schedule.size(1):
            return False
        return int(generation_schedule[0, next_position].item()) in audio_placeholder_ids

    def _build_prefill_inputs_embeds(
        self,
        generation_schedule: torch.Tensor,
        *,
        prompt_patch_embeddings: torch.Tensor | None,
        prompt_span_positions: torch.Tensor,
    ) -> torch.Tensor:
        """构造 prefill 的输入嵌入 / Build input embeddings for the prefill chunk.

        文本 token 走 LLM 词嵌入;但 prompt 的 audio span 占位符不能用词嵌入——要替换成 patch
        encoder 编出的 prompt latent 嵌入 (``prompt_patch_embeddings``), 这样模型才"听见"参考音频
        的真实内容。``.clone()`` 是因为后面要就地散写, 避免改到共享的嵌入张量。
        """
        inputs_embeds = self.core.llm.get_input_embeddings()(
            generation_schedule
        ).clone()
        if prompt_span_positions.numel() > 0:
            if prompt_patch_embeddings is None:
                raise RuntimeError(
                    "Prompt patch embeddings are required when prefill includes prompt audio spans."
                )
            patch_embeddings = prompt_patch_embeddings[
                :, : prompt_span_positions.numel()
            ].to(inputs_embeds.dtype)
            if patch_embeddings.size(1) != prompt_span_positions.numel():
                raise RuntimeError(
                    f"Prompt patch embeddings ({patch_embeddings.size(1)}) do not match prompt span count ({prompt_span_positions.numel()})."
                )
            inputs_embeds[:, prompt_span_positions, :] = patch_embeddings
        return inputs_embeds

    def _prefill(
        self,
        generation_schedule: torch.Tensor,
        *,
        state: _GenerateState,
        span_positions: torch.Tensor,
        prompt_patches: torch.Tensor | None,
        prompt_patch_embeddings: torch.Tensor | None,
        audio_placeholder_ids: set[int],
    ) -> int:
        """一次性预填 prompt 区间 / Prefill the prompt region into LLM + FM history.

        【重点·难点】把 ``[0, prefill_end)`` 的文本 + prompt 音频 span 一次喂进 LLM, 并同步把对应的
        条件/历史写进 FM 静态 buffer, 然后返回自由生成的起始位置 ``prefill_end``。

        prefill_end 由 ``_locate_prefill_boundary`` 给出 (第 ``prompt_patch_count`` 个 audio span 处)。
        逐个 prompt span 处理时, 交错地往 FM 历史里追加:
        1) span 之前若有新文本 (span_position > cursor): 追加该段末端 LLM hidden 作语义条件;
        2) ``_append_history_chunk``: 追加这个 prompt latent patch 本身 (声学历史);
        3) 若紧接着还是 audio span: 再补一段该位置的 hidden 条件 (相邻 audio 之间需要 hidden 桥接);
        循环后若 prefill_end 之后还有残余文本, 补最后一段 hidden。``llm_hiddens`` 取末位供后续 EOS 判定。
        """
        prompt_patch_count = (
            0 if prompt_patches is None else int(prompt_patches.size(1))
        )
        prefill_end, prompt_span_positions = self._locate_prefill_boundary(
            span_positions=span_positions,
            prompt_patch_count=prompt_patch_count,
        )
        if prefill_end == 0:
            return 0  # schedule 一上来就是 decode span, 无需 prefill
        inputs_embeds = self._build_prefill_inputs_embeds(
            generation_schedule[:, :prefill_end],
            prompt_patch_embeddings=prompt_patch_embeddings,
            prompt_span_positions=prompt_span_positions,
        )
        with measure_inference("LLM"):
            _, llm_hiddens, _, state.llm_cache = self.core.step_llm(
                inputs_embeds=inputs_embeds,
                past_key_values=state.llm_cache,
            )
        state.llm_hiddens = llm_hiddens[:, -1:, :]  # 末位 hidden, 留作 EOS 判定与下一步条件

        cursor = 0  # 已处理到的 schedule 位置 (用于切出 span 之间的文本段)
        for prompt_index, span_position in enumerate(prompt_span_positions.tolist()):
            if span_position > cursor:
                # span 前有文本: 用紧邻该 span 的前一位 hidden 作语义条件。
                self._append_hidden_chunk(
                    state, llm_hiddens[:, span_position - 1 : span_position, :]
                )
            # 追加这个 prompt latent patch 的声学历史 (让模型"听过"参考内容)。
            self._append_history_chunk(state, prompt_patches[:, prompt_index])
            if self._next_token_is_audio_span(
                generation_schedule,
                position=span_position,
                audio_placeholder_ids=audio_placeholder_ids,
            ):
                # 下一位还是 audio: 当前位置 hidden 也要作为下个 patch 的条件 (hidden 桥接)。
                self._append_hidden_chunk(
                    state, llm_hiddens[:, span_position : span_position + 1, :]
                )
            cursor = span_position + 1
        if prefill_end > cursor:
            # prefill 末尾仍有残余文本 (最后一个 prompt span 之后到 prefill_end): 补一段 hidden。
            self._append_hidden_chunk(
                state, llm_hiddens[:, prefill_end - 1 : prefill_end, :]
            )
        return prefill_end

    def _decode_next_audio(
        self,
        state: _GenerateState,
        *,
        device: torch.device,
        g_cond: torch.Tensor | None,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """解出一个 latent patch / Solve one latent patch via the flow-matching head.

        【重点·难点】声学头单步: 给定到目前为止的历史 (hidden 条件 + 已生成 latent), 在连续潜在空间里
        从高斯噪声出发, 沿 velocity field 用 ODE solver 积分出**一个** latent patch (B, patch_size, D)。

        按 ``self.core.mode`` 分派两种求解器:
        - ``meanflow``: MeanFlow 蒸馏模型, 用极少 NFE (函数评估次数) 直接跳到终点, 走
          ``meanflow_solver_step`` + ``_meanflow_step_fm``;
        - 否则: 标准 flow-matching, 走 ``fm_solver_step`` + ``_flow_matching_step_fm``, 内部按
          ``ode_method`` (euler/...) 迭代 ``num_steps`` 步, 并用 ``guidance_scale`` 做 CFG。

        编译签名: 优化开启时按 bucket 容量 (history_bucket_capacity) 取, 关闭时按真实 fm_seq_len;
        据此 ``torch.compile`` 同一档位共享产物。attn_mask / pos_ids 每步现填进静态工作区。
        """
        if state.fm_seq_len <= 0:
            raise RuntimeError(
                "Cannot decode audio before any conditioning state has been prefetched."
            )
        if state.fm_sequence is None or state.fm_cfg_sequence is None:
            raise RuntimeError("FM static buffers are not initialized.")
        if state.fm_null_g_cond is None:
            raise RuntimeError("FM null conditioning buffer is not initialized.")
        fm_sequence, fm_cfg_sequence, fm_attn_mask, fm_pos_ids, history_bucket_capacity = (
            self._prepare_fm_decode_inputs(state)
        )
        # 优化开启 → 用 bucket 容量做签名 (跨步稳定, 同档复用编译产物);关闭 → 用真实长度。
        compile_signature = (
            (history_bucket_capacity, state.fm_sequence.dtype)
            if self._optimize_enabled
            else (state.fm_seq_len, state.fm_sequence.dtype)
        )
        if g_cond is None:
            g_cond = state.fm_null_g_cond  # 无说话人条件: 用全零兜底
        else:
            g_cond = g_cond.to(
                device=state.fm_null_g_cond.device,
                dtype=state.fm_null_g_cond.dtype,
            )
        with measure_inference("FM"):
            attn_mask = self._build_fm_attn_mask(
                state=state,
                attn_mask=fm_attn_mask,
            )
            pos_ids = self._build_fm_pos_ids(
                state=state,
                pos_ids=fm_pos_ids,
            )
            # 分派 1: MeanFlow 蒸馏求解器 (少步、无显式 CFG 双分支)。
            if self.core.mode == "meanflow":
                fm_solver_step = self._get_compiled_method(
                    "FM.meanflow.solver_step",
                    self.core,
                    "meanflow_solver_step",
                    signature=compile_signature,
                )
                return self.core._meanflow_step_fm(
                    input_sequence=fm_sequence,
                    attn_mask=attn_mask,
                    pos_ids=pos_ids,
                    patch_size=self.core.latent_patch_size,
                    g_cond=g_cond,
                    nfe=num_steps,
                    solver_step=fm_solver_step,
                )

            # 分派 2: 标准 flow-matching, 沿 velocity field 用 ODE solver 迭代 num_steps 步 (含 CFG)。
            fm_solver_step = self._get_compiled_method(
                "FM.flow_matching.solver_step",
                self.core,
                "fm_solver_step",
                signature=compile_signature,
            )
            return self.core._flow_matching_step_fm(
                input_sequence=fm_sequence,
                cfg_sequence=fm_cfg_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                hidden_size=self.core.hidden_patch_size,
                patch_size=self.core.latent_patch_size,
                g_cond=g_cond,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                solver_step=fm_solver_step,
            )

    def _consume_audio_patch(
        self,
        state: _GenerateState,
        *,
        audio_patch: torch.Tensor,
    ) -> None:
        """把刚生成的 latent patch 反馈回模型 / Feed a freshly decoded latent patch back in.

        【重点·难点】patch 状态机的一步, 把 ``_decode_next_audio`` 产出的归一化 latent patch 同时写进
        两个回路:
        1) FM 历史: ``_append_history_chunk`` (供下一个 patch 的声学头看到已生成内容);
        2) LLM 回路: 先 ``denormalize`` 回原始尺度 → patch encoder ``decode_patch`` 把它编成一个 LLM
           输入嵌入 → ``step_llm`` 推进自回归主干 (更新 KV cache 与 ``llm_hiddens``, 后者供 EOS 判定)。
        patch encoder 是流式的, 需先确保其 KV cache 容量够 (``_ensure_..._capacity``), 解码后把更新后的
        conv_tail 写回、并按 ``out_ds_rate`` 推进其 seq_len。
        """
        audio_patch_for_llm = self.core.io_helper.denormalize(audio_patch)
        self._append_history_chunk(state, audio_patch)
        current_seq_len = (
            0
            if state.patch_encoder_state is None
            else state.patch_encoder_state.seq_len
        )
        self._ensure_patch_encoder_state_capacity(
            state,
            required_seq_len=current_seq_len + self.core.patch_encoder.out_ds_rate,
            device=audio_patch_for_llm.device,
            dtype=(
                state.fm_sequence.dtype
                if state.fm_sequence is not None
                else audio_patch_for_llm.dtype
            ),
        )
        patch_encoder_decode = self._get_compiled_model(
            "patch_encoder.decode_patch",
            self.core.patch_encoder.decode_patch,
            signature=self._patch_encoder_compile_signature(state.patch_encoder_state),
        )
        # 本 patch 在 encoder 序列里的绝对位置 = 已有 seq_len + 本 patch 的 out_ds_rate 个 token。
        patch_positions = (
            torch.arange(
                self.core.patch_encoder.out_ds_rate,
                device=audio_patch_for_llm.device,
                dtype=torch.long,
            )
            + state.patch_encoder_state.seq_len
        )
        with measure_inference("patch_encoder"):
            llm_embedding, conv_tail = patch_encoder_decode(
                audio_patch_for_llm,
                state.patch_encoder_state.conv_tail,
                state.patch_encoder_state.layer_caches,
                patch_positions,
            )
        state.patch_encoder_state.conv_tail.copy_(conv_tail)
        state.patch_encoder_state.seq_len += self.core.patch_encoder.out_ds_rate
        with measure_inference("LLM"):
            _, state.llm_hiddens, _, state.llm_cache = self.core.step_llm(
                inputs_embeds=llm_embedding,
                past_key_values=state.llm_cache,
            )

    def _decode(
        self,
        generation_schedule: torch.Tensor,
        *,
        position: int,
        state: _GenerateState,
        audio_placeholder_ids: set[int],
        span_positions: torch.Tensor,
        device: torch.device,
        g_cond: torch.Tensor | None,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        eos_threshold: float,
    ) -> Iterator[torch.Tensor]:
        """自由生成主循环 / Main decode loop: walk the schedule, yield latent patches.

        【重点·难点】从 ``position`` 起遍历 schedule:
        - 遇 audio span: 先按 EOS 头判定是否该停 → ``_decode_next_audio`` 解出 latent patch →
          ``_consume_audio_patch`` 反馈回模型 → yield 该 patch;若下个 token 仍是 audio, 追加一段
          hidden 桥接;若 EOS 触发则在产出当前 patch 后置 ``end_flag`` 并结束。
        - 遇文本: 一次性消费到下一个 audio span 之前 (``_consume_text_schedule``)。
        ``span_cursor`` 用二分 (``searchsorted``) 初始化, 指向下一个 audio span 在 ``span_positions``
        中的下标, 以便 O(1) 取"下一个 audio 位置"。生成器逐 patch yield, 支撑流式合成。
        """
        span_cursor = torch.searchsorted(
            span_positions,
            torch.tensor(
                position,
                device=span_positions.device,
                dtype=span_positions.dtype,
            ),
        ).item()
        while position < generation_schedule.size(1):
            token_id = int(generation_schedule[0, position].item())
            if token_id in audio_placeholder_ids:
                # EOS 判定要在解码当前 patch 之前做 (基于此刻的 llm_hiddens), 但停在产出后。
                stop_after_current_audio = self._should_stop_after_current_audio(
                    state,
                    eos_threshold=eos_threshold,
                )
                audio_patch = self._decode_next_audio(
                    state,
                    device=device,
                    g_cond=g_cond,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
                self._consume_audio_patch(
                    state,
                    audio_patch=audio_patch,
                )
                if self._next_token_is_audio_span(
                    generation_schedule,
                    position=position,
                    audio_placeholder_ids=audio_placeholder_ids,
                ):
                    # 相邻 audio span: 把刚 step 出来的 hidden 作为下一个 patch 的条件桥接。
                    self._append_hidden_chunk(state, state.llm_hiddens)
                position += 1
                span_cursor += 1
                yield audio_patch
                if stop_after_current_audio:
                    state.end_flag = True
                    return  # EOS: 已产出当前 patch, 终止生成
                continue
            # 文本段: 取下一个 audio span 位置 (没有则到序列末), 一次性把这段文本喂进 LLM。
            next_audio_position = (
                int(span_positions[span_cursor].item())
                if span_cursor < span_positions.numel()
                else generation_schedule.size(1)
            )
            position = self._consume_text_schedule(
                generation_schedule,
                position=position,
                next_audio_position=next_audio_position,
                state=state,
            )

    def _should_stop_after_current_audio(
        self, state: _GenerateState, *, eos_threshold: float
    ) -> bool:
        """判断是否应在当前 audio patch 后停止 / EOS check before decoding this patch.

        用 EOS 头对最近一位 ``llm_hiddens`` 求 softmax, 取 "停止" 类 (index 1) 概率与
        ``eos_threshold`` 比较;或 ``end_flag`` 已被先前置位。返回 True 表示产出当前 patch 后结束。
        """
        if state.llm_hiddens is None:
            return False
        eos = (
            self.core.eos_proj(state.llm_hiddens).softmax(dim=-1)[:, -1, 1]
            > eos_threshold
        )
        return state.end_flag or bool(eos.item())

    # endregion Prompt conditioning and decode state helpers

    # region Public generation APIs
    @torch.no_grad()
    def _generate_latents_stream(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
    ) -> Iterator[torch.Tensor]:
        """流式产出反归一化 latent patch / Stream denormalized latent patches (no vocoder).

        端到端的 latent 级编排 (不含声码器):
        1) 准备 prompt 条件 (g_cond / prompt_patches / prompt_latents);
        2) 找出 schedule 里所有 audio span 并校验数量 (>= prompt patch 数 + 1);
        3) 分配静态生成状态;若有 prompt prefill, 先把 prompt latent 编成 LLM 嵌入;
        4) ``_prefill`` 预填 prompt 区间, ``_decode`` 自由生成主循环;
        5) 若有 prompt prefill, 丢弃第一个"重生成的 prompt 尾巴" patch, 其余反归一化后 yield。

        ``data`` 仅支持 batch=1 (流式自回归)。``eos_threshold`` 越低越早停。生成器, 供
        ``generate_audio_stream`` / ``generate_audio`` 复用。
        """
        dtype = get_dtype(precision)
        device = next(self.core.parameters()).device
        use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            generation_schedule: torch.Tensor = data["generation_schedule"]
            if generation_schedule.size(0) != 1:
                raise ValueError(
                    "DotsTtsModel.generate expects batch size 1 for generation_schedule."
                )

            # 只有"既给参考音频又给参考文本"才做内容 prefill (语音克隆);否则仅借音色 (g_cond)。
            use_prompt_prefill = data.get("prompt_audio") is not None and bool(
                data.get("prompt_text")
            )
            prompt_conditioning = self._prepare_prompt_conditioning(
                data.get("prompt_audio"),
                use_prompt_prefill=use_prompt_prefill,
                speaker_scale=speaker_scale,
            )
            has_prompt_prefill = prompt_conditioning.prompt_patches is not None
            prompt_patch_count = (
                0
                if not has_prompt_prefill
                else int(prompt_conditioning.prompt_patches.size(1))
            )
            audio_placeholder_ids = set(self.core.audio_span_token_ids)
            span_positions = self._find_audio_span_positions(
                generation_schedule,
                audio_placeholder_ids=audio_placeholder_ids,
            )
            span_count = int(span_positions.numel())
            minimum_required_spans = prompt_patch_count + 1
            if span_count < minimum_required_spans:
                raise ValueError(
                    f"generation_schedule provides {span_count} audio spans, but prompt prefill requires "
                    f"{prompt_patch_count} spans and generation requires at least one additional decode span."
                )
            logger.info(
                "Latent generation prepared: schedule_audio_spans={} prompt_patch_count={} "
                "minimum_required_spans={}",
                span_count,
                prompt_patch_count,
                minimum_required_spans,
            )

            state = self._allocate_generate_state(
                max_audio_patch_count=span_count,
                device=device,
                dtype=dtype,
            )
            prompt_patch_embeddings = self._prefill_prompt_latents(
                prompt_conditioning.prompt_latents,
                state=state,
            )
            position = self._prefill(
                generation_schedule,
                state=state,
                span_positions=span_positions,
                prompt_patches=prompt_conditioning.prompt_patches,
                prompt_patch_embeddings=prompt_patch_embeddings,
                audio_placeholder_ids=audio_placeholder_ids,
            )

            payload_patch_count = 0
            # 有 prompt prefill 时, decode 的第一个 patch 是对 prompt 尾段的"重生成", 需丢弃。
            should_drop_regenerated_prompt_patch = has_prompt_prefill
            for audio_patch in self._decode(
                generation_schedule,
                position=position,
                state=state,
                audio_placeholder_ids=audio_placeholder_ids,
                span_positions=span_positions,
                device=device,
                g_cond=prompt_conditioning.g_cond,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                eos_threshold=eos_threshold,
            ):
                if should_drop_regenerated_prompt_patch:
                    should_drop_regenerated_prompt_patch = False
                    continue
                payload_patch_count += 1
                if payload_patch_count == 1 or payload_patch_count % 10 == 0:
                    logger.info(
                        "Latent generation progress: payload_audio_patches={}",
                        payload_patch_count,
                    )
                yield self.core.io_helper.denormalize(audio_patch)

            if payload_patch_count == 0:
                if has_prompt_prefill:
                    raise RuntimeError(
                        "Generation produced no payload latents after discarding the regenerated prompt-tail patch. "
                        "This usually means EOS triggered immediately after prompt continuation "
                        "or the generation schedule did not provide an effective decode span."
                    )
                raise RuntimeError(
                    "Generation produced no decodable latents. "
                    "This usually means EOS triggered before the first decode patch "
                    "or the generation schedule did not provide an effective decode span."
                )
            logger.info(
                "Latent generation completed: payload_audio_patches={}",
                payload_patch_count,
            )

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """整段把 latent 解成波形 / Decode the full latent sequence to a waveform (non-stream).

        ``latents`` 形如 (B, T, D);转成声码器期望的 (B, D, T) 且转 fp32, ``do_sample=False`` 取确定性解码。
        """
        with measure_inference("latent_decoder"):
            return self.vocoder.inference_from_latents(
                latents.transpose(1, 2).float(),
                do_sample=False,
            )

    @torch.no_grad()
    def _init_vocoder_stream_state(self) -> Any:
        """初始化声码器流式状态 / Init the vocoder streaming state (LSTM + sliding window).

        ``chunk_size`` 取 ``latent_patch_size``——每次正好喂一个 latent patch, 与自回归生成的粒度对齐。
        """
        return self.vocoder.init_stream_state(
            batch_size=1,
            chunk_size=self.core.latent_patch_size,
        )

    @torch.no_grad()
    def _stream_vocoder_patch(
        self,
        latent_patch: torch.Tensor,
        *,
        stream_state: Any,
    ) -> torch.Tensor:
        """流式把一个 latent patch 解成音频块 / Stream one latent patch into an audio chunk.

        非优化路径直接调 ``vocoder.stream_step``。优化路径手动展开成可编译的 ``compiled_stream_step``:
        把 LSTM 隐状态 (h, c)、滑动窗口、有效帧数显式传入/传出, 再手动更新 ``stream_state``——这样
        ``torch.compile`` 看到的是纯函数 (无 Python 副作用), 利于 ``fullgraph`` 与 CUDA graph。
        ``valid_frames`` 取 LSTM 已累计帧数与窗口长度的较小者, 防止越界。``.clone()`` 是因为编译输出
        可能复用 buffer, 必须拷出后再写回状态。
        """
        latents = latent_patch.transpose(1, 2)
        if not self._optimize_enabled:
            with measure_inference("vocoder"):
                return self.vocoder.stream_step(latents, stream_state)

        valid_frames = min(
            stream_state.decoder.total_frames,
            stream_state.decoder.window.size(-1),
        )
        valid_frames_tensor = stream_state.decoder.window.new_tensor(
            valid_frames,
            dtype=torch.int64,
        )
        vocoder_step = self._get_compiled_method(
            "vocoder.step",
            self.vocoder,
            "compiled_stream_step",
        )
        with measure_inference("vocoder"):
            audio_window, hidden_h, hidden_c, new_window = vocoder_step(
                latents,
                stream_state.lstm_hidden[0],
                stream_state.lstm_hidden[1],
                stream_state.decoder.window,
                valid_frames_tensor,
            )
        stream_state.lstm_hidden = (hidden_h.clone(), hidden_c.clone())
        stream_state.decoder.window = new_window.clone()
        stream_state.decoder.total_frames += int(latents.size(-1))
        audio_chunk = self.vocoder._slice_stream_audio_window(
            audio_window,
            stream_state,
            final=False,
        )
        return audio_chunk.clone()

    @torch.no_grad()
    def _flush_vocoder_stream(self, stream_state: Any) -> torch.Tensor:
        """冲刷声码器尾部残留 / Flush the vocoder's trailing buffered audio.

        流式解码每步只输出"可定稿"的音频区间, 末尾仍有窗口内未吐出的样本, 生成结束时一次性冲出。
        """
        with measure_inference("vocoder"):
            return self.vocoder.stream_flush(stream_state)

    @torch.no_grad()
    def generate_audio_stream(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
    ) -> Iterator[torch.Tensor]:
        """流式合成音频 (公开 API) / Public streaming TTS: yield waveform chunks.

        把 ``_generate_latents_stream`` 产出的每个 latent patch 经流式声码器解成音频块即时 yield,
        生成结束再冲出尾部残留。这是 double-streaming 的下半段 (latent 流 → 音频流), 适合边生成边播放。
        空音频块 (长度 0) 跳过不 yield。
        """
        stream_state = self._init_vocoder_stream_state()
        for latent_patch in self._generate_latents_stream(
            data,
            precision=precision,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            speaker_scale=speaker_scale,
            eos_threshold=eos_threshold,
        ):
            audio_chunk = self._stream_vocoder_patch(
                latent_patch,
                stream_state=stream_state,
            )
            if audio_chunk.size(-1) > 0:
                yield audio_chunk

        final_chunk = self._flush_vocoder_stream(stream_state)
        if final_chunk.size(-1) > 0:
            yield final_chunk

    @torch.no_grad()
    def generate_audio(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
    ) -> torch.Tensor:
        """整段合成音频 (公开 API) / Public non-streaming TTS: return the full waveform.

        先把所有 latent patch 收集齐 (``_generate_latents_stream`` 跑到底), 拼成完整 latent 序列后用
        ``_decode_latents`` 一次性解码——相比流式版没有窗口拼接误差, 但需等全部生成完才出声。
        返回波形张量 (B, n_samples)。注意此路径不接收 ``eos_threshold`` (用 ``_decode`` 默认值)。
        """
        latent_patches = list(
            self._generate_latents_stream(
                data,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
            )
        )
        logger.info(
            "Vocoder decode started: latent_patch_count={}",
            len(latent_patches),
        )
        audio = self._decode_latents(torch.cat(latent_patches, dim=1))
        logger.info(
            "Vocoder decode completed: waveform_samples={}",
            audio.shape[-1],
        )
        return audio
    # endregion Public generation APIs
