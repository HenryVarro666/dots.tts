from __future__ import annotations

# =============================================================================
# 模块概览 / Module overview
# -----------------------------------------------------------------------------
# 本文件是 dots.tts 的「推理编排层」(inference orchestration)。它把上层调用
# (CLI / API) 提交的「文本 + 可选 prompt 音频」请求，整理成模型可直接消费的
# inputs，然后驱动底层 DotsTtsModel 完成自回归 + flow-matching 解码，最终产出
# 24/48 kHz 波形。它本身不含任何神经网络前向，只负责「准备 → 调度 → 计时」。
#
# This is the inference-orchestration layer of dots.tts. It turns a user request
# ("text + optional prompt audio") into model-ready `inputs`, drives the
# underlying DotsTtsModel through autoregressive + flow-matching decoding, and
# returns a waveform. It contains no neural forward pass itself — only request
# preparation, schedule assembly, generation dispatch, and timing/profiling.
#
# 在整条推理数据流里的位置 / Where it sits in the inference pipeline:
#   原始请求 (text, prompt_audio_path, prompt_text, language, template ...)
#     └─ DotsTtsRuntime._prepare_inputs  ← 本文件：文本规整 / 语言标签 /
#        prompt 音频加载与重采样 / 用 build_generation_schedule 构建 schedule
#          └─ DotsTtsModel.generate_audio[_stream]  ← 真正的解码 (LLM 主干 +
#             flow-matching DiT 声学头 + BigVGAN AudioVAE 解码)
#               └─ 波形 waveform (torch.Tensor)
#
# 关键类 / 函数 (Key classes / functions):
#   - RuntimeInputs        : 传给模型的输入字典的 TypedDict 形状声明
#   - DotsTtsRuntime       : 推理编排主类
#       · from_pretrained / _resolve_pretrained_path : 加载权重(本地或 HF 下载)
#       · _prepare_inputs (+ 一组 _process_* 辅助)    : 请求规整 → inputs
#       · generate / generate_stream                  : 一次性 / 流式生成入口
#
# 关键 ML 概念 (后续注释会反复出现) / Key ML concepts referenced below:
#   continuous-latent TTS、flow-matching、velocity field、ODE solver (euler 等)、
#   classifier-free guidance (CFG, 即 guidance_scale)、x-vector 说话人条件
#   (speaker_scale)、patch (latent 被切成定长 patch 自回归生成)、
#   generation schedule (预先排好的 token 调度序列，告诉模型何时吐音频 patch)。
# =============================================================================

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator, TypedDict

import librosa
import torch
from huggingface_hub import snapshot_download
from loguru import logger

from dots_tts.data.pipelines.tokenizing import build_generation_schedule
from dots_tts.data.pipelines.tts_pipeline import (
    DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
    DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    DEFAULT_TRAIN_TEMPLATE,
)
from dots_tts.models.dots_tts.model import DotsTtsModel
from dots_tts.utils.audio import high_quality_resample
from dots_tts.utils.profiling import (
    InferenceProfiler,
    activate_inference_profiler,
    inference_profiling,
    log_inference_profile,
)
from dots_tts.utils.text import (
    attach_language_tag,
    detect,
    normalize_language_code,
    normalize_text,
)
from dots_tts.utils.util import get_dtype

RUNTIME_TEMPLATE_BY_NAME = {
    "tts": DEFAULT_TRAIN_TEMPLATE,
    "instruction_tts": DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    "text_to_audio": DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    "tts_interleave": DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
}


class RuntimeInputs(TypedDict, total=False):
    """传给 DotsTtsModel.generate_audio[_stream] 的输入字典的结构声明。

    Typed shape of the dict handed to the model. `total=False` 表示所有字段都
    可选 —— 例如纯文本合成 (无 voice clone) 时不会有 `prompt_audio`。

    字段 / Fields:
        fid                : 请求指纹 (sha1 截断)，仅用于日志关联，便于追踪一次请求。
        language           : 规整后的语言码 (如 "ZH"/"EN")，无则为空串 ""。
        text               : 规整后的目标文本 (待合成)。
        prompt_text        : prompt 音频对应的转写文本 (voice clone 时使用)，无则 ""。
        template_name      : 选用的 prompt 模板名 (见 RUNTIME_TEMPLATE_BY_NAME)。
        generation_schedule: (1, L) long 张量；预排好的 token 调度序列，告诉模型
                             文本条件在哪、音频 patch span 占多少步 —— 自回归解码
                             按此 schedule 推进。
        prompt_audio       : (1, N) float 张量；已重采样到模型 sample_rate 的参考
                             音频波形，供 AudioVAE 编码出 latent 作为声学/音色条件。
    """

    fid: str
    language: str
    text: str
    prompt_text: str
    template_name: str
    generation_schedule: torch.Tensor
    prompt_audio: torch.Tensor


class DotsTtsRuntime:
    """dots.tts 的推理编排主类 (high-level inference façade)。

    职责 / Responsibility:
        持有一个已加载到目标设备/精度的 DotsTtsModel，对外暴露简单的
        generate / generate_stream 接口；内部完成请求规整、generation schedule
        构建、解码调度与计时/profiling。它是「无状态」的 —— 每次 generate 都自带
        全部条件 (text / prompt)，不在请求之间保留对话状态。

    设计要点 / Design notes:
        - 设备与精度在 __init__ 一次性固定 (CUDA 优先，否则 CPU 单线程)。
        - `optimize=True` 时会预跑 warmup，触发 torch.compile / CUDA graph 之类
          的一次性编译开销，让后续请求延迟更稳定。
        - sample_rate 取自 vocoder 配置，是后续 RTF (real-time factor) 计算与
          波形输出的基准。
    """

    # region Lifecycle and pretrained loading
    def __init__(
        self,
        model: DotsTtsModel,
        pretrained_path: Path,
        *,
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
    ):
        """绑定模型到设备/精度并完成运行期初始化。

        参数 / Args:
            model              : 已构建好的 DotsTtsModel (权重已 load，但尚未 .to())。
            pretrained_path    : 权重所在目录 (本地路径或 HF snapshot 落地目录)，仅记录用。
            precision          : 计算精度，如 "bfloat16"/"float32"；映射为 torch dtype。
            optimize           : 是否启用推理优化 + warmup 预热。
            max_generate_length: 单次生成的最大音频 patch 数 (自回归步数上限)，
                                 也是 schedule 里音频 span 的长度上限。
        """
        self.model = model
        self.pretrained_path = pretrained_path
        self.precision = precision
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # CPU 回退路径：限制为单线程，避免 over-subscription 反而拖慢小算子。
            # CPU fallback: pin to one thread to avoid thread over-subscription.
            self.device = torch.device("cpu")
            torch.set_num_threads(1)
        if self.device.type == "cuda" and self.precision.lower() in {
            "fp32",
            "torch.float32",
            "float32",
        }:
            # 仅在 CUDA + fp32 时打开 TF32 matmul (用 TensorCore 加速 fp32 矩阵乘)，
            # 在精度几乎无损的前提下提速；bf16 路径无需此设置。
            torch.set_float32_matmul_precision("high")
        target_dtype = get_dtype(self.precision)  # 字符串精度名 → torch.dtype
        # 只把 core (LLM 主干 + DiT 声学头) 转到目标 dtype；vocoder 等子模块保持其
        # 自身精度，避免低精度破坏 BigVGAN 的数值稳定性。
        self.model.core.to(dtype=target_dtype)
        self.model = self.model.to(self.device).eval()  # 搬到设备并进入 eval 模式
        self.optimize = bool(optimize)
        self.max_generate_length = int(max_generate_length)
        self.model.set_optimize(self.optimize)
        self.sample_rate = int(self.model.config.vocoder.sample_rate)  # RTF/输出基准采样率
        if self.optimize and hasattr(self.model, "run_warmup"):
            # 预热：用一次伪请求触发 torch.compile / CUDA graph 等一次性编译，
            # 把编译开销挪到初始化阶段，让真实请求延迟稳定。
            self.model.run_warmup(
                max_generate_length=self.max_generate_length,
                precision=self.precision,
            )
        logger.info(
            "Runtime initialized: pretrained_path={} device={} sample_rate={} "
            "precision={} "
            "optimize={} max_audio_patch_count={}",
            self.pretrained_path,
            self.device,
            self.sample_rate,
            self.precision,
            self.optimize,
            self.max_generate_length,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
    ) -> DotsTtsRuntime:
        """从本地目录或 HF Hub repo id 加载权重并构造一个可用的 Runtime。

        这是大多数调用方的入口：给定 `model_name_or_path` (本地路径或形如
        "rednote-hilab/dots.tts" 的 repo id)，先解析/下载权重，再 from_pretrained
        出 DotsTtsModel，最后交给 __init__ 完成设备/精度绑定。

        Args / 同名参数语义见 __init__；额外的 revision/cache_dir 仅在需要从
        HF 下载 snapshot 时生效 (本地已存在则忽略)。
        """
        logger.info(
            "Runtime load started: model={} revision={} cache_dir={} precision={}",
            model_name_or_path,
            revision,
            cache_dir,
            precision,
        )
        pretrained_path = cls._resolve_pretrained_path(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        loaded_model = DotsTtsModel.from_pretrained(pretrained_path)
        logger.info("Runtime load completed: pretrained_path={}", pretrained_path)
        return cls(
            model=loaded_model,
            pretrained_path=pretrained_path,
            precision=precision,
            optimize=optimize,
            max_generate_length=max_generate_length,
        )

    @classmethod
    def _resolve_pretrained_path(
        cls,
        model_name_or_path: str,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> Path:
        """把 model_name_or_path 解析为一个本地存在的权重目录 Path。

        优先把入参当本地路径试；不存在则视为 HF repo id，用 snapshot_download
        拉取整个 repo (含 config / 权重 / tokenizer) 到本地缓存后返回落地目录。
        """
        logger.info(
            "Resolving pretrained path: model={} revision={} cache_dir={}",
            model_name_or_path,
            revision,
            cache_dir,
        )
        # 先按本地路径解释 (展开 ~ 并转绝对路径)；存在即直接用，跳过任何网络访问。
        resolved_path = Path(model_name_or_path).expanduser().resolve()
        if resolved_path.exists():
            logger.info("Using local pretrained directory: path={}", resolved_path)
            return resolved_path

        logger.info(
            "Downloading pretrained snapshot: repo_id={} revision={} cache_dir={}",
            model_name_or_path,
            revision,
            cache_dir,
        )
        # 本地不存在 → 当作 HF repo id，整仓下载到缓存 (会被 hub 去重/复用)。
        snapshot_dir = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        resolved_path = Path(snapshot_dir).expanduser().resolve()
        logger.info("Pretrained snapshot ready: path={}", resolved_path)
        return resolved_path
    # endregion Lifecycle and pretrained loading

    # region Request normalization and metadata
    @staticmethod
    def _build_request_id(
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str,
        language: str | None = None,
    ) -> str:
        """对请求内容做哈希，得到一个稳定的短 request id (fid) 用于日志关联。

        把影响输出的所有字段塞进一个 dict，JSON 序列化后取 sha1 前 16 位。
        相同请求 → 相同 fid，便于在日志里串起一次生成的多条记录；它不是音频
        缓存键，仅用于可观测性 (observability)。
        """
        payload = {
            "text": text,
            "prompt_audio_path": prompt_audio_path,
            "prompt_text": prompt_text,
            "template_name": template_name,
        }
        if language is not None:
            payload["language"] = language
        # sort_keys 保证字段顺序无关，序列化结果稳定 → 同请求恒得同 fid。
        digest = hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def _load_prompt_audio(
        self,
        prompt_audio_path: str,
    ) -> torch.Tensor:
        """加载并预处理 voice-clone 用的参考音频，返回 (1, N) 波形张量。

        流程 / Pipeline:
            1) librosa 以原生采样率读单声道 (sr=None, mono=True)。
            2) 去除首尾静音 (trim, top_db=30) —— 静音会浪费 patch、干扰音色条件。
            3) 高质量重采样到模型 sample_rate，使其与训练分布一致。
            4) 保证至少二维 (1, N)，下游 AudioVAE 编码期望带 batch/channel 维。

        返回 / Returns:
            (1, N) 的 float 波形，N 为重采样后样本数。后续会被编码成 latent 作为
            说话人/音色与韵律条件。
        """
        logger.info("Loading prompt audio: path={}", prompt_audio_path)
        prompt_audio, sample_rate = librosa.load(prompt_audio_path, sr=None, mono=True)
        # trim 返回 (trimmed_array, index)，取 [0] 即去静音后的波形。
        prompt_audio = librosa.effects.trim(prompt_audio, top_db=30)[0]
        prompt_audio = torch.from_numpy(prompt_audio).unsqueeze(0)  # (N,) → (1, N)
        prompt_audio = high_quality_resample(
            prompt_audio,
            orig_sr=sample_rate,
            target_sr=self.sample_rate,
        )
        # 重采样实现可能压回一维；统一补回 (1, N) 形状再返回，下游形状假设才成立。
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        logger.info(
            "Prompt audio loaded: path={} original_sample_rate={} resampled_sample_rate={} "
            "samples={}",
            prompt_audio_path,
            sample_rate,
            self.sample_rate,
            prompt_audio.shape[-1],
        )
        return prompt_audio

    def _resolve_language(
        self,
        language: str | None,
        *,
        text: str,
    ) -> str | None:
        """把用户给的 language 解析为标准语言码 (如 "ZH")，或 None 表示不加标签。

        三类入参 / Three cases:
            - None / ""/ "none" → 返回 None：不显式约束语言。
            - "auto_detect"     → 用 lingua 在 `text` 上检测语言再标准化。
            - 其它              → 当作显式语言码/名，标准化；非法则报错。
        """
        if language is None:
            return None

        stripped = language.strip()
        if not stripped or stripped.lower() == "none":
            return None
        if stripped.lower() == "auto_detect":
            # 自动检测：用规整后的文本跑语言识别，再统一成标准码。
            return normalize_language_code(detect(text))

        normalized_language = normalize_language_code(stripped)
        if normalized_language is None:
            raise ValueError(
                f"Unsupported language={language!r}. "
                "Expected 'none', 'auto_detect', or a valid language code/name."
            )
        return normalized_language

    def _process_prompt_text(
        self,
        prompt_text: str | None,
        *,
        language: str | None = None,
    ) -> str:
        """规整 prompt 转写文本：去空白、补换行分隔符、按需加语言标签。

        prompt_text 是参考音频的转写，会和目标文本拼在一起喂模型。结尾补 "\\n"
        是为了在拼接 (`{prompt_text}{text}`) 时把 prompt 段与正文段分开；语言标签
        (形如 "[ZH]") 加在 prompt 前，让模型从一开始就锚定语种。
        空输入统一规整成空串 ""，方便上层用真值判断。
        """
        if prompt_text is None:
            return ""
        prompt_text = prompt_text.strip()
        if not prompt_text:
            return ""

        prompt_text += "\n"  # 分隔符：拼接时把 prompt 段与目标文本段隔开
        if language is not None:
            prompt_text = attach_language_tag(prompt_text, language)
        return prompt_text

    def _process_text(
        self,
        text: str,
        *,
        language: str | None = None,
        normalize: bool = False,
    ) -> tuple[str, str | None]:
        """规整目标文本并解析其语言。

        先去首尾空白；`normalize=True` 时再做 TN (text normalization，把数字/符号
        等读法展开成可朗读形式)。语言解析在规整后的文本上做，保证 auto_detect 用
        的是最终要合成的文本。

        Returns:
            (规整后文本, 语言码或 None)。
        """
        stripped = text.strip()
        if normalize:
            stripped = normalize_text(stripped)
        # 语言检测/解析放在规整之后：auto_detect 应基于最终待合成文本。
        resolved_language = self._resolve_language(language, text=stripped)
        return stripped, resolved_language

    def _estimate_prompt_audio_patch_count(
        self,
        *,
        prompt_audio: torch.Tensor | None,
        prompt_text: str,
    ) -> int:
        """估算 prompt 音频会占用多少个 audio patch (向上取整)。

        为什么要算 / Why:
            voice clone 时，prompt 音频本身会先占掉 schedule 里若干音频 patch，
            真正"新生成"的预算 = max_generate_length - 这部分。提前估出来，用于
            在 _prepare_inputs 里做边界校验，避免预算不足导致生成被 prompt 占满。

        换算 / Conversion:
            samples_per_patch = patch_size(每 patch 含几个 latent 帧) ×
                                 hop_size(每 latent 帧对应几个波形样本)。
            然后用 ceil(prompt_samples / samples_per_patch) 得到 patch 数。
            无 prompt 音频或无 prompt_text 时返回 0 (不构成 clone 条件)。
        """
        if prompt_audio is None or not prompt_text:
            return 0
        samples_per_patch = int(self.model.config.patch_size * self.model.hop_size)
        prompt_samples = int(prompt_audio.shape[-1])
        # (a + b - 1) // b 是整数向上取整 ceil(a/b) 的惯用写法。
        return (prompt_samples + samples_per_patch - 1) // samples_per_patch
    # endregion Request normalization and metadata

    # region Generation schedule assembly
    def _normalize_template_name(self, template_name: str | None) -> str:
        """校验并归一 prompt 模板名；None 默认回退到 "tts"。

        模板决定文本/音频占位符如何排布 (见 RUNTIME_TEMPLATE_BY_NAME)，进而决定
        build_generation_schedule 生成怎样的调度序列。未知名直接报错以尽早失败。
        """
        if template_name is None:
            return "tts"
        if template_name not in RUNTIME_TEMPLATE_BY_NAME:
            raise ValueError(
                f"Unknown template_name={template_name!r}. "
                f"Expected one of {sorted(RUNTIME_TEMPLATE_BY_NAME)}."
            )
        return template_name

    def _prepare_inputs(
        self,
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str | None,
        language: str | None = None,
        normalize_text: bool = False,
    ) -> RuntimeInputs:
        """把一次原始请求装配成模型可消费的 RuntimeInputs 字典。

        这是本文件的核心：串起前面所有 _process_* 辅助，产出含 generation
        schedule、(可选) prompt_audio、语言标签等的完整 inputs。

        关键步骤 / Key steps:
            1) 归一模板名并取出模板；校验 prompt_text 必须伴随 prompt_audio。
            2) 规整目标文本 + 解析语言；规整 prompt 文本并按需打语言标签。
            3) 无 prompt 但有语言时，把语言标签直接打在目标文本上 (否则标签已随
               prompt 段携带，避免重复)。
            4) 估 prompt 占用的 patch 数，校验 max_generate_length 仍有富余。
            5) 用 build_generation_schedule 把 "prompt+text" 编成调度 token 序列，
               转成 (1, L) long 张量放进 inputs["generation_schedule"]。

        Returns:
            RuntimeInputs (见该 TypedDict 的字段说明)。
        """
        normalized_template_name = self._normalize_template_name(template_name)
        template = RUNTIME_TEMPLATE_BY_NAME[normalized_template_name]
        # voice clone 的两个条件成对出现：有转写必须有对应参考音频，否则无意义。
        if prompt_text and not prompt_audio_path:
            raise ValueError("prompt_text requires prompt_audio_path.")

        normalized_text, normalized_language = self._process_text(
            text,
            language=language,
            normalize=normalize_text,
        )
        normalized_prompt_text = self._process_prompt_text(
            prompt_text,
            language=normalized_language,
        )
        # 语言标签只打一次：若有 prompt 段，标签已随 prompt 携带；只有在「无 prompt
        # 文本」时才需要把标签补到目标文本上，避免出现重复的 [LANG] 前缀。
        if normalized_language is not None and not normalized_prompt_text:
            normalized_text = attach_language_tag(normalized_text, normalized_language)
        inputs: RuntimeInputs = {
            "fid": self._build_request_id(
                text=normalized_text,
                prompt_audio_path=prompt_audio_path,
                prompt_text=normalized_prompt_text,
                template_name=normalized_template_name,
                language=normalized_language,
            ),
            "language": normalized_language or "",
            "text": normalized_text,
            "prompt_text": normalized_prompt_text,
            "template_name": normalized_template_name,
        }

        # 仅在提供参考音频时才加载，避免纯文本合成做无谓 IO。
        if prompt_audio_path:
            inputs["prompt_audio"] = self._load_prompt_audio(prompt_audio_path)
        prompt_audio_patch_count = self._estimate_prompt_audio_patch_count(
            prompt_audio=inputs.get("prompt_audio"),
            prompt_text=normalized_prompt_text,
        )
        # 边界校验：prompt 已占掉一部分 patch 预算，max_generate_length 必须严格大于
        # 它，否则没有空间留给真正要新生成的音频 → 直接报错而非生成空结果。
        if (
            prompt_audio_patch_count > 0
            and self.max_generate_length <= prompt_audio_patch_count
        ):
            raise ValueError(
                "max_generate_length must exceed prompt audio patch count when prompt_text is provided: "
                f"max_generate_length={self.max_generate_length} "
                f"prompt_audio_patch_count={prompt_audio_patch_count}."
            )

        # 把 "prompt 文本 + 目标文本" 套进模板，编成调度 token id 序列：文本条件
        # token + 标记音频 span 的特殊 token (start / span×N / end)，N 受
        # max_audio_tokens 约束。自回归解码就按这条 schedule 推进。
        schedule_spec = build_generation_schedule(
            text=f"{normalized_prompt_text}{normalized_text}",
            tokenizer=self.model.tokenizer,
            template=template,
            max_audio_tokens=self.max_generate_length,
        )
        schedule = torch.tensor(
            schedule_spec["schedule_ids"],
            dtype=torch.long,
            device=self.device,
        )
        inputs["generation_schedule"] = schedule.unsqueeze(0)  # (L,) → (1, L) 加 batch 维
        logger.info(
            "Inputs prepared: request_id={} template_name={} "
            "language={} text_len={} prompt_text_len={} schedule_length={} "
            "prompt_audio_patch_count={} max_audio_patch_count={} has_prompt_audio={}",
            inputs["fid"],
            normalized_template_name,
            normalized_language,
            len(normalized_text),
            len(normalized_prompt_text),
            schedule.numel(),
            prompt_audio_patch_count,
            self.max_generate_length,
            bool(prompt_audio_path),
        )
        return inputs
    # endregion Generation schedule assembly

    # region Public generation APIs
    def generate_stream(
        self,
        *,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str | None = None,
        language: str | None = None,
        speaker_scale: float = 1.5,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        normalize_text: bool = False,
        profile_inference: bool = False,
    ) -> Iterator[torch.Tensor]:
        """流式生成：边解码边吐音频 chunk (low-latency / double-streaming)。

        与 generate 的唯一区别是「随产随出」：内部调 generate_audio_stream，每解出
        一个/一批 latent patch 并经 vocoder 解码后立刻 yield，适合实时播放。

        关键采样/引导参数 / Key sampling & guidance args:
            speaker_scale  : x-vector 说话人条件的缩放；放大可强化音色相似度。
            ode_method     : flow-matching 的 ODE solver (如 "euler")，把 velocity
                             field 沿时间积分还原 latent。
            num_steps      : ODE 积分步数；越多越精细、越慢。
            guidance_scale : classifier-free guidance (CFG) 强度；>1 让条件分支
                             相对无条件分支被放大，提升与文本/音色的贴合度。
            normalize_text : 是否对输入文本做 TN。
            profile_inference: 是否采集各阶段耗时做 profiling。

        Yields:
            一段段 (..., samples) 的波形张量；调用方负责拼接/播放。
        """
        inputs = self._prepare_inputs(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            normalize_text=normalize_text,
        )
        logger.info(
            "Streaming generation started: request_id={} text_len={} has_prompt_audio={} "
            "has_prompt_text={} template_name={} language={} precision={} ode_method={} num_steps={} "
            "guidance_scale={} speaker_scale={} max_audio_patch_count={} normalize_text={}",
            inputs["fid"],
            len(inputs["text"]),
            bool(prompt_audio_path),
            bool(inputs["prompt_text"]),
            inputs["template_name"],
            inputs["language"] or None,
            self.precision,
            ode_method,
            num_steps,
            guidance_scale,
            speaker_scale,
            self.max_generate_length,
            normalize_text,
        )
        start_time = time.time()
        emitted_samples = 0  # 累计已吐出的样本数 → 末尾换算音频时长与 RTF
        chunk_count = 0
        profiler: InferenceProfiler | None = None
        try:
            profiler = (
                InferenceProfiler(self.device) if profile_inference else None
            )
            stream = self.model.generate_audio_stream(
                inputs,
                precision=self.precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
            )
            # 手动 next() 而非 for：以便用 activate_inference_profiler 把每次「取下一
            # 个 chunk」的解码耗时单独计入 profiler；StopIteration 即流结束。
            while True:
                try:
                    with activate_inference_profiler(profiler):
                        chunk = next(stream)
                except StopIteration:
                    break
                emitted_samples += int(chunk.shape[-1])
                chunk_count += 1
                yield chunk
        except Exception:
            # 生成器内的异常会在迭代点抛出；记日志带上 fid 再重抛，便于定位。
            logger.exception(
                "Streaming generation failed: request_id={}",
                inputs["fid"],
            )
            raise
        time_used = time.time() - start_time
        duration_seconds = emitted_samples / self.sample_rate
        # RTF (real-time factor) = 计算耗时 / 音频时长，<1 即快于实时；时长为 0
        # 时避免除零，记为 inf。
        rtf = time_used / duration_seconds if duration_seconds > 0 else float("inf")
        if profile_inference and profiler is not None:
            log_inference_profile(
                request_id=inputs["fid"],
                profiling=profiler.summary(duration_seconds=duration_seconds),
                duration_seconds=duration_seconds,
            )
        logger.info(
            "Streaming generation finished: request_id={} chunk_count={} elapsed_seconds={:.3f} "
            "audio_seconds={:.3f} rtf={:.4f} sample_rate={}",
            inputs["fid"],
            chunk_count,
            time_used,
            duration_seconds,
            rtf,
            self.sample_rate,
        )

    def generate(
        self,
        *,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str | None = None,
        language: str | None = None,
        speaker_scale: float = 1.5,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        normalize_text: bool = False,
        profile_inference: bool = False,
    ) -> dict[str, Any]:
        """一次性 (non-streaming) 生成：解码完整段音频后整体返回。

        参数与 generate_stream 完全一致 (含 speaker_scale / ode_method / num_steps /
        guidance_scale 等，语义见 generate_stream 的 docstring)。差别仅在于这里调
        generate_audio 把所有 latent patch 解完再一并 vocoder 解码。

        Returns:
            dict，包含:
                fid          : 请求指纹 (日志关联用)。
                audio        : (..., samples) 的完整波形张量。
                sample_rate  : 采样率。
                time_used    : 端到端耗时 (秒)。
                rtf          : real-time factor (耗时/音频时长)。
                profiling    : 若开启 profile_inference 则为各阶段统计，否则 None。
        """
        inputs = self._prepare_inputs(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            normalize_text=normalize_text,
        )
        logger.info(
            "Generation started: request_id={} text_len={} has_prompt_audio={} "
            "has_prompt_text={} template_name={} language={} precision={} ode_method={} num_steps={} "
            "guidance_scale={} speaker_scale={} max_audio_patch_count={} normalize_text={}",
            inputs["fid"],
            len(inputs["text"]),
            bool(prompt_audio_path),
            bool(inputs["prompt_text"]),
            inputs["template_name"],
            inputs["language"] or None,
            self.precision,
            ode_method,
            num_steps,
            guidance_scale,
            speaker_scale,
            self.max_generate_length,
            normalize_text,
        )
        start_time = time.time()
        profiling = None
        try:
            # inference_profiling 是上下文管理器：enabled=False 时给出空 profiler
            # (无开销)，enabled=True 时在退出后可 summary() 出分阶段耗时。
            with inference_profiling(
                enabled=profile_inference,
                device=self.device,
            ) as profiler:
                audio = self.model.generate_audio(
                    inputs,
                    precision=self.precision,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    speaker_scale=speaker_scale,
                )
        except Exception:
            logger.exception("Generation failed: request_id={}", inputs["fid"])
            raise
        time_used = time.time() - start_time
        duration_seconds = audio.shape[-1] / self.sample_rate  # 最后一维即样本数
        # RTF：耗时/音频时长，<1 快于实时；时长 0 时避免除零记为 inf。
        rtf = time_used / duration_seconds if duration_seconds > 0 else float("inf")
        if profiler is not None:
            profiling = profiler.summary(duration_seconds=duration_seconds)
            log_inference_profile(
                request_id=inputs["fid"],
                profiling=profiling,
                duration_seconds=duration_seconds,
            )
        logger.info(
            "Generation completed: request_id={} elapsed_seconds={:.3f} audio_seconds={:.3f} "
            "rtf={:.4f} sample_rate={}",
            inputs["fid"],
            time_used,
            duration_seconds,
            rtf,
            self.sample_rate,
        )
        return {
            "fid": inputs["fid"],
            "audio": audio,
            "sample_rate": self.sample_rate,
            "time_used": time_used,
            "rtf": rtf,
            "profiling": profiling,
        }
    # endregion Public generation APIs
