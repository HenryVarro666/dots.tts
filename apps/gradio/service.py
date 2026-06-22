"""Gradio 后端服务层 / Gradio backend service layer for dots.tts.

本文件做什么 (What this file does):
    把 ``dots_tts.runtime.DotsTtsRuntime`` (真正的连续潜在 AR-TTS 推理引擎) 封装成一个
    线程安全、可被 Gradio UI 直接调用的服务对象。它**不**包含任何模型/网络结构代码——
    模型加载、flow-matching DiT 采样、BigVGAN 解码等全部委托给 runtime；本文件只负责
    "请求 → 校验/归一化 → 调度 runtime → 落盘 wav → 汇总 metrics" 这条服务侧数据流。

在推理数据流里的位置 (Position in the inference pipeline):
    Gradio UI (apps/gradio/ui 等)  →  本文件 GradioAppService.generate()
        →  DotsTtsRuntime.generate() / generate_stream()  (Qwen2.5 AR 主干 + flow-matching
           声学头 + AudioVAE 解码)  →  返回 waveform 张量  →  本文件写成 .wav 并返回路径/metrics。

关键类/函数清单 (Key classes / functions):
    - PromptPreset / 系列 prompt 函数: 发现并解析磁盘上的参考音色 (reference voice) 预设,
      用于 voice cloning 的 prompt_audio + prompt_text。
    - GradioAppConfig + build_gradio_app_config(): 把命令行/默认参数固化成一份不可变配置。
    - SynthesisRequest / SynthesisResult: 服务层的请求/响应数据契约 (data contract)。
    - GradioAppService: 核心服务对象——缓存 runtime、warmup、归一化请求、流式/非流式生成、
      写文件与输出目录清理。

注意 (Note):
    本文件通过 ``runtime._process_text`` / ``runtime._build_request_id`` 等私有方法
    (带 ``# noqa: SLF001``) 复用 runtime 的内部归一化逻辑, 以便在流式路径下也能算出与
    runtime 内部一致的 request_id (fid), 详见 _build_stream_request_id。
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# 仓库根 = 本文件向上 3 层 (apps/gradio/service.py → 仓库根); src/ 是真正的包根目录。
# REPO_ROOT = repo root (3 levels up); SRC_ROOT holds the importable dots_tts package.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# 把仓库根与 src/ 注入 sys.path, 使得无论从哪个 cwd 启动都能 import dots_tts 与 apps.*。
# Prepend both roots to sys.path so dots_tts / apps imports resolve regardless of cwd.
for import_root in (REPO_ROOT, SRC_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from apps.gradio.constants import (  # noqa: E402
    DEFAULT_EXECUTION_MODE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HOST,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_NUM_STEPS,
    DEFAULT_ODE_METHOD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_RETENTION,
    DEFAULT_PORT,
    DEFAULT_PRECISION,
    DEFAULT_PROMPT_MAPPING_FILE,
    DEFAULT_PROMPT_NAME,
    DEFAULT_PROMPT_NONE,
    DEFAULT_PROMPT_SOURCE_DIR,
    DEFAULT_PROMPTS_DIR,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_SCALE,
    DEFAULT_WARMUP_TEXT,
    PROMPT_AUDIO_SUFFIXES,
)
from apps.gradio.languages import (  # noqa: E402
    SUPPORTED_LANGUAGE_CODE_BY_NAME,
    build_language_choice_items,
)
from dots_tts.runtime import DotsTtsRuntime  # noqa: E402
from dots_tts.utils.util import seed_everything  # noqa: E402

# 执行模式: "generate" = 一次性整段合成; "generate_stream" = double-streaming 流式分块。
# ExecutionMode: one-shot full synthesis vs. chunked (double-streaming) generation.
ExecutionMode = Literal["generate", "generate_stream"]
# UI 下拉项 → runtime 模板名的映射: (UI 内部 id, runtime template_name)。
#   tts                = 纯声音克隆/朗读; instruct_tts = 指令式 TTS; text_to_audio = 通用文本转音频。
# Maps UI synthesis-mode choices to runtime template names used to build the prompt.
GRADIO_SYNTHESIS_MODE_CHOICES = (
    ("tts", "tts"),
    ("instruct_tts", "instruction_tts"),
    ("instruct_tts_general", "text_to_audio"),
)
GRADIO_SYNTHESIS_MODE_TEMPLATE_NAMES = tuple(
    value for _, value in GRADIO_SYNTHESIS_MODE_CHOICES
)


@dataclass(frozen=True)
class PromptPreset:
    """一个参考音色预设 / A single reference-voice (cloning) prompt preset.

    字段 (Fields):
        name        : 预设名 = 音频文件 stem (如 "male_zh"), 也是 UI 下拉里展示/选中的 key。
        audio_path  : 参考音频的绝对路径; 供 CAM++ 提取 x-vector + AudioVAE 编码出参考 latent。
        prompt_text : 参考音频对应的文字转写 (transcript); 留空表示只用音频做音色条件。
    不可变 (frozen) 以便安全地放进 GradioAppConfig 并跨线程共享。
    """

    name: str
    audio_path: str
    prompt_text: str


def _is_prompt_asset(path: Path) -> bool:
    """判断一个文件是否属于 prompt 资产 / Is this path a prompt asset to sync?

    只认两类: 名为 ``prompt_text`` 的映射文件, 或后缀属于受支持音频格式的音频文件。
    Accepts the ``prompt_text`` mapping file or any supported audio file by suffix.
    """
    return path.is_file() and (
        path.name == "prompt_text" or path.suffix.lower() in PROMPT_AUDIO_SUFFIXES
    )


def sync_default_prompt_library(
    source_dir: Path = DEFAULT_PROMPT_SOURCE_DIR,
    target_dir: Path = DEFAULT_PROMPTS_DIR,
) -> None:
    """把内置 prompt 库同步 (镜像) 到运行时目录 / Mirror the bundled prompt library.

    设计意图 (Why): 让 ``target_dir`` 成为 ``source_dir`` 的精确镜像——新增/变更的资产
    被拷贝过去, 源里已删除的资产被清掉——这样 UI 启动时看到的预设始终与仓库内置的一致。
    Idempotent: 默认 source==target, 此时是无害的自同步。
    返回 None; 仅产生文件系统副作用与日志。
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        logger.info(
            "Prompt library sync skipped: source_dir={} does not exist.",
            source_dir,
        )
        return

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Prompt library sync started: source_dir={} target_dir={}",
        source_dir,
        target_dir,
    )

    # 源端资产清单: {文件名: 路径}; 后面据此决定拷贝哪些、删除目标端哪些多余文件。
    source_assets = {
        asset.name: asset for asset in sorted(source_dir.iterdir()) if _is_prompt_asset(asset)
    }
    copied_count = 0
    for asset_name, source_asset in source_assets.items():
        target_asset = target_dir / asset_name
        # 仅当目标缺失、或 size/mtime 不一致时才重拷, 避免每次启动都全量复制。
        # copy2 保留 mtime, 使得下一次比较能命中"未变更"分支。
        if (
            not target_asset.exists()
            or target_asset.stat().st_size != source_asset.stat().st_size
            or target_asset.stat().st_mtime_ns != source_asset.stat().st_mtime_ns
        ):
            shutil.copy2(source_asset, target_asset)
            copied_count += 1

    removed_count = 0
    # 反向清理: 目标端存在但源端已无的 prompt 资产删掉, 保证镜像精确。
    for target_asset in sorted(target_dir.iterdir()):
        if _is_prompt_asset(target_asset) and target_asset.name not in source_assets:
            target_asset.unlink(missing_ok=True)
            removed_count += 1
    logger.info(
        "Prompt library sync completed: copied_assets={} removed_assets={} "
        "available_assets={}",
        copied_count,
        removed_count,
        len(source_assets),
    )


def _load_prompt_text_map(mapping_file: Path) -> dict[str, str]:
    """解析 ``prompt_text`` 映射文件 / Parse the ``name|transcript`` mapping file.

    文件格式: 每行 ``<preset_name>|<参考文本>``; 空行、以 ``#`` 开头的注释行、
    以及不含 ``|`` 的行都跳过。返回 {preset_name: prompt_text}, 供 voice cloning 用。
    """
    if not mapping_file.is_file():
        return {}

    prompt_text_map: dict[str, str] = {}
    with mapping_file.open(encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            # 只按第一个 "|" 切分, 这样参考文本里本身含 "|" 也不会被破坏。
            name, text = line.split("|", 1)
            prompt_text_map[name.strip()] = text.strip()
    return prompt_text_map


def discover_prompt_presets(
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    mapping_file: Path = DEFAULT_PROMPT_MAPPING_FILE,
) -> tuple[PromptPreset, ...]:
    """扫描目录, 组装出全部音色预设 / Discover all voice presets in a directory.

    把目录下的音频文件与 ``prompt_text`` 映射表 join 起来, 每个音频产出一个 PromptPreset。
    返回不可变 tuple, 顺序稳定 (见下方排序 key), 便于直接喂给 UI 下拉框。
    """
    prompts_dir = Path(prompts_dir)
    if not prompts_dir.is_dir():
        return ()

    prompt_text_map = _load_prompt_text_map(Path(mapping_file))
    # 排序 key: ``stem == "child"`` 排在最后 (False<True), 其余按 stem 字母序; 保证展示顺序确定。
    prompt_audio_paths = [
        audio_path
        for audio_path in sorted(prompts_dir.iterdir(), key=lambda path: (path.stem == "child", path.stem))
        if audio_path.is_file() and audio_path.suffix.lower() in PROMPT_AUDIO_SUFFIXES
    ]
    return tuple(
        PromptPreset(
            name=audio_path.stem,
            audio_path=str(audio_path.resolve()),
            prompt_text=prompt_text_map.get(audio_path.stem, ""),
        )
        for audio_path in prompt_audio_paths
    )


def build_prompt_choice_items(
    prompt_presets: tuple[PromptPreset, ...],
) -> list[tuple[str, str]]:
    """生成 UI 下拉的 (label, value) 列表 / Build dropdown (label, value) items.

    永远在最前面插入一个 "No Preset" 哨兵项 (value=DEFAULT_PROMPT_NONE), 代表"不做
    voice cloning, 用模型默认音色"。其余项 label 与 value 都用预设名。
    """
    return [("No Preset", DEFAULT_PROMPT_NONE), *[(preset.name, preset.name) for preset in prompt_presets]]


def resolve_default_prompt_selection(
    prompt_presets: tuple[PromptPreset, ...],
    default_prompt_name: str = DEFAULT_PROMPT_NAME,
) -> tuple[str, str | None, str]:
    """决定 UI 初始选中的音色 / Resolve the initially-selected preset.

    返回 (selected_name, audio_path | None, prompt_text)。
    - 无任何预设时, 退回到 "No Preset" 哨兵, audio_path=None。
    - 期望的 default_prompt_name 不存在时, 回退到第一个预设 (容错, 不抛错)。
    """
    if not prompt_presets:
        return DEFAULT_PROMPT_NONE, None, ""

    preset_by_name = {preset.name: preset for preset in prompt_presets}
    # 名字命中则用它, 否则 fallback 到第一个预设, 保证总有一个可用默认。
    selected_name = default_prompt_name if default_prompt_name in preset_by_name else prompt_presets[0].name
    selected_preset = preset_by_name[selected_name]
    return selected_name, selected_preset.audio_path, selected_preset.prompt_text


def resolve_prompt_selection(
    prompt_name: str,
    prompt_presets: tuple[PromptPreset, ...],
) -> tuple[str | None, str]:
    """把 UI 选中的预设名翻译成 (audio_path, prompt_text) / Resolve a chosen preset name.

    返回 (audio_path | None, prompt_text)。哨兵值 DEFAULT_PROMPT_NONE 或未匹配到任何
    预设时, 返回 (None, "") 表示不做 voice cloning。
    """
    if prompt_name == DEFAULT_PROMPT_NONE:
        return None, ""

    for preset in prompt_presets:
        if preset.name == prompt_name:
            return preset.audio_path, preset.prompt_text
    return None, ""


def discover_local_model_choices(repo_root: Path = REPO_ROOT) -> list[str]:
    """枚举本地已下载的模型权重目录 / List locally-available model checkpoints.

    约定: 权重放在 ``pretrained_models/**/model/`` 下 (含 config + DiT/AR/AudioVAE 权重)。
    返回这些目录相对仓库根的 POSIX 路径, 排序后供 UI 下拉与默认模型选取。无则返回 []。
    """
    model_root = Path(repo_root) / "pretrained_models"
    if not model_root.is_dir():
        return []
    return sorted(
        path.relative_to(repo_root).as_posix()
        for path in model_root.glob("**/model")
        if path.is_dir()
    )


def resolve_model_name_or_path(model_name_or_path: str, repo_root: Path = REPO_ROOT) -> str:
    """把用户给的模型标识解析成可加载路径 / Resolve a model id into a loadable path.

    解析优先级 (resolution order):
        1) 当作绝对/用户路径 (展开 ``~``), 存在则返回其绝对路径;
        2) 当作仓库相对路径 (repo_root/<x>), 存在则返回绝对路径;
        3) 都不匹配则原样返回——交给 runtime/HF 当作 hub model id 处理。
    """
    normalized = model_name_or_path.strip()
    if not normalized:
        raise ValueError("model_name_or_path 不能为空。")

    direct_path = Path(normalized).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())

    repo_relative_path = Path(repo_root) / normalized
    if repo_relative_path.exists():
        return str(repo_relative_path.resolve())

    # 既非本地绝对路径也非仓库相对路径 → 视作 HF hub id, 原样下传。
    return normalized


def default_model_name_or_path(repo_root: Path = REPO_ROOT) -> str:
    """挑一个默认模型 / Pick a default model when none is specified.

    取 discover_local_model_choices 的第一个 (排序后最靠前) 作为默认; 没有任何本地模型
    时返回空串, 由上层 (build_gradio_app_config) 决定是否报错要求显式传 --model-name-or-path。
    """
    discovered = discover_local_model_choices(repo_root=repo_root)
    if not discovered:
        return ""
    return discovered[0]


@dataclass(frozen=True)
class GradioAppConfig:
    """Gradio 应用的不可变配置快照 / Immutable config snapshot for the Gradio app.

    由 build_gradio_app_config() 一次性构建并固化, 之后服务/UI 全程只读它。
    分两类字段 (Two groups of fields):
        - 运行时行为: host/port/execution_mode/precision/optimize/output_dir/
          output_retention_count/max_generate_length —— 决定模型怎么加载与生成。
        - UI 初值: 一堆 ``default_*`` 与已发现的 prompt_presets / local_model_choices ——
          只用来给前端控件填初始值, 不影响后端推理逻辑。
    frozen=True 保证跨线程共享时不可变, 配合 GradioAppService 内部锁是线程安全的。
    """

    host: str
    port: int
    execution_mode: ExecutionMode
    precision: str
    optimize: bool
    output_dir: Path
    prompts_dir: Path
    output_retention_count: int
    max_generate_length: int
    default_model_name_or_path: str
    prompt_presets: tuple[PromptPreset, ...]
    default_prompt_name: str
    default_prompt_audio_path: str | None
    default_prompt_text: str
    default_precision: str
    default_num_steps: int
    default_guidance_scale: float
    default_speaker_scale: float
    default_max_generate_length: int
    local_model_choices: tuple[str, ...]
    repo_root: Path = REPO_ROOT


def build_gradio_app_config(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    execution_mode: ExecutionMode = DEFAULT_EXECUTION_MODE,
    precision: str = DEFAULT_PRECISION,
    optimize: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_retention_count: int = DEFAULT_OUTPUT_RETENTION,
    max_generate_length: int = DEFAULT_MAX_GENERATE_LENGTH,
    model_name_or_path: str | None = None,
    default_prompt_name: str = DEFAULT_PROMPT_NAME,
    default_precision: str = DEFAULT_PRECISION,
    default_num_steps: int = DEFAULT_NUM_STEPS,
    default_guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    default_speaker_scale: float = DEFAULT_SPEAKER_SCALE,
    default_max_generate_length: int = DEFAULT_MAX_GENERATE_LENGTH,
    repo_root: Path = REPO_ROOT,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    prompt_source_dir: Path = DEFAULT_PROMPT_SOURCE_DIR,
) -> GradioAppConfig:
    """组装并校验一份 GradioAppConfig / Build & validate the app config.

    职责 (What it does), 按顺序:
        1) 同步内置 prompt 库到运行目录;
        2) 发现本地模型与 prompt 预设, 解析默认选中项;
        3) 选定要加载的模型 (显式传入优先, 否则取本地第一个);
        4) 校验 execution_mode / max_generate_length / precision 等关键参数, 不合法即抛错;
        5) 返回一份不可变配置 (含运行时参数 + UI 初值)。
    全部仅做装配与校验, **不**触发模型加载——真正的 from_pretrained 推迟到首个请求 (懒加载)。
    """
    # 先把内置 prompt 资产镜像到运行目录, 再去发现可用预设。
    sync_default_prompt_library(
        source_dir=prompt_source_dir,
        target_dir=prompts_dir,
    )
    discovered_models = discover_local_model_choices(repo_root=repo_root)
    prompt_presets = discover_prompt_presets(
        prompts_dir=prompts_dir,
        mapping_file=prompts_dir / "prompt_text",
    )
    resolved_default_prompt_name, default_prompt_audio_path, default_prompt_text = (
        resolve_default_prompt_selection(
            prompt_presets,
            default_prompt_name=default_prompt_name,
        )
    )
    # 显式传入的模型优先 (去空白); 未传则回退到自动发现的默认模型。
    selected_model_name_or_path = (
        model_name_or_path.strip()
        if model_name_or_path is not None
        else default_model_name_or_path(repo_root=repo_root)
    )
    # 下面三段是 fail-fast 校验: 没模型 / 模式非法 / 长度非正 / 精度空, 都在装配期直接拦截。
    if not selected_model_name_or_path:
        raise ValueError("No default model found. Please pass --model-name-or-path.")
    if execution_mode not in ("generate", "generate_stream"):
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
    resolved_max_generate_length = int(max_generate_length)
    if resolved_max_generate_length <= 0:
        raise ValueError("max_generate_length must be positive.")
    resolved_precision = precision.strip() or DEFAULT_PRECISION  # 空串回退到默认精度 bfloat16
    logger.info(
        "Gradio app config prepared: host={} port={} output_dir={} "
        "output_retention_count={} max_generate_length={} execution_mode={} precision={} optimize={} "
        "default_model_name_or_path={} prompt_preset_count={} language_count={} local_model_choice_count={}",
        host,
        port,
        output_dir,
        output_retention_count,
        resolved_max_generate_length,
        execution_mode,
        resolved_precision,
        bool(optimize),
        selected_model_name_or_path,
        len(prompt_presets),
        len(SUPPORTED_LANGUAGE_CODE_BY_NAME),
        len(discovered_models),
    )
    return GradioAppConfig(
        host=host,
        port=int(port),
        execution_mode=execution_mode,
        precision=resolved_precision,
        optimize=bool(optimize),
        output_dir=Path(output_dir),
        prompts_dir=Path(prompts_dir),
        output_retention_count=int(output_retention_count),
        max_generate_length=resolved_max_generate_length,
        default_model_name_or_path=selected_model_name_or_path,
        prompt_presets=prompt_presets,
        default_prompt_name=resolved_default_prompt_name,
        default_prompt_audio_path=default_prompt_audio_path,
        default_prompt_text=default_prompt_text,
        default_precision=default_precision,
        default_num_steps=int(default_num_steps),
        default_guidance_scale=float(default_guidance_scale),
        default_speaker_scale=float(default_speaker_scale),
        default_max_generate_length=int(default_max_generate_length),
        local_model_choices=tuple(discovered_models),
        repo_root=repo_root,
    )


@dataclass(frozen=True)
class SynthesisRequest:
    """一次合成请求的完整参数 / All parameters of one synthesis request.

    采样/引导相关字段对应的 ML 概念 (mapping to ML concepts):
        - prompt_audio_path / prompt_text : voice cloning 的参考音色 + 其转写; 二者一起作为
          condition (x-vector + 参考 latent) 喂给模型。
        - ode_method / num_steps          : flow-matching 声学头解 ODE 时的 solver 与步数;
          步数越多越精细但越慢 (典型 euler + 10 步)。
        - guidance_scale                  : classifier-free guidance (CFG) 强度, 控制对文本条件
          的贴合程度 (1.0=无引导)。
        - speaker_scale                   : 对说话人 (speaker) 条件单独的引导强度, 控制音色相似度。
        - normalize_text                  : 是否走文本归一化 (数字/符号读法等)。
        - seed                            : 复现实验用随机种子。
    frozen=True: 请求一旦构造即不可变, 便于日志与缓存 key 推导。
    """

    model_name_or_path: str
    text: str
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    execution_mode: ExecutionMode = DEFAULT_EXECUTION_MODE
    template_name: str = "tts"
    language: str | None = None
    ode_method: str = DEFAULT_ODE_METHOD
    num_steps: int = DEFAULT_NUM_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    speaker_scale: float = DEFAULT_SPEAKER_SCALE
    normalize_text: bool = False
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class SynthesisResult:
    """一次合成的结果 / The result of one synthesis call.

    字段: audio_path = 落盘 wav 的绝对路径; metrics = 结构化指标 (request_id/rtf/耗时/
    采样率等, 见 GradioAppService.generate); status = 给 UI 直接展示的中文状态串。
    """

    audio_path: str
    metrics: dict[str, Any]
    status: str


class GradioAppService:
    """线程安全的推理服务对象 / Thread-safe inference service wrapping the runtime.

    职责 (Responsibilities):
        - 懒加载并缓存单个 DotsTtsRuntime (按 resolved 模型路径判断是否需要重载);
        - 用一把互斥锁串行化所有生成调用 (模型/采样状态不可重入, 且共享显存);
        - 校验/归一化请求, 调度流式或非流式生成, 把 waveform 写成 wav 并算 metrics;
        - 维护输出目录 (建目录 + 按保留数清理旧 wav)。

    设计要点 (Design notes):
        懒加载使得装配配置时不必占显存; 缓存命中 (同一模型) 时直接复用 runtime, 避免反复
        from_pretrained。``_lock`` 保证即便 Gradio 并发回调进来, 底层 KV cache / 采样器
        也是一次只服务一个请求。
    """

    def __init__(self, config: GradioAppConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        # 串行化生成调用的互斥锁; runtime 内部有 KV cache 等可变状态, 不可并发进入。
        self._lock = threading.Lock()
        # 懒加载的 runtime 与其对应的 resolved 模型路径 (用作缓存命中判据)。
        self._runtime: DotsTtsRuntime | None = None
        self._runtime_model_name_or_path: str | None = None
        logger.info(
            "Gradio service initialized: output_dir={} default_model_name_or_path={} "
            "output_retention_count={} max_generate_length={} execution_mode={} precision={} optimize={}",
            self.config.output_dir,
            self.config.default_model_name_or_path,
            self.config.output_retention_count,
            self.config.max_generate_length,
            self.config.execution_mode,
            self.config.precision,
            self.config.optimize,
        )

    def metadata(self) -> dict[str, Any]:
        """汇报服务当前状态 / Report current service state for the UI / health check.

        把配置项 + 运行时加载状态 (是否已加载、加载了哪个模型) + UI 初值 + 支持的语言/模板
        打包成一个普通 dict, 供前端展示或健康检查读取。纯只读, 不触发加载。
        """
        return {
            "repo_root": str(self.config.repo_root),
            "default_model_name_or_path": self.config.default_model_name_or_path,
            "local_model_choices": list(self.config.local_model_choices),
            "prompts_dir": str(self.config.prompts_dir),
            "prompt_preset_names": [preset.name for preset in self.config.prompt_presets],
            "default_prompt_name": self.config.default_prompt_name,
            "output_dir": str(self.config.output_dir),
            "output_retention_count": self.config.output_retention_count,
            "configured_max_generate_length": self.config.max_generate_length,
            "configured_execution_mode": self.config.execution_mode,
            "configured_precision": self.config.precision,
            "optimize": self.config.optimize,
            "loaded_model_name_or_path": self._runtime_model_name_or_path,
            "loaded_max_generate_length": (
                self.config.max_generate_length if self._runtime is not None else None
            ),
            "loaded_precision": (
                self.config.precision if self._runtime is not None else None
            ),
            "model_loaded": self._runtime is not None,
            "host": self.config.host,
            "port": self.config.port,
            "default_precision": self.config.default_precision,
            "default_num_steps": self.config.default_num_steps,
            "default_guidance_scale": self.config.default_guidance_scale,
            "default_speaker_scale": self.config.default_speaker_scale,
            "default_max_generate_length": self.config.default_max_generate_length,
            # [1:] 跳过列表首项 (通常是 "Auto/自动检测" 哨兵), 只暴露真实可选语言。
            "supported_languages": build_language_choice_items()[1:],
            "supported_template_names": list(GRADIO_SYNTHESIS_MODE_TEMPLATE_NAMES),
        }

    def _get_runtime(
        self,
        model_name_or_path: str,
    ) -> tuple[DotsTtsRuntime, str]:
        """懒加载/缓存命中地拿到 runtime / Lazily load (or reuse) the runtime.

        返回 (runtime, resolved_model_name_or_path)。先把入参解析成确定路径, 若与已缓存
        runtime 的路径不同 (或尚未加载), 就 from_pretrained 重新加载并更新缓存; 否则直接复用。
        前置条件: 调用方已持有 ``self._lock`` (本方法不自加锁)。
        """
        resolved_model_name_or_path = resolve_model_name_or_path(
            model_name_or_path,
            repo_root=self.config.repo_root,
        )
        # 缓存判据: 未加载过, 或请求的 (resolved) 模型与当前缓存的不是同一个 → 需要重载。
        if (
            self._runtime is None
            or self._runtime_model_name_or_path != resolved_model_name_or_path
        ):
            logger.info(
                "Gradio runtime cache miss: requested_model={} resolved_model={} "
                "max_generate_length={} execution_mode={} precision={} optimize={}",
                model_name_or_path,
                resolved_model_name_or_path,
                self.config.max_generate_length,
                self.config.execution_mode,
                self.config.precision,
                self.config.optimize,
            )
            self._runtime = DotsTtsRuntime.from_pretrained(
                resolved_model_name_or_path,
                precision=self.config.precision,
                optimize=self.config.optimize,
                max_generate_length=self.config.max_generate_length,
            )
            self._runtime_model_name_or_path = resolved_model_name_or_path
        else:
            logger.info(
                "Gradio runtime cache hit: requested_model={} resolved_model={} "
                "max_generate_length={} execution_mode={} precision={} optimize={}",
                model_name_or_path,
                resolved_model_name_or_path,
                self.config.max_generate_length,
                self.config.execution_mode,
                self.config.precision,
                self.config.optimize,
            )
        return self._runtime, resolved_model_name_or_path

    def _build_stream_request_id(
        self,
        runtime: DotsTtsRuntime,
        request: SynthesisRequest,
    ) -> str:
        """为流式生成复算出与 runtime 一致的 request_id (fid) / Recompute the stream fid.

        为什么需要 (Why): runtime.generate() 会在内部自己算 request_id 并放进返回 dict;
        但 runtime.generate_stream() 只产出音频块, 不回传 fid。为了让流式路径也能得到与
        非流式完全一致的 request_id, 这里**重放 runtime 内部的归一化步骤**——文本归一化、
        prompt 文本归一化、(无参考文本时) 给正文 attach language tag——再调用同一个
        ``_build_request_id`` 私有方法。因此结果与 runtime 内部口径严格一致。
        复用私有方法 (``_process_text`` 等, 带 # noqa: SLF001) 是刻意为之, 以避免归一化逻辑
        分叉。返回归一化输入派生出的稳定 id 字符串。
        """
        # 复用 runtime 的文本归一化, 同时拿到推断出的 language (可能由文本自动检测得到)。
        normalized_text, normalized_language = runtime._process_text(  # noqa: SLF001
            request.text,
            language=request.language,
            normalize=request.normalize_text,
        )
        normalized_prompt_text = runtime._process_prompt_text(  # noqa: SLF001
            request.prompt_text,
            language=normalized_language,
        )
        # 仅当指定了语言、且**没有**参考文本时, 才把 language tag 直接拼到正文上——
        # 有 prompt_text 时语言信息已随参考文本进入条件, 无需重复打标。
        if normalized_language is not None and not normalized_prompt_text:
            from dots_tts.utils.text import attach_language_tag  # noqa: PLC0415

            normalized_text = attach_language_tag(
                normalized_text,
                normalized_language,
            )
        request_id_kwargs = {
            "text": normalized_text,
            "prompt_audio_path": request.prompt_audio_path,
            "prompt_text": normalized_prompt_text,
            "template_name": request.template_name,
        }
        # 仅在确实有语言时才把 language 传进去, 与 runtime._build_request_id 的签名口径对齐。
        if normalized_language is not None:
            request_id_kwargs["language"] = normalized_language
        return runtime._build_request_id(  # noqa: SLF001
            **request_id_kwargs,
        )

    @staticmethod
    def _build_runtime_generate_kwargs(request: SynthesisRequest) -> dict[str, Any]:
        """把 SynthesisRequest 摊平成 runtime.generate(**kwargs) 的关键字参数 / Flatten request to runtime kwargs.

        只挑 runtime 生成接口需要的字段 (text/prompt/采样参数等); model_name_or_path、seed、
        execution_mode 不在此列 (它们由服务层在外层处理)。language 为 None 时**省略**该键,
        让 runtime 走其自动检测分支, 而非显式传 None。
        """
        runtime_kwargs: dict[str, Any] = {
            "text": request.text,
            "prompt_audio_path": request.prompt_audio_path,
            "prompt_text": request.prompt_text,
            "template_name": request.template_name,
            "ode_method": request.ode_method,
            "num_steps": request.num_steps,
            "guidance_scale": request.guidance_scale,
            "speaker_scale": request.speaker_scale,
            "normalize_text": request.normalize_text,
        }
        # language 为 None 时不写入键 → runtime 走自动语言检测; 非 None 才显式指定。
        if request.language is not None:
            runtime_kwargs["language"] = request.language
        return runtime_kwargs

    def _run_stream_generation(
        self,
        runtime: DotsTtsRuntime,
        request: SynthesisRequest,
    ) -> dict[str, Any]:
        """跑流式生成并拼成整段音频 + 指标 / Drive streaming generation, concat to one clip.

        runtime.generate_stream() 是个产出音频块 (chunk) 的生成器; 这里把每块 detach 到
        float32 CPU 后收集, 再沿最后一维 (时间/样本维) ``torch.cat`` 成完整波形, 顺带计算
        耗时与 RTF (real-time factor = 推理墙钟时间 / 音频时长, <1 表示快于实时)。
        因 generate_stream 不回传 fid, 这里另用 _build_stream_request_id 补算。
        返回 dict, 键与 runtime.generate() 对齐 (fid/audio/sample_rate/time_used/rtf) 外加
        chunk_count, 以便上层 generate/warmup 统一处理。
        """
        start_time = time.time()
        # 逐块拉取并搬到 CPU/float32; detach 切断计算图 (推理无需梯度), 降低显存占用。
        chunks = [
            chunk.detach().float().cpu()
            for chunk in runtime.generate_stream(
                **self._build_runtime_generate_kwargs(request)
            )
        ]
        if not chunks:
            raise ValueError("流式生成未返回任何音频块。")

        # 沿最后一维 (样本维) 拼接所有块 → (..., T_total) 的整段波形。
        audio = torch.cat(chunks, dim=-1)
        elapsed_seconds = time.time() - start_time
        audio_seconds = audio.shape[-1] / runtime.sample_rate
        # 防 0 除: 空音频时 RTF 记为 +inf。
        rtf = elapsed_seconds / audio_seconds if audio_seconds > 0 else float("inf")
        return {
            "fid": self._build_stream_request_id(runtime, request),
            "audio": audio,
            "sample_rate": runtime.sample_rate,
            "time_used": elapsed_seconds,
            "rtf": rtf,
            "chunk_count": len(chunks),
        }

    def warmup(self, text: str | None = None) -> dict[str, Any]:
        """预热: 加载模型并跑一次小合成 / Warm up: load model + run one synthesis.

        为什么 (Why): 首次推理要付出模型加载 + CUDA kernel/JIT 编译 + (可能的) torch.compile
        的一次性开销; 在服务起来时先用固定文本/种子跑一遍, 把这些代价提前到启动阶段, 让真实
        用户请求一上来就快。同时验证 runtime 端到端可用。
        全程持锁串行执行; 返回 metrics dict (含 request_id/rtf/耗时等), 不写 wav。
        失败时记录 exception 并向上抛, 让启动尽早暴露问题。
        """
        # 优先用调用方文本, 否则退到内置 DEFAULT_WARMUP_TEXT。
        warmup_text = (text or "").strip() or DEFAULT_WARMUP_TEXT.strip()
        if not warmup_text:
            raise ValueError("DEFAULT_WARMUP_TEXT 不能为空。")

        with self._lock:
            logger.info(
                "Gradio warmup requested: default_model_name_or_path={} execution_mode={} precision={} optimize={} seed={}",
                self.config.default_model_name_or_path,
                self.config.execution_mode,
                self.config.precision,
                self.config.optimize,
                DEFAULT_SEED,
            )
            try:
                seed_everything(DEFAULT_SEED)  # 固定种子, 预热结果可复现且无副作用差异。
                runtime, resolved_model_name_or_path = self._get_runtime(
                    self.config.default_model_name_or_path,
                )
                warmup_request = SynthesisRequest(
                    model_name_or_path=self.config.default_model_name_or_path,
                    text=warmup_text,
                    execution_mode=self.config.execution_mode,
                    template_name="tts",
                    ode_method=DEFAULT_ODE_METHOD,
                    num_steps=self.config.default_num_steps,
                    guidance_scale=self.config.default_guidance_scale,
                    speaker_scale=self.config.default_speaker_scale,
                    normalize_text=False,
                    seed=DEFAULT_SEED,
                )
                request_id = self._build_stream_request_id(runtime, warmup_request)
                # 按配置的执行模式分两条路: 流式路径自带耗时统计; 非流式需在此手动计时并补
                # time_used/chunk_count, 使两条路返回的 result 字段结构统一。
                if self.config.execution_mode == "generate_stream":
                    result = self._run_stream_generation(runtime, warmup_request)
                else:
                    start_time = time.time()
                    result = runtime.generate(**self._build_runtime_generate_kwargs(warmup_request))
                    result["time_used"] = time.time() - start_time
                    result["chunk_count"] = 1
                audio_samples = int(result["audio"].shape[-1])  # 最后一维 = 样本数
            except Exception:
                logger.exception(
                    "Gradio warmup failed: default_model_name_or_path={}",
                    self.config.default_model_name_or_path,
                )
                raise
            audio_seconds = audio_samples / runtime.sample_rate
            metrics = {
                "request_id": request_id,
                "execution_mode": self.config.execution_mode,
                "chunk_count": int(result["chunk_count"]),
                "resolved_model_name_or_path": resolved_model_name_or_path,
                "sample_rate": runtime.sample_rate,
                "elapsed_seconds": round(float(result["time_used"]), 3),
                "audio_seconds": round(float(audio_seconds), 3),
                "rtf": round(float(result["rtf"]), 4),
                "seed": DEFAULT_SEED,
                "text": warmup_text,
            }
            logger.info(
                "Gradio warmup ready: request_id={} execution_mode={} resolved_model_name_or_path={}",
                metrics["request_id"],
                metrics["execution_mode"],
                metrics["resolved_model_name_or_path"],
            )
            return metrics

    def _normalize_request(self, request: SynthesisRequest) -> SynthesisRequest:
        """校验并归一化一次请求 / Validate & normalize an incoming request.

        把来自 UI 的"脏"输入 (空白、空串、非法枚举) 清洗成可信值, 并对非法组合 fail-fast:
        正文不能空; 有 prompt_text 必须配 prompt_audio_path; template_name/language 必须在
        受支持集合内。返回一个全新的 (frozen) SynthesisRequest, 不修改入参。
        这样后续生成路径可以假定输入已合法, 无需再防御。
        """
        normalized_text = request.text.strip()
        if not normalized_text:
            raise ValueError("text 不能为空。")

        # 空串/纯空白一律折叠成 None, 代表"无参考"。
        normalized_prompt_audio_path = request.prompt_audio_path or None
        normalized_prompt_text = (request.prompt_text or "").strip() or None
        # voice cloning 约束: 只有参考文本却没有参考音频是无意义的, 直接拒绝。
        if normalized_prompt_text and not normalized_prompt_audio_path:
            raise ValueError("prompt_text requires prompt_audio_path.")
        normalized_template_name = request.template_name.strip() or "tts"
        if normalized_template_name not in GRADIO_SYNTHESIS_MODE_TEMPLATE_NAMES:
            raise ValueError(
                f"Unsupported template_name={normalized_template_name!r}. "
                f"Expected one of {list(GRADIO_SYNTHESIS_MODE_TEMPLATE_NAMES)}."
            )
        # 语言空 → None (交给自动检测); 非空则必须是受支持的语言 code, 否则报错。
        normalized_language = (request.language or "").strip() or None
        supported_language_codes = set(SUPPORTED_LANGUAGE_CODE_BY_NAME.values())
        if (
            normalized_language is not None
            and normalized_language not in supported_language_codes
        ):
            raise ValueError(
                f"Unsupported language={normalized_language!r}. "
                f"Expected one of {sorted(supported_language_codes)}."
            )

        resolved_seed = int(request.seed)
        return SynthesisRequest(
            model_name_or_path=request.model_name_or_path.strip(),
            text=normalized_text,
            prompt_audio_path=normalized_prompt_audio_path,
            prompt_text=normalized_prompt_text,
            execution_mode=request.execution_mode,
            template_name=normalized_template_name,
            language=normalized_language,
            ode_method=request.ode_method.strip() or DEFAULT_ODE_METHOD,
            num_steps=int(request.num_steps),
            guidance_scale=float(request.guidance_scale),
            speaker_scale=float(request.speaker_scale),
            normalize_text=bool(request.normalize_text),
            seed=resolved_seed,
        )

    def _build_output_path(self) -> Path:
        """生成一个不冲突的输出 wav 路径 / Build a unique output wav path.

        文件名 = 时间戳 + 8 位随机 uuid 片段, 既可读 (按时间排序) 又几乎不会撞名。
        """
        output_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"
        return self.config.output_dir / output_name

    def _cleanup_outputs(self) -> None:
        """按保留数清理旧输出 / Prune old wav outputs beyond the retention count.

        把输出目录里的 wav 按 mtime 倒序 (新→旧) 排, 保留前 N 个, 删掉其余。
        retention_count<=0 视为"不限制/不清理", 直接返回。防止长期运行把磁盘写满。
        """
        if self.config.output_retention_count <= 0:
            return

        wav_files = sorted(
            self.config.output_dir.glob("*.wav"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        removed_count = 0
        # 切片 [N:] 即"保留最新 N 个之外的全部", 逐个删除。
        for stale_file in wav_files[self.config.output_retention_count :]:
            stale_file.unlink(missing_ok=True)
            removed_count += 1
        if removed_count > 0:
            logger.info(
                "Gradio output cleanup completed: removed_files={} retention_limit={}",
                removed_count,
                self.config.output_retention_count,
            )

    @staticmethod
    def _waveform_to_numpy(audio: torch.Tensor):
        """把波形张量转成可写盘的 numpy / Convert a waveform tensor to a writable numpy array.

        输入 audio: 形如 (1, T) 或 (T,) 的张量。detach→float32→cpu→squeeze 后去掉长度为 1
        的批/通道维, 得到 1-D (T,) 波形。若 squeeze 后变成 0 维 (标量), 说明音频为空 → 报错。
        """
        waveform = audio.detach().float().cpu().squeeze()
        # 0 维意味着没有时间样本 (空音频), soundfile 无法写出有效 wav。
        if waveform.ndim == 0:
            raise ValueError("生成音频为空。")
        return waveform.numpy()

    def _write_audio(self, audio: torch.Tensor, sample_rate: int) -> str:
        """把波形写成 wav 并触发清理 / Write the waveform to a wav file, then prune.

        返回写出文件的绝对路径字符串。写完调用 _cleanup_outputs 维持保留上限。
        """
        output_path = self._build_output_path()
        logger.info(
            "Writing synthesized audio: output_path={} sample_rate={} samples={}",
            output_path,
            sample_rate,
            audio.shape[-1],
        )
        sf.write(output_path, self._waveform_to_numpy(audio), sample_rate)
        self._cleanup_outputs()
        logger.info("Synthesized audio written: output_path={}", output_path)
        return str(output_path)

    def generate(self, request: SynthesisRequest) -> SynthesisResult:
        """服务层主入口: 一次完整合成 / Main entry: synthesize one request end-to-end.

        流程 (Pipeline):
            归一化/校验 → 持锁串行 → 固定随机种子 → 取/加载 runtime → 按 execution_mode 走
            流式或非流式生成 → 波形写盘 → 组装 metrics 与中文 status → 返回 SynthesisResult。
        全程在 ``self._lock`` 内, 保证同一时刻只有一个请求在用模型/显存。出错时打印结构化
        诊断日志 (含各采样参数) 再向上抛, 便于复盘。
        """
        normalized_request = self._normalize_request(request)

        with self._lock:
            try:
                seed_everything(normalized_request.seed)  # 按请求种子固定随机性, 保证可复现。
                runtime, resolved_model_name_or_path = self._get_runtime(
                    normalized_request.model_name_or_path,
                )
                logger.info(
                    "Gradio request accepted: resolved_model_name_or_path={} execution_mode={} seed={}",
                    resolved_model_name_or_path,
                    normalized_request.execution_mode,
                    normalized_request.seed,
                )
                # 流式 vs 非流式: 流式自带 fid/rtf/time_used; 非流式 runtime.generate 已含这些,
                # 仅需补 chunk_count=1 让两条路的 result 字段一致, 供下方统一取用。
                if normalized_request.execution_mode == "generate_stream":
                    result = self._run_stream_generation(runtime, normalized_request)
                else:
                    result = runtime.generate(
                        **self._build_runtime_generate_kwargs(normalized_request)
                    )
                    result["chunk_count"] = 1
                audio_path = self._write_audio(result["audio"], result["sample_rate"])
            except Exception:
                logger.exception(
                    "Gradio request failed: model_name_or_path={} execution_mode={} text_len={} has_prompt_audio={} has_prompt_text={} template_name={} language={} "
                    "precision={} ode_method={} num_steps={} guidance_scale={} speaker_scale={} max_generate_length={} "
                    "normalize_text={} seed={}",
                    normalized_request.model_name_or_path,
                    normalized_request.execution_mode,
                    len(normalized_request.text),
                    bool(normalized_request.prompt_audio_path),
                    bool(normalized_request.prompt_text),
                    normalized_request.template_name,
                    normalized_request.language,
                    self.config.precision,
                    normalized_request.ode_method,
                    normalized_request.num_steps,
                    normalized_request.guidance_scale,
                    normalized_request.speaker_scale,
                    self.config.max_generate_length,
                    normalized_request.normalize_text,
                    normalized_request.seed,
                )
                raise
            # 音频时长(秒) = 样本数 / 采样率; 用于 RTF 与展示。
            audio_seconds = result["audio"].shape[-1] / result["sample_rate"]
            metrics = {
                "request_id": result["fid"],
                "execution_mode": normalized_request.execution_mode,
                "chunk_count": int(result["chunk_count"]),
                "template_name": normalized_request.template_name,
                "language": normalized_request.language,
                "resolved_model_name_or_path": resolved_model_name_or_path,
                "sample_rate": result["sample_rate"],
                "elapsed_seconds": round(float(result["time_used"]), 3),
                "audio_seconds": round(float(audio_seconds), 3),
                "rtf": round(float(result["rtf"]), 4),
                "seed": normalized_request.seed,
                "output_path": audio_path,
            }
            logger.info(
                "Gradio request output ready: request_id={} execution_mode={} resolved_model_name_or_path={} output_path={}",
                metrics["request_id"],
                metrics["execution_mode"],
                metrics["resolved_model_name_or_path"],
                metrics["output_path"],
            )
            status = (
                f"完成：{Path(audio_path).name} | "
                f"模式 {metrics['execution_mode']} | "
                f"耗时 {metrics['elapsed_seconds']}s | "
                f"音频 {metrics['audio_seconds']}s | "
                f"RTF {metrics['rtf']}"
            )
            return SynthesisResult(
                audio_path=audio_path,
                metrics=metrics,
                status=status,
            )
