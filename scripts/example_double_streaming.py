"""dots.tts double-streaming（双流式）推理示例脚本。

本文件做什么 / What this file does
----------------------------------
一个最小可跑的命令行示例，演示如何使用 ``DotsTtsRuntimeDoubleStreaming`` 的
**double-streaming session API**：把一段文本先 tokenize 成 text token 序列，再
**逐 token** 喂进会话；每喂一个 token 就可能即时拿回一个 audio chunk（边喂边出音），
最后把所有 chunk 在时间维拼接、写成一个 wav 文件。这模拟"上游 LLM 逐 token 产出文本、
TTS 同步逐 token 出语音"的边想边说场景。

在推理数据流里的位置 / Position in the inference pipeline
--------------------------------------------------------
本脚本是 ``src/dots_tts/runtime_double_streaming.py`` 的调用方/使用范例，处于整条
推理链路的最外层。底层链路为：
``text token -> push_text_token() 驱动状态机 -> LLM（autoregressive 主干）->
flow-matching DiT 声学头（velocity field + ODE solver + CFG）-> 连续 latent patch ->
BigVGAN AudioVAE 流式 vocoder -> 波形 chunk``。本文件只负责"喂 token、收 chunk、拼波形"，
真正的解码状态机在 ``DoubleStreamingSession`` 内。

调用契约（与底层会话一致）/ Call contract
------------------------------------------
``start_double_streaming() -> 多次 push_text_token() -> finish_text()``。
``push_text_token`` 每步返回 0 或 1 个 audio chunk（可能为 ``None``，表示 vocoder
本步尚未攒够一个输出窗口，需跳过继续）；``finish_text`` 是 generator，喂入 text-end
标记后把 EOS 之后的音频尾部 flush 出来。

关键函数 / Key functions
------------------------
- ``parse_args``：解析命令行参数（模型路径、文本、采样/CFG/EOS 等超参）。
- ``_prepare_text``：文本预处理（strip + 可选 ``normalize_text``）。
- ``main``：装配 runtime、tokenize、跑 double-streaming 主循环、拼接并写 wav。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# 把仓库根与 src/ 加入 sys.path，使脚本无需安装即可 import ``dots_tts`` 包（源码布局）。
for import_root in (REPO_ROOT, SRC_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from dots_tts.utils.logging import configure_logging  # noqa: E402
from dots_tts.runtime_double_streaming import (  # noqa: E402
    DotsTtsRuntimeDoubleStreaming,
)
from dots_tts.utils.text import normalize_text  # noqa: E402
from dots_tts.utils.util import seed_everything  # noqa: E402


def parse_args(argv=None):
    """解析命令行参数，返回 ``argparse.Namespace``。

    参数分两类：**模型/加载**（``--model-name-or-path`` / ``--revision`` /
    ``--cache-dir`` / ``--precision`` / ``--optimize``）与 **采样/声学头超参**
    （``--ode-method`` / ``--num-steps`` flow-matching ODE solver 设置、
    ``--guidance-scale`` classifier-free guidance (CFG) 强度、``--eos-threshold``
    finish_text 尾部解码的 EOS 阈值、``--max-generate-length`` 最大 audio patch 数）。
    ``--prompt-audio`` 给 reference audio 做 x-vector 说话人/音色条件（仅 ref_audio）。
    """
    parser = argparse.ArgumentParser(
        description="Temporary example for dots.tts double streaming session API."
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Local pretrained directory or Hugging Face repo id",
    )
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--output", default="double_streaming.wav", help="Output wav path")
    parser.add_argument(
        "--prompt-audio",
        default=None,
        help="Optional reference audio for ref_audio_only speaker conditioning",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache dir")
    parser.add_argument("--precision", default="bfloat16", help="Inference precision")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable inference optimization and warmup",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument("--ode-method", default="euler", help="ODE solver method")
    parser.add_argument("--num-steps", type=int, default=10, help="Diffusion sampling steps")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.2,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--eos-threshold",
        type=float,
        default=0.8,
        help="EOS stop threshold for finish_text() tail decode",
    )
    parser.add_argument(
        "--max-generate-length",
        type=int,
        default=500,
        help="Maximum number of decoded audio patches in double streaming",
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Normalize text before tokenizer encode",
    )
    return parser.parse_args(argv)


def _prepare_text(text: str, *, normalize: bool) -> str:
    """文本入模前的轻量预处理：去首尾空白，可选文本归一化。

    ``normalize=True`` 时调用 ``normalize_text``（数字/标点/全半角等规整，利于
    tokenizer 切分一致）。预处理后若为空串则报错——空文本无法合成。返回清洗后的字符串。
    """
    prepared = text.strip()
    if normalize:
        prepared = normalize_text(prepared)
    if not prepared:
        raise ValueError("Input text is empty after preprocessing.")
    return prepared


def main(argv=None):
    """脚本主流程：装配 runtime -> tokenize 文本 -> 跑 double-streaming -> 拼波形写 wav。

    流程 / Pipeline
    ---------------
    1. 配置日志、解析参数、固定随机种子（``seed_everything`` 让采样可复现）。
    2. ``from_pretrained`` 加载 ``DotsTtsRuntimeDoubleStreaming``（权重 + tokenizer +
       vocoder），并把文本 tokenize 成 text token id 列表。
    3. ``start_double_streaming`` 开一个会话，逐 token ``push_text_token`` 收 audio chunk。
    4. ``finish_text`` flush 尾部，把所有 chunk 在时间维 ``cat`` 拼成整段波形写出。

    成功返回 ``None``（脚本以 ``SystemExit(main())`` 收尾，``None`` 即退出码 0）。
    """
    configure_logging()
    args = parse_args(argv)
    seed_everything(args.seed)  # 固定种子：让 flow-matching 采样路径可复现

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    runtime = DotsTtsRuntimeDoubleStreaming.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        precision=args.precision,
        optimize=args.optimize,
        max_generate_length=args.max_generate_length,
    )
    prepared_text = _prepare_text(args.text, normalize=args.normalize_text)
    # add_special_tokens=False：本示例手动驱动流式前缀与 text-end 标记（由会话内部注入），
    # 这里只要纯文本 token，不让 tokenizer 自动加 BOS/EOS 等特殊符。
    text_token_ids = runtime.model.tokenizer.encode(
        prepared_text,
        add_special_tokens=False,
    )
    if not text_token_ids:
        raise ValueError("Tokenizer produced no text tokens.")

    logger.info(
        "Double streaming example started: text_len={} text_token_count={} output={}",
        len(prepared_text),
        len(text_token_ids),
        output_path,
    )

    # 开会话：此时已注入采样/CFG/EOS 超参；若给了 prompt_audio 则抽 x-vector 说话人条件。
    session = runtime.start_double_streaming(
        prompt_audio_path=args.prompt_audio,
        ode_method=args.ode_method,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        eos_threshold=args.eos_threshold,
    )

    chunks: list[torch.Tensor] = []
    # double-streaming 主循环：逐 token 喂入、即时收音频。每步至多回一个 chunk，
    # 为 None 表示 vocoder 本步未攒够输出窗口（正常，跳过继续喂下一个 token）。
    for index, token_id in enumerate(text_token_ids, start=1):
        chunk = session.push_text_token(token_id)
        logger.info(
            "Double streaming step: token_index={} token_id={} emitted_audio={}",
            index,
            token_id,
            chunk is not None,
        )
        if chunk is not None:
            # detach + 搬到 CPU：切断计算图并离开显存，避免拼接时累积占用 GPU。
            chunks.append(chunk.detach().cpu())

    # 收尾：finish_text 是 generator，喂 text-end 标记后把 EOS 之后的尾部音频 flush 出来。
    for chunk in session.finish_text():
        chunks.append(chunk.detach().cpu())

    if not chunks:
        raise RuntimeError("Double streaming produced no audio chunks.")

    # 沿时间维（最后一维 samples）拼接所有 chunk，得到整段波形。
    audio = torch.cat(chunks, dim=-1)
    # 写 wav：squeeze 掉 batch/channel 单维成 1-D，转 float numpy；采样率取自 runtime。
    sf.write(
        output_path,
        audio.float().squeeze().numpy(),
        runtime.sample_rate,
    )
    logger.info(
        "Double streaming example completed: output={} chunk_count={} samples={}",
        output_path,
        len(chunks),
        audio.shape[-1],
    )


if __name__ == "__main__":
    raise SystemExit(main())
