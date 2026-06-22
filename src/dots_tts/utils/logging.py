"""Logging configuration / 日志配置 (loguru-based).

本文件做什么 (What this file does)
----------------------------------
集中配置整个 dots.tts 项目的日志行为，底层用 loguru（而非标准库 logging）。
它提供一个统一入口 :func:`configure_logging`，把日志同时输出到 stderr，
并可选地落盘到日志文件。整个项目其余模块只需 ``from loguru import logger``
直接打日志，sink（输出目的地）/级别/格式则由这里集中决定。

在数据流里的位置 (Where it sits in the data flow)
-------------------------------------------------
这是一个横切的基础设施模块，不参与 TTS 推理/训练的张量计算；它在
CLI / 服务启动早期被调用一次，用来初始化日志，之后被各处复用。
(Cross-cutting infrastructure, not part of the inference/training tensor path;
called once at process startup.)

关键符号清单 (Key symbols)
--------------------------
- ``DEFAULT_LOG_LEVEL``  : 默认日志级别 (default verbosity threshold)。
- ``DEFAULT_LOG_FORMAT`` : 默认日志行格式 (per-line format template)。
- :func:`configure_logging` : 重置并安装 stderr（及可选文件）sink。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

# 默认日志级别；可被函数参数或环境变量覆盖。
# Default verbosity; overridable via the `level` arg or DOTS_TTS_LOG_LEVEL env.
DEFAULT_LOG_LEVEL = "INFO"
# 每行日志的格式模板：时间(毫秒) | 级别(左对齐占 8 列) | 模块:函数:行号 | 正文。
# Per-line format: timestamp(ms) | level(left-padded 8) | name:function:line | message.
DEFAULT_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
    "{name}:{function}:{line} | {message}"
)


def configure_logging(
    *,
    level: str | None = None,
    log_file: str | os.PathLike[str] | None = None,
) -> None:
    """重置并安装 loguru sink / Reset and install loguru sink(s).

    职责 (Responsibility)
    ---------------------
    先清空 loguru 默认/历史 sink，再装一个 stderr sink；若给了 ``log_file``
    就额外再装一个文件 sink。两个 sink 共享同一级别与格式，保证终端和文件
    输出一致。设计成「先 remove 再 add」是为了让重复调用幂等——不会因为多次
    初始化而让同一条日志被打印多遍。
    (Clears existing sinks first so repeated calls are idempotent / non-duplicating.)

    参数 (Parameters)
    -----------------
    level : str | None
        日志级别字符串（如 "DEBUG"/"INFO"）。仅关键字参数。优先级：
        显式 ``level`` > 环境变量 ``DOTS_TTS_LOG_LEVEL`` > ``DEFAULT_LOG_LEVEL``。
        (Verbosity threshold; precedence: arg > env var > default.)
    log_file : str | os.PathLike | None
        可选的日志文件路径；为 falsy 时只输出到 stderr。仅关键字参数。
        (Optional on-disk log path; stderr-only when omitted.)

    返回 (Returns)
    --------------
    None — 仅产生副作用（修改全局 ``logger`` 的 sink 配置）。
    (Side-effecting only; mutates the global logger's sinks.)
    """
    # 解析最终级别：三级回退后统一转大写，匹配 loguru 的级别命名。
    # Resolve level via 3-tier fallback, then upper-case to match loguru's level names.
    resolved_level = (level or os.environ.get("DOTS_TTS_LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper()
    # 清掉所有已存在的 sink（含 loguru 默认 sink），避免重复输出。
    # Drop every existing sink (including loguru's default) to prevent duplicate lines.
    logger.remove()
    # 控制台 sink：写到 stderr。backtrace=True 保留完整异常回溯链；
    # diagnose=False 关闭变量值展开（避免把敏感值/大张量打进日志）；
    # enqueue=False 同步写入（无独立后台线程，简单且无队列开销）。
    # Console sink on stderr: full tracebacks, no variable-value expansion, synchronous.
    logger.add(
        sys.stderr,
        level=resolved_level,
        format=DEFAULT_LOG_FORMAT,
        backtrace=True,
        diagnose=False,
        enqueue=False,
    )
    if log_file:
        # 展开 ~ 等用户路径，并确保父目录存在（exist_ok 避免并发/重复创建报错）。
        # Expand "~" and ensure the parent dir exists before loguru opens the file.
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 文件 sink：配置与 stderr 对齐，额外显式 utf-8 编码以正确写入中文等字符。
        # File sink: mirrors stderr config; explicit utf-8 so non-ASCII text is written correctly.
        logger.add(
            log_path,
            level=resolved_level,
            format=DEFAULT_LOG_FORMAT,
            backtrace=True,
            diagnose=False,
            enqueue=False,
            encoding="utf-8",
        )
