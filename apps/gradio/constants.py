"""Gradio demo 的集中常量定义 / Centralised constants for the Gradio demo.

本文件不含任何逻辑，只把 Gradio 演示 app(`app.py` / `service.py`)用到的
**默认值与路径**集中到一处，方便 fork 拥有者在一个地方调参，而不必在业务代码里到处找
magic number。它处在推理数据流的最外层(用户在浏览器里点按钮 → service 读取这些默认值
→ 调用 dots.tts 运行时做合成),不参与训练。

This module holds no logic — it just centralises the **default values and filesystem
paths** consumed by the Gradio demo (`app.py` / `service.py`), so a fork owner can tune
everything in one place instead of hunting for magic numbers in the business logic. It
sits at the outermost layer of the inference data flow (user clicks in the browser →
service reads these defaults → invokes the dots.tts runtime to synthesise); it has no
role in training.

关键常量分组 / Key groups of constants:
  - 服务进程 / server process: ``DEFAULT_HOST``, ``DEFAULT_PORT``
  - 文件路径 / filesystem paths: ``REPO_ROOT``, ``DEFAULT_OUTPUT_DIR``, ``DEFAULT_LOG_FILE``,
    ``DEFAULT_PROMPTS_DIR`` 及其衍生项 / and its derivatives
  - 推理超参 / inference hyper-parameters: ``DEFAULT_EXECUTION_MODE``, ``DEFAULT_PRECISION``,
    ``DEFAULT_ODE_METHOD``, ``DEFAULT_NUM_STEPS``, ``DEFAULT_GUIDANCE_SCALE``,
    ``DEFAULT_SPEAKER_SCALE``, ``DEFAULT_MAX_GENERATE_LENGTH``, ``DEFAULT_SEED``
  - 默认文案 / default texts: ``DEFAULT_INPUT_TEXT``, ``DEFAULT_WARMUP_TEXT``
  - 默认音色 prompt / default voice prompt: ``DEFAULT_PROMPT_NAME``, ``DEFAULT_PROMPT_NONE``,
    ``PROMPT_AUDIO_SUFFIXES``
"""

from __future__ import annotations

from pathlib import Path

# 以本文件位置向上回溯两级目录得到仓库根 / Resolve the repo root by going two levels up
# from this file: ``apps/gradio/constants.py`` → ``apps/gradio`` → ``apps`` → repo root.
# 用 ``resolve()`` 先转成绝对路径再取 parents,保证无论从哪个工作目录启动都能定位资源。
# ``resolve()`` makes it absolute first so resources are found regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
# ``0.0.0.0`` 监听所有网卡,容器/远程访问可达 / Bind all interfaces so the demo is
# reachable from containers and remote hosts (not only localhost).
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860  # Gradio 约定俗成的默认端口 / Gradio's conventional default port
# 合成出的 .wav 落盘目录 / Directory where synthesised .wav files are written.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apps" / "gradio" / "outputs"
DEFAULT_LOG_FILE = REPO_ROOT / "apps" / "gradio" / "gradio.log"
# 内置示例音色 prompt 所在目录 / Folder holding the bundled example voice prompts.
DEFAULT_PROMPTS_DIR = REPO_ROOT / "apps" / "gradio" / "default_prompts"
# 拷贝用户上传 prompt 时的源目录(默认与内置目录同一处)/ Source dir used when copying
# user-supplied prompts (defaults to the same bundled folder).
DEFAULT_PROMPT_SOURCE_DIR = DEFAULT_PROMPTS_DIR
# prompt 名 → 对应参考文本(ref_text)的映射文件 / File mapping a prompt name to its
# reference transcript (ref_text), needed to condition the model on the prompt audio.
DEFAULT_PROMPT_MAPPING_FILE = DEFAULT_PROMPTS_DIR / "prompt_text"
# 输出目录里最多保留几个历史 .wav,超出按时间清理 / Cap on retained output .wav files;
# older ones are pruned once this count is exceeded.
DEFAULT_OUTPUT_RETENTION = 20
# 运行时调用方式:``generate_stream`` 走 double-streaming 流式逐块产音,
# ``generate`` 一次性整段返回 / Runtime call style: ``generate_stream`` yields audio
# chunk-by-chunk (double-streaming), ``generate`` returns the whole clip at once.
DEFAULT_EXECUTION_MODE = "generate_stream"
# 计算精度 / compute dtype:bfloat16 在新 GPU 上兼顾速度与数值范围。
DEFAULT_PRECISION = "bfloat16"
# flow-matching 声学头的 ODE solver / ODE solver for the flow-matching acoustic head;
# ``euler`` 是最简单的一阶定步长积分器 / Euler is the simplest first-order fixed-step integrator.
DEFAULT_ODE_METHOD = "euler"
# ODE 积分步数:步数越多越精细但越慢;连续潜在(无离散 token)合成靠它把 velocity field
# 积分成 latent / Number of ODE steps: more steps = finer but slower; this integrates the
# velocity field into the continuous latent (no discrete audio tokens).
DEFAULT_NUM_STEPS = 10
# classifier-free guidance (CFG) 强度,>1 增强对文本条件的贴合度 / CFG strength; >1
# pushes the output to follow the text condition more strongly.
DEFAULT_GUIDANCE_SCALE = 1.2
# 说话人条件(CAM++ x-vector)的缩放系数,越大越贴近参考音色 / Scale applied to the
# speaker condition (CAM++ x-vector); larger = closer to the reference timbre.
DEFAULT_SPEAKER_SCALE = 1.5
# AR 主干单次最多生成的 token 数上限,防止失控长输出 / Upper bound on tokens the AR
# backbone may emit per request, guarding against runaway-length generations.
DEFAULT_MAX_GENERATE_LENGTH = 500
# 随机种子,固定后采样可复现 / Random seed; fixing it makes sampling reproducible.
DEFAULT_SEED = 42
DEFAULT_INPUT_TEXT = ""  # 文本框初始为空 / Text box starts empty.
DEFAULT_WARMUP_TEXT = "dots.tts is a 2B-parameter fully continuous, end-to-end autoregressive (AR) text-to-speech system. The backbone pairs a semantic encoder, an LLM, and an autoregressive flow-matching acoustic head over a 48 kHz AudioVAE"
# 下拉框默认选中的内置音色 prompt / Default voice prompt pre-selected in the dropdown.
DEFAULT_PROMPT_NAME = "male_zh"
# “不使用任何 prompt”的哨兵值(走无参考/默认音色合成)/ Sentinel meaning "no prompt"
# (synthesise without a reference, using the model's default voice).
DEFAULT_PROMPT_NONE = "__none__"
# 扫描 prompt 目录时认可的音频扩展名 / Audio extensions recognised when scanning the
# prompt directory for usable reference clips.
PROMPT_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
