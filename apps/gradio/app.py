"""dots.tts 的 Gradio Web Demo 前端 / Gradio web demo front-end for dots.tts.

本文件做什么 (What this file does)
---------------------------------
搭建一个浏览器里的交互式语音合成 playground：让用户上传参考音频 (prompt audio)
+ 转写文本 (prompt text) 做 voice cloning，输入待合成文本，调几个推理超参
(Num Steps、Guidance Scale 等)，点 Generate 就能听到合成结果。本文件只负责
**界面层 (UI layer)**：声明 Gradio 组件、布局、CSS 主题，以及把组件值打包成
``SynthesisRequest`` 交给 service 层执行的回调函数。真正的模型加载与推理在
``apps/gradio/service.py`` 的 ``GradioAppService`` 里，acoustic 推理再下沉到
``dots_tts.runtime.DotsTtsRuntime``。

在整条数据流里的位置 (Position in the data flow)
-----------------------------------------------
    用户交互 (本文件 UI)
        -> SynthesisRequest (打包 UI 控件值)
        -> GradioAppService.generate (service.py：归一化/加锁/拿 runtime)
        -> DotsTtsRuntime.generate / generate_stream
            (Qwen2.5 AR 主干 + flow-matching DiT acoustic head + BigVGAN AudioVAE 解码)
        -> .wav 文件路径 + metrics
        -> 回填到 UI 的 Audio / JSON 组件

关键超参与对应 ML 概念 (Key knobs -> ML concepts)
------------------------------------------------
- Num Steps        : flow-matching ODE solver 的离散步数 (步数越多越精，越慢)。
- Guidance Scale   : classifier-free guidance (CFG) 强度，放大文本条件的影响。
- Speaker Scale    : 说话人 x-vector (CAM++) 条件强度，控制音色相似度 (仅 Debug 面板可调)。
- ODE Method       : flow-matching 的数值积分方法 (默认 euler)。
- Seed             : 随机种子，固定后采样可复现。

关键函数清单 (Key callables)
---------------------------
- ``build_playground_theme`` : 构造 Gradio Soft 主题对象。
- ``parse_args``             : 解析命令行启动参数 (host/port/精度/默认 UI 值等)。
- ``build_startup_config_panel`` : 渲染只读的“启动固定参数”折叠面板。
- ``build_demo``             : 核心，声明所有 UI 组件、布局、CFG 调试分支与事件绑定。
- ``main``                   : 入口，装配 config/service，预热 (warmup) 后启动服务器。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for import_root in (REPO_ROOT, SRC_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from apps.gradio.constants import (  # noqa: E402
    DEFAULT_EXECUTION_MODE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HOST,
    DEFAULT_INPUT_TEXT,
    DEFAULT_LOG_FILE,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_NUM_STEPS,
    DEFAULT_ODE_METHOD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_RETENTION,
    DEFAULT_PORT,
    DEFAULT_PRECISION,
    DEFAULT_PROMPT_NAME,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_SCALE,
)

if TYPE_CHECKING:
    import gradio as gr

# 环境变量开关：DEBUG_GRADIO=1 时解锁高级调试控件 (SynthesisMode/ODE Method/
# Speaker Scale/Metrics/启动参数面板)；否则这些值用固定的 gr.State 占位，界面只暴露
# 基本控件。注意它还决定 run_synthesis 里 template_name 是走用户选的合成模式还是强制 "tts"。
DEBUG_GRADIO_ENABLED = os.environ.get("DEBUG_GRADIO", "0") == "1"


# 整个 playground 的自定义 CSS：统一字体、蓝色品牌主色 (#6666FF)、卡片/标签样式与
# 响应式宽度。纯样式，不影响逻辑；通过 gr.HTML(<style>...) 注入并再传给 demo.launch(css=...)。
PLAYGROUND_CSS = """
.gradio-container {
    width: min(1600px, calc(100vw - 32px)) !important;
    max-width: none !important;
    margin: 0 auto !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

.gradio-container,
.gradio-container .gradio-container {
    --block-label-background-fill: #CCE5FF;
    --block-label-text-color: #6666FF;
    --block-label-border-color: #99c7ee;
    --block-label-text-weight: 600;
    --block-title-background-fill: #CCE5FF;
    --block-title-text-color: #6666FF;
    --block-title-border-color: #99c7ee;
    --block-title-border-width: var(--block-label-border-width);
    --block-title-radius: var(--block-label-radius);
    --block-title-padding: var(--block-label-padding);
    --block-title-text-size: var(--block-label-text-size);
    --block-title-text-weight: 600;
}

.gradio-container label[data-testid="block-label"],
.gradio-container label[data-testid="block-label"] *,
.gradio-container span[data-testid="block-info"],
.gradio-container span[data-testid="block-info"] * {
    background: #CCE5FF !important;
    border-color: #99c7ee !important;
    color: #6666FF !important;
    fill: #6666FF !important;
    font-family: Verdana, Geneva, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif !important;
    font-style: normal !important;
    font-size: 0.78rem !important;
    line-height: 1.2 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
.gradio-container label[data-testid="block-label"],
.gradio-container span[data-testid="block-info"],
.gradio-container [data-testid="block-title"],
.gradio-container .block-title {
    border: var(--block-label-border-width) solid #99c7ee !important;
    border-top: none !important;
    border-left: none !important;
    border-radius: var(--block-label-radius) !important;
    box-shadow: var(--block-label-shadow) !important;
    padding: var(--block-label-padding) !important;
}
.gradio-container label[data-testid="block-label"],
.gradio-container label[data-testid="block-label"] *,
.gradio-container span[data-testid="block-info"],
.gradio-container span[data-testid="block-info"] *,
.gradio-container [data-testid="block-title"],
.gradio-container [data-testid="block-title"] *,
.gradio-container .block-title,
.gradio-container .block-title * {
    font-weight: 600 !important;
}
.gradio-container .block label > span,
.gradio-container .block label > span *,
.gradio-container .form label > span,
.gradio-container .form label > span *,
.gradio-container label > span:first-child,
.gradio-container label > span:first-child * {
    font-weight: 600 !important;
}
.strong-label [data-testid="block-label"],
.strong-label [data-testid="block-label"] *,
.strong-label span[data-testid="block-info"],
.strong-label span[data-testid="block-info"] *,
.strong-label [data-testid="block-title"],
.strong-label [data-testid="block-title"] *,
.strong-label .block-label,
.strong-label .block-label *,
.strong-label .block-title,
.strong-label .block-title *,
.strong-label label > span:first-child,
.strong-label label > span:first-child * {
    font-weight: 600 !important;
}
.gradio-container .info-text,
.gradio-container .info-text * {
    font-weight: 400 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container [role="textbox"],
.gradio-container [contenteditable="true"] {
    font-weight: 400 !important;
}
.gradio-container label[data-testid="block-label"] > span:first-child {
    display: none !important;
}

.generate-button {
    background: #6666FF !important;
    color: #ffffff !important;
    border: 1px solid #5555ee !important;
    font-family: Verdana, Geneva, sans-serif !important;
}
.generate-button:hover {
    background: #5555ee !important;
}

#playground-banner {
    padding: 0;
    border-radius: 0;
    margin-bottom: 18px;
    background: transparent;
    border: 0;
}
#playground-banner h1 {
    margin: 0 0 4px 0;
    font-size: 1.7rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0;
}
#playground-banner .subtitle {
    margin: 0;
    color: #1e293b;
    font-size: 0.9rem;
}

.info-card {
    padding: 14px 18px;
    border-radius: 8px;
    border: 1px solid #99c7ee;
    border-left: 4px solid #2563eb;
    background: transparent;
    font-size: 0.86rem;
    line-height: 1.55;
    margin-bottom: 16px;
    box-sizing: border-box;
    color: #0f172a;
}
.info-card .card-title,
.info-card .notice-title {
    display: block;
    font-weight: 600;
    font-size: 0.92rem;
    color: #0f172a;
}
.info-card .card-title {
    margin-bottom: 4px;
}
.info-card .notice-title {
    margin-top: 8px;
    margin-bottom: 4px;
}
.info-card ol,
.info-card ul {
    margin: 0;
    padding-left: 18px;
}
.info-card li {
    margin: 2px 0;
}

.main-workspace {
    gap: 18px !important;
    align-items: stretch !important;
}

.prompt-column,
.synthesis-column {
    gap: 14px !important;
}

.control-row,
.settings-slider-row {
    gap: 14px !important;
}

.settings-card {
    margin-top: 2px !important;
}

.generate-button {
    margin-top: 2px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    flex: 0 0 auto !important;
    min-height: 44px !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

.output-audio {
    flex: 0 0 auto !important;
    min-height: 190px !important;
}
.output-audio audio {
    width: 100% !important;
}

@media (max-width: 768px) {
    .gradio-container {
        width: calc(100vw - 20px) !important;
    }

}

"""


def build_playground_theme(gr):
    """构造 Gradio 主题对象 / Build the Gradio Soft theme used by the demo.

    把 slate 灰色系 + Inter 字体封装成一个 ``gr.themes.Soft`` 实例，供
    ``demo.launch(theme=...)`` 使用。``gr`` 作为参数传入而非顶层 import，是为了把
    gradio 这个重依赖延迟到运行时才加载 (见 main 里的 ``import gradio as gr``)。
    """
    return gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="md",
        text_size="md",
        spacing_size="md",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行启动参数 / Parse CLI launch arguments.

    分两类参数：
    1. **运行时固定参数** (host/port/execution-mode/precision/optimize/
       model-name-or-path/max-generate-length 等)——服务启动后不可在 UI 改，
       改了要重启 (对应只读的“启动固定参数”面板)。
    2. **UI 默认值** (``--default-*``)——决定界面控件初始落点 (默认音色、Num Steps、
       Guidance Scale、Speaker Scale 等)，启动后用户仍可在界面里调。

    返回解析后的 ``argparse.Namespace``。``argv=None`` 时读取 ``sys.argv``。
    """
    parser = argparse.ArgumentParser(description="dots.tts Gradio app.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument(
        "--execution-mode",
        choices=("generate", "generate_stream"),
        default=DEFAULT_EXECUTION_MODE,
        help="Runtime execution mode fixed for the app",
    )
    parser.add_argument(
        "--precision",
        default=DEFAULT_PRECISION,
        help="Inference precision fixed for the app runtime",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable runtime optimize acceleration",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Default model directory or Hugging Face repo id",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated wav outputs",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Path to the Gradio log file",
    )
    parser.add_argument(
        "--output-retention-count",
        type=int,
        default=DEFAULT_OUTPUT_RETENTION,
        help="Maximum number of generated wav files to keep",
    )
    parser.add_argument(
        "--max-generate-length",
        type=int,
        default=DEFAULT_MAX_GENERATE_LENGTH,
        help="Maximum generation schedule length fixed for the app runtime",
    )
    parser.add_argument(
        "--default-prompt-name",
        default=DEFAULT_PROMPT_NAME,
        help="Default built-in voice preset name",
    )
    parser.add_argument(
        "--default-precision",
        default=DEFAULT_PRECISION,
        choices=["bfloat16", "float32", "float16"],
        help="Default precision selected in the UI",
    )
    parser.add_argument(
        "--default-num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Default Num Steps selected in the UI",
    )
    parser.add_argument(
        "--default-guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="Default Guidance Scale selected in the UI",
    )
    parser.add_argument(
        "--default-speaker-scale",
        type=float,
        default=DEFAULT_SPEAKER_SCALE,
        help="Default Speaker Scale selected in the UI",
    )
    parser.add_argument(
        "--default-max-generate-length",
        type=int,
        default=DEFAULT_MAX_GENERATE_LENGTH,
        help="Default Max Generate Length selected in the UI",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Start the Gradio server without running an initial synthesis warmup.",
    )
    return parser.parse_args(argv)


def build_startup_config_panel(gr, app_config) -> None:
    """渲染只读的“启动固定参数”面板 / Render the read-only startup-config panel.

    把那些在服务进程生命周期内不可变的运行时配置 (模型路径、execution mode、精度、
    最大生成长度、是否 optimize) 以 ``interactive=False`` 的形式展示出来，提示用户
    “要改这些得重启并传新启动参数”。只在 Debug 模式 (DEBUG_GRADIO=1) 下挂载，
    无返回值——组件直接注册进当前 ``gr.Blocks`` 上下文。
    """
    with gr.Accordion("启动固定参数", open=False):
        gr.Markdown("只读。修改这部分需要重启服务并传入新的启动参数。")
        gr.Textbox(
            label="Model",
            value=app_config.default_model_name_or_path,
            interactive=False,
        )
        with gr.Row():
            gr.Textbox(
                label="Execution Mode",
                value=app_config.execution_mode,
                interactive=False,
            )
            gr.Textbox(
                label="Precision",
                value=app_config.precision,
                interactive=False,
            )
        with gr.Row():
            gr.Number(
                label="Max Generate Length",
                value=app_config.max_generate_length,
                precision=0,
                interactive=False,
            )
            gr.Checkbox(
                label="Optimize",
                value=app_config.optimize,
                interactive=False,
            )


def build_demo(gr, app_config, app_service) -> "gr.Blocks":
    """组装整个 Gradio 界面并绑定事件 / Build the full Gradio UI and wire callbacks.

    职责
    ----
    - 声明所有 UI 组件 (音色下拉、参考音频、参考转写、待合成文本、Settings 滑杆、
      Generate 按钮、输出音频)，并按左右两列布局。
    - 定义两个回调闭包：``run_synthesis`` (点 Generate 时跑合成) 与
      ``select_prompt_preset`` (切音色预设时回填参考音频/转写)。
    - 处理 Debug vs 普通两种模式：Debug 暴露真实控件，普通模式用 ``gr.State`` 占位
      隐藏的高级参数 (见下方 if/else 分支)。

    参数
    ----
    - ``gr``         : 运行时传入的 gradio 模块 (延迟 import)。
    - ``app_config`` : ``GradioAppConfig``，提供默认值与音色预设。
    - ``app_service``: ``GradioAppService``，实际执行推理 (``app_service.generate``)。

    返回
    ----
    一个已 ``.queue(...)`` 的 ``gr.Blocks``——队列把并发限制为 1，避免多请求同时抢
    GPU runtime (service 层内部也用 ``threading.Lock`` 串行化)。
    """
    from apps.gradio.service import (
        GRADIO_SYNTHESIS_MODE_CHOICES,
        SynthesisRequest,
        build_prompt_choice_items,
        resolve_prompt_selection,
    )

    def select_prompt_preset(prompt_name: str):
        """音色预设切换回调 / Callback when the voice-preset dropdown changes.

        把选中的预设名映射回它的参考音频路径和转写文本，用于自动回填下方的
        “参考音频”和“参考转写”两个组件 (即 voice cloning 的 reference)。
        选 "No Preset" 时返回 (None, "") 清空。
        """
        audio_path, prompt_text = resolve_prompt_selection(
            prompt_name,
            app_config.prompt_presets,
        )
        return audio_path, prompt_text

    def run_synthesis(
        text: str,
        synthesis_mode: str,
        prompt_audio_path: str | None,
        prompt_text: str,
        ode_method: str,
        num_steps: float,
        guidance_scale: float,
        speaker_scale: float,
        normalize_text: bool,
        seed: float,
    ):
        """Generate 按钮回调：把 UI 控件值打包并触发一次合成 / Run one synthesis.

        各参数即对应同名 UI 组件的当前值；注意有几个 (synthesis_mode/ode_method/
        speaker_scale) 在非 Debug 模式下来自隐藏的 ``gr.State`` 占位而非真实控件。
        滑杆/Number 组件给的是 float，这里显式转回 int/float 以匹配
        ``SynthesisRequest`` 的字段类型。返回 (音频文件路径, metrics dict)，分别
        回填到输出 Audio 与 Metrics 组件。
        """
        # 非 Debug 模式强制走 "tts" 模板；只有显式开 DEBUG_GRADIO=1 才允许用户在
        # SynthesisMode 下拉里切到 instruct_tts / text_to_audio 等其它合成模板。
        resolved_synthesis_mode = synthesis_mode if DEBUG_GRADIO_ENABLED else "tts"
        request = SynthesisRequest(
            model_name_or_path=app_config.default_model_name_or_path,
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            execution_mode=app_config.execution_mode,
            template_name=resolved_synthesis_mode,
            ode_method=ode_method,
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            speaker_scale=float(speaker_scale),
            normalize_text=normalize_text,
            seed=int(seed),
        )
        result = app_service.generate(request)
        return result.audio_path, result.metrics

    # 仅当发现了内置音色样本时才显示音色下拉；没有预设就把整列隐藏掉。
    show_prompt_preset = bool(app_config.prompt_presets)

    with gr.Blocks(title="dots.tts") as demo:
        # 顶部 banner：把自定义 CSS 内联进 <style> 再拼上标题/副标题 HTML 一起注入。
        gr.HTML(
            "<style>\n"
            + PLAYGROUND_CSS
            + "\n</style>\n"
            + """
            <div id="playground-banner">
              <h1>dots.tts</h1>
              <p class="subtitle">Fully-continuous Autoregressive TTS · 48 kHz · Voice Cloning</p>
            </div>
            """,
        )

        gr.HTML(
            """
            <div class="info-card">
              <span class="card-title">使用说明 · Instructions</span>
              <ol>
                <li>上传参考音频并填写对应转写文本 · Upload prompt audio and fill in its transcript.</li>
                <li>在文本框中输入要合成的内容 · Enter the text to synthesize.</li>
                <li>点击 <b>Generate</b> 合成声音 · Click <b>Generate</b> to synthesize speech.</li>
              </ol>
            </div>
            """,
        )

        # 主工作区：左列放 voice cloning 的参考输入 (prompt)，右列放待合成文本与设置。
        with gr.Row(equal_height=True, elem_classes="main-workspace"):
            # 左列：reference / prompt —— continuation cloning 的“声音样本 + 转写”。
            with gr.Column(scale=1, min_width=480, elem_classes="prompt-column"):
                prompt_preset = gr.Dropdown(
                    label="音色 · Voice Preset",
                    choices=build_prompt_choice_items(app_config.prompt_presets),
                    value=app_config.default_prompt_name,
                    info="内置音色clone样本；选择后自动填入参考音频与转写。",
                    elem_id="voice-preset-dropdown",
                    elem_classes="strong-label",
                    visible=show_prompt_preset,
                )
                # 参考音频组件：type="filepath" 让回调拿到的是磁盘路径字符串 (而非
                # numpy 波形)，正好对应 SynthesisRequest.prompt_audio_path 的契约。
                prompt_audio_path = gr.Audio(
                    label="参考音频 · Prompt Audio",
                    sources=["upload"],
                    type="filepath",
                    value=app_config.default_prompt_audio_path,
                    elem_classes="strong-label",
                )
                prompt_text = gr.Textbox(
                    label="参考音频转写 · Prompt Text",
                    lines=5,
                    value=app_config.default_prompt_text,
                    placeholder="Prompt audio 对应的文本转写（continuation cloning 必填）",
                    elem_classes="strong-label",
                )

            # 右列：synthesis —— 待合成文本 + 折叠的推理超参 + Generate + 输出音频。
            with gr.Column(scale=1, min_width=480, elem_classes="synthesis-column"):
                text = gr.Textbox(
                    label="待合成文本 · Text",
                    lines=5,
                    max_lines=8,
                    value=DEFAULT_INPUT_TEXT,
                    placeholder="输入待合成的文本",
                    elem_classes="strong-label",
                )
                with gr.Accordion("⚙️ Settings", open=False, elem_classes="settings-card"):
                    with gr.Row(elem_classes="settings-slider-row"):
                        # Num Steps = flow-matching ODE solver 的积分步数：越大越精细、越慢。
                        num_steps = gr.Slider(
                            label="Num Steps",
                            minimum=1,
                            maximum=32,
                            step=1,
                            value=app_config.default_num_steps,
                        )
                    with gr.Row(elem_classes="settings-slider-row"):
                        # Guidance Scale = classifier-free guidance (CFG) 强度：1.0 等价无引导，
                        # 调大让输出更贴合文本条件 (代价是可能过饱和/失真)。
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=3.0,
                            step=0.1,
                            value=app_config.default_guidance_scale,
                        )
                    with gr.Row(elem_classes="control-row"):
                        seed = gr.Number(
                            label="Seed",
                            value=DEFAULT_SEED,
                            precision=0,
                            scale=1,
                            min_width=180,
                        )
                        normalize_text = gr.Checkbox(
                            label="Normalize Text",
                            value=False,
                            scale=1,
                            min_width=180,
                        )
                generate = gr.Button(
                    "Generate",
                    variant="primary",
                    size="lg",
                    elem_classes="generate-button",
                )
                audio_out = gr.Audio(
                    label="生成音频 · Output",
                    type="filepath",
                    elem_classes="output-audio",
                )

        # 关键分支：Debug 模式把 synthesis_mode / ode_method / speaker_scale / metrics
        # 做成真实可交互控件；否则把它们换成隐藏的 gr.State 常量占位。两条分支必须产出
        # 同名变量，因为下面 generate.click 的 inputs/outputs 引用的是这些名字——无论真控件
        # 还是 State，都能作为事件的输入/输出参与同一套回调签名。
        if DEBUG_GRADIO_ENABLED:
            with gr.Accordion("Debug", open=False):
                synthesis_mode = gr.Dropdown(
                    label="SynthesisMode",
                    choices=list(GRADIO_SYNTHESIS_MODE_CHOICES),
                    value="tts",
                    info="选择合成模式；界面显示名会自动映射到 runtime 对应模板。",
                )
                ode_method = gr.Textbox(
                    label="ODE Method",
                    value=DEFAULT_ODE_METHOD,
                    lines=1,
                )
                # Speaker Scale = 说话人 x-vector (CAM++) 条件强度，放大音色相似度的引导。
                speaker_scale = gr.Slider(
                    label="Speaker Scale",
                    minimum=0.0,
                    maximum=3.0,
                    step=0.1,
                    value=app_config.default_speaker_scale,
                    info="说话人 x-vector 强度",
                )
                metrics = gr.JSON(label="Metrics", value=app_service.metadata())
                build_startup_config_panel(gr, app_config)
        else:
            # 非 Debug：用 gr.State 把这些高级参数固定成常量，界面上不可见也不可调，
            # 但仍能作为 generate.click 的 inputs 喂给 run_synthesis (值恒定)。
            synthesis_mode = gr.State(value="tts")
            ode_method = gr.State(value=DEFAULT_ODE_METHOD)
            speaker_scale = gr.State(value=app_config.default_speaker_scale)
            metrics = gr.State(value={})  # 输出占位：非 Debug 模式不展示 metrics

        # 事件绑定：inputs 的顺序必须与 run_synthesis 的位置参数严格一一对应。
        # concurrency_limit=1 保证同一时刻只有一个合成在跑 (单 GPU runtime 不可并发)。
        generate.click(
            fn=run_synthesis,
            inputs=[
                text,
                synthesis_mode,
                prompt_audio_path,
                prompt_text,
                ode_method,
                num_steps,
                guidance_scale,
                speaker_scale,
                normalize_text,
                seed,
            ],
            outputs=[audio_out, metrics],
            concurrency_limit=1,
        )
        # 切换音色预设 -> 自动回填参考音频与转写两个组件。
        prompt_preset.change(
            fn=select_prompt_preset,
            inputs=[prompt_preset],
            outputs=[prompt_audio_path, prompt_text],
            concurrency_limit=1,
        )

    # 开队列：全局并发 1 + 最多排队 8 个请求，多余请求会被拒绝/排队等待。
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main() -> None:
    """应用入口：装配 config/service、预热、启动 Gradio 服务器 / App entry point.

    流程：解析 CLI -> 配置日志 -> ``build_gradio_app_config`` 组装运行时配置 ->
    构造 ``GradioAppService`` (此时尚未加载模型) -> 可选 warmup (跑一次合成把模型
    权重/KV cache/CUDA kernel 预热好，避免首个真实请求超慢) -> ``build_demo`` 搭界面
    -> ``demo.launch`` 起服务器。无返回值。

    注意 ``import gradio``/``loguru``/service 都放在函数体内做延迟 import：避免仅
    ``import app`` (例如做参数解析) 就被迫加载这些重依赖。
    """
    args = parse_args()
    import gradio as gr
    from loguru import logger

    from apps.gradio.service import GradioAppService, build_gradio_app_config
    from dots_tts.utils.logging import configure_logging

    configure_logging(log_file=args.log_file)
    logger.info(
        "Gradio app starting: host={} port={} model_name_or_path={} output_dir={} "
        "log_file={} output_retention_count={} max_generate_length={} execution_mode={} precision={} optimize={} "
        "default_prompt_name={} skip_warmup={}",
        args.host,
        args.port,
        args.model_name_or_path,
        args.output_dir,
        args.log_file,
        args.output_retention_count,
        args.max_generate_length,
        args.execution_mode,
        args.precision,
        args.optimize,
        args.default_prompt_name,
        args.skip_warmup,
    )
    app_config = build_gradio_app_config(
        host=args.host,
        port=args.port,
        execution_mode=args.execution_mode,
        precision=args.precision,
        optimize=args.optimize,
        model_name_or_path=args.model_name_or_path,
        output_dir=Path(args.output_dir),
        output_retention_count=args.output_retention_count,
        max_generate_length=args.max_generate_length,
        default_prompt_name=args.default_prompt_name,
        default_precision=args.default_precision,
        default_num_steps=args.default_num_steps,
        default_guidance_scale=args.default_guidance_scale,
        default_speaker_scale=args.default_speaker_scale,
        default_max_generate_length=args.default_max_generate_length,
    )
    app_service = GradioAppService(app_config)
    # 预热：默认跑一次合成把模型加载 + CUDA kernel 编译/缓存提前做掉；--skip-warmup
    # 可跳过 (启动更快，但第一个真实请求会承担这部分冷启动开销)。
    if args.skip_warmup:
        logger.info("Gradio app warmup skipped by --skip-warmup.")
    else:
        warmup_metrics = app_service.warmup()
        logger.info("Gradio app warmup metrics: {}", warmup_metrics)
    demo = build_demo(gr, app_config, app_service)
    logger.info(
        "Gradio app ready: host={} port={} execution_mode={} precision={} optimize={} default_model_name_or_path={}",
        app_config.host,
        app_config.port,
        app_config.execution_mode,
        app_config.precision,
        app_config.optimize,
        app_config.default_model_name_or_path,
    )
    demo.launch(
        server_name=app_config.host,
        server_port=app_config.port,
        theme=build_playground_theme(gr),
        css=PLAYGROUND_CSS,
    )


if __name__ == "__main__":
    main()
