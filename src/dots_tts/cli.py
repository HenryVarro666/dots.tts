"""dots.tts 推理命令行入口 / dots.tts inference CLI entry point.

本文件做什么 (What this file does):
    把 dots.tts 的"文本 → 语音"推理流程包装成一条命令行命令。它只负责解析参数、
    加载运行时、调用一次生成、把结果写成 WAV，本身不含任何声学/模型逻辑——真正的
    flow-matching / DiT / AudioVAE 计算都在 `dots_tts.runtime.DotsTtsRuntime` 里。
    This module wraps dots.tts text-to-speech inference into a single CLI command:
    parse args -> load runtime -> one `generate()` call -> write a WAV. No model
    math lives here; all flow-matching / DiT / AudioVAE work happens in the runtime.

在推理数据流里的位置 (Position in the inference pipeline):
    用户 CLI 参数  ->  parse_args()  ->  DotsTtsRuntime.from_pretrained()  (加载 Qwen2.5
    自回归主干 + flow-matching DiT 声学头 + BigVGAN AudioVAE + CAM++ x-vector)
    ->  runtime.generate()  (返回 {audio, sample_rate, fid, ...})  ->  soundfile.write 落盘。

关键函数清单 (Key functions):
    - parse_args(argv): 定义并解析所有 argparse 参数(采样步数、CFG、说话人缩放、
      ODE solver、模板、语言、prompt 音频克隆等)。
    - main(argv): 编排"加载运行时 -> 生成 -> 写 WAV"的整个一次性命令生命周期。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv=None):
    """解析 CLI 参数 / Parse command-line arguments.

    职责 (Responsibility):
        声明 dots.tts 推理需要的全部开关并解析它们;这些参数最终分别喂给
        `from_pretrained`(加载相关)与 `generate`(采样相关)。

    参数 (Args):
        argv: 可选的参数列表(list[str]);为 None 时 argparse 从 sys.argv 读取。
              便于在测试/脚本中以编程方式传参。

    返回 (Returns):
        argparse.Namespace,字段与下面 add_argument 的 dest 一一对应。

    参数分组(便于阅读,代码顺序不变):
        · 加载类: --model-name-or-path / --revision / --cache-dir / --precision
        · 生成核心: --text / --output / --seed
        · voice clone(音色克隆): --prompt-audio / --prompt-text / --speaker-scale
        · 采样控制: --ode-method / --num-steps / --guidance-scale / --max-generate-length
        · 文本/语言: --language / --template-name / --normalize-text
        · 诊断: --profile-inference
    """
    parser = argparse.ArgumentParser(description="dots.tts inference CLI.")
    # 四种生成模板预设(prompt 模板),决定文本如何被组织进自回归主干的输入序列:
    #   tts            纯文本转语音(最常用)
    #   instruction_tts  带指令/风格描述的可控合成
    #   text_to_audio    文本转通用音频(不限于语音)
    #   tts_interleave   文本与音频 token 交错(double-streaming 流式相关)
    template_choices = ("tts", "instruction_tts", "text_to_audio", "tts_interleave")
    # 权重来源:本地目录或 HF repo id。runtime 会先按本地路径找,找不到再走 snapshot_download。
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Local pretrained directory or Hugging Face repo id",
    )
    # --revision / --cache-dir 仅在从 HF 下载时生效;指定本地目录时被忽略。
    parser.add_argument(
        "--revision", default=None, help="Optional Hugging Face revision"
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Optional Hugging Face cache dir"
    )
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--output", default="output.wav", help="Output wav file path")
    # 推理精度;dots.tts 默认 bfloat16(数值范围大、对 DiT/AR 主干更稳)。
    parser.add_argument(
        "--precision", type=str, default="bfloat16", help="Inference precision"
    )
    # 固定随机种子使采样可复现:flow-matching ODE 初值噪声与任何 stochastic 步都由它决定。
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for inference.",
    )
    # voice clone:prompt-audio 提供参考音色(经 CAM++ 提取 x-vector + AudioVAE 编成 latent
    # 前缀),prompt-text 是这段参考音频的转写。二者应配对给出,否则克隆条件不完整。
    parser.add_argument(
        "--prompt-audio", type=str, default=None, help="Path to prompt audio"
    )
    parser.add_argument(
        "--prompt-text", type=str, default=None, help="Transcript of prompt audio"
    )
    # 语言标签模式:None=不加标签;"auto_detect"=自动判别;或显式给语言码/名(EN/en/english/chinese)。
    # 该标签会拼进 prompt,提示自回归主干切到对应语种的发音先验。
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language tag mode. Default: none. Supported values: none, auto_detect, or a language code/name such as EN/en/english/chinese.",
    )
    # 选择上面 template_choices 之一;None 时由 runtime 按是否有 prompt 音频挑默认模板。
    parser.add_argument(
        "--template-name",
        choices=template_choices,
        default=None,
        help="Named template preset for generation.",
    )
    # flow-matching 求解连续 latent 的 ODE solver 方法;"euler" 为一阶显式,简单稳定、最常用。
    parser.add_argument(
        "--ode-method", type=str, default="euler", help="ODE solver method"
    )
    # ODE 离散化步数(NFE):步数越多越精细但越慢。dots.tts 是连续潜在 flow-matching,
    # 此处的 "diffusion steps" 即 velocity field 沿时间轴的积分步数;经 MeanFlow 蒸馏后可取很小。
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Diffusion sampling steps"
    )
    # classifier-free guidance (CFG) 强度:在条件与无条件 velocity 之间外推。
    # >1 增强对文本/音色条件的贴合(更准但可能更"硬"),=1 等于不做 CFG。默认 1.2 偏温和。
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.2,
        help="Classifier-free guidance scale",
    )
    # 单独施加于参考说话人 x-vector 的缩放:放大可让克隆音色更像参考人;
    # 与 CFG 解耦,是 dots.tts 对"音色条件"与"文本条件"分别调强度的设计。
    parser.add_argument(
        "--speaker-scale",
        type=float,
        default=1.5,
        help="Scale applied to the reference speaker embedding",
    )
    # 生成长度上限,单位是 audio patch 数(prompt 前缀 + 新生成共计);
    # 防止自回归主干无限解码,也是显存/时延的硬上界。1 个 patch 对应固定时长的 latent 片段。
    parser.add_argument(
        "--max-generate-length",
        type=int,
        default=500,
        help="Maximum total audio patch count (prompt + generated)",
    )
    # 文本归一化开关(store_true,默认 False):开启后数字/缩写等会先被规范成可读写法再合成。
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Whether to normalize text before inference",
    )
    # 性能剖析开关:开启后收集各模块逐次推理耗时,用于定位瓶颈(不影响音频结果)。
    parser.add_argument(
        "--profile-inference",
        action="store_true",
        help="Collect per-module inference timing statistics",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """CLI 主流程 / CLI orchestration entry point.

    职责 (Responsibility):
        编排一次性的"加载 -> 生成 -> 落盘"命令生命周期:解析参数、配置日志、固定种子、
        加载 `DotsTtsRuntime`、调用 `generate`、把波形写成 WAV,并在异常时记录后重新抛出。

    参数 (Args):
        argv: 透传给 parse_args 的参数列表;None 时取 sys.argv。

    返回 (Returns):
        None。被 `__main__` 包成 `SystemExit(main())`;正常返回 None => 退出码 0。

    设计说明 (Why):
        soundfile / loguru / runtime 都在函数体内 **延迟导入**,而非模块顶层——
        这样仅 `--help` 或参数解析失败时无需付出加载 torch/runtime 的启动开销。
    """
    args = parse_args(argv)
    # 延迟导入(lazy import):把重依赖推迟到真正要推理时再加载,加快 CLI 启动与 --help。
    import soundfile as sf
    from loguru import logger

    from dots_tts.runtime import DotsTtsRuntime
    from dots_tts.utils.logging import configure_logging
    from dots_tts.utils.util import seed_everything

    configure_logging()
    seed_everything(args.seed)  # 在加载/生成之前就播种,确保整个流程可复现
    output_path = Path(args.output)
    # 提前建好输出目录(含父级),避免生成成功后却因目录不存在而写盘失败。
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "CLI command started: model={} output={} seed={}",
        args.model_name_or_path,
        output_path,
        args.seed,
    )

    try:
        # 加载运行时:此调用内部组装 Qwen2.5-1.5B 自回归主干 + flow-matching DiT 声学头
        # + BigVGAN AudioVAE + CAM++ x-vector,返回一个可复用的 DotsTtsRuntime。
        runtime = DotsTtsRuntime.from_pretrained(
            args.model_name_or_path,
            revision=args.revision,
            cache_dir=args.cache_dir,
            precision=args.precision,
            max_generate_length=args.max_generate_length,
        )
        # 单次合成:返回 dict,含 audio(波形张量)/sample_rate/fid(request id)等。
        result = runtime.generate(
            text=args.text,
            prompt_audio_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            language=args.language,
            template_name=args.template_name,
            ode_method=args.ode_method,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            speaker_scale=args.speaker_scale,
            normalize_text=args.normalize_text,
            profile_inference=args.profile_inference,
        )
        # 写 WAV:把张量转 fp32(soundfile 不接受 bf16)、搬回 CPU、squeeze 掉 batch/channel
        # 维(从 (B,1,T) 之类压成一维 (T,)),再转 numpy 交给 soundfile 按采样率写盘。
        sf.write(
            output_path,
            result["audio"].float().cpu().squeeze().numpy(),
            result["sample_rate"],
        )
    except Exception:
        # 不吞异常:先把完整 traceback 记进日志便于排错,再原样向上抛(交给 SystemExit)。
        logger.exception(
            "CLI inference failed: model={} output={}",
            args.model_name_or_path,
            output_path,
        )
        raise

    logger.info(
        "CLI output written: request_id={} output={} sample_rate={} samples={}",
        result["fid"],
        output_path,
        result["sample_rate"],
        int(result["audio"].shape[-1]),
    )


if __name__ == "__main__":
    # 以脚本方式运行时:把 main() 的返回值作为进程退出码(None => 0,表示成功)。
    raise SystemExit(main())
