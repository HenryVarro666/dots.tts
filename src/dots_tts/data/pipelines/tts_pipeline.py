"""TTS 训练样本处理流水线 / The TTS training-sample processing pipeline.

本文件做什么 / What this file does
==================================
把 adapter 产出的一条原始样本(含 text + audio 文件路径)落地成可直接喂给模型的
训练样本。核心是 ``BasicTtsPipeline``: 一站式完成
``text -> 模板拼装 + tokenize`` 与 ``audio 波形 -> 重采样/静音规整/对齐补零 ->
fbank + audio-token 计数``,产出 ``input_ids / labels / loss_mask / fbank`` 等张量。

dots.tts 是连续潜在(continuous-latent)自回归 TTS:音频不被离散化成 audio token,
但 AR 主干(Qwen2.5-1.5B)的序列里仍要为每段音频占据若干 **audio-span 占位 token**
(``[文本对应语音]`` 段),其个数由波形长度除以 token hop 算出;真正的声学内容由
flow-matching DiT 声学头在连续 latent 空间里生成。本文件负责把波形长度精确对齐到
token 边界,使"占位 token 个数 ↔ 波形采样点数"严格成整除关系。

在训练数据流里的位置 / Position in the data flow
------------------------------------------------
[adapter 产出 {fid, text, audio_path}]
  -> 本文件 BasicTtsPipeline.process_sample:
       text  -> build_tokenized_example(模板/tokenize) -> input_ids/labels/loss_mask
       audio -> 读取 -> 重采样 -> 边缘静音规整 -> 对齐补零 -> num_audio_tokens
             -> extract_speaker_fbank -> x-vector 用的 fbank
  -> collate 组 batch -> AudioVAE 编码 latent + AR 主干 + DiT 声学头训练

关键类清单 / Key classes
------------------------
- ``BasicTtsPipeline``    : 标准 TTS 模板(text 段 + audio 段)的样本处理器。
- ``InterleaveTtsPipeline``: 仅替换模板为 interleave(流式/double-streaming)变体,
  其余逻辑完全复用基类。
"""

from __future__ import annotations

import soundfile as sf
import torch

from dots_tts.utils.profiling import ensure_data_profiler
from dots_tts.data.pipelines.base import BaseSamplePipeline
from dots_tts.data.pipelines.preprocessing import (
    compute_num_audio_tokens,
    normalize_edge_silence_duration,
    pad_waveform_align_only,
)
from dots_tts.data.pipelines.tokenizing import build_tokenized_example
from dots_tts.modules.speaker.fbank import extract_speaker_fbank
from dots_tts.utils.audio import high_quality_resample

# --- 模板段标记(中文标签作为 special-token 样式的分段提示)/ template segment markers ---
# 这些字符串会被 tokenizer 编成普通 token,夹在 {text}/{audio} 占位之间,起到"段类型提示"
# 的作用——告诉模型接下来是文本条件还是要生成的语音。占位符 {text}/{audio}/{interleave}
# 在 tokenizing.build_tokenized_example 里被替换成真正的 token 序列。
TTS_TEXT_PREFIX = "[文本]"  # 普通 TTS 的文本条件段前缀 / plain-text condition prefix
TTS_AUDIO_PREFIX = "[文本对应语音]"  # "该文本对应的语音"段前缀,后接 audio 占位 / target-speech prefix
TTS_INSTRUCTION_TEXT_PREFIX = "[带指令文本]"  # 带指令(instruction)的文本段前缀 / instruction-text prefix
TTA_TEXT_PREFIX = "[声音描述]"  # text-to-audio: 声音描述段前缀 / sound-description prefix
TTA_AUDIO_PREFIX = "[描述对应声音]"  # text-to-audio: 描述对应声音段前缀 / described-sound prefix
TTS_INTERLEAVE_PREFIX = "[流式语音合成]"  # 流式(interleave/double-streaming)合成段前缀 / streaming prefix
# 各任务的默认模板:用 str.format 风格的 {text}/{audio}/{interleave} 占位拼接而成。
# {audio} 表示"在此处插入整段连续的 audio-span 占位";{interleave} 表示按 token 逐字交错
# 排布文本与音频 span(流式),二者互斥(见 tokenizing.parse_template)。
DEFAULT_TRAIN_TEMPLATE = f"{TTS_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
DEFAULT_INSTRUCTION_TTS_TEMPLATE = (
    f"{TTS_INSTRUCTION_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
)
DEFAULT_TEXT_TO_AUDIO_TEMPLATE = f"{TTA_TEXT_PREFIX}{{text}}{TTA_AUDIO_PREFIX}{{audio}}"
DEFAULT_INTERLEAVE_TRAIN_TEMPLATE = f"{TTS_INTERLEAVE_PREFIX}{{interleave}}"


class BasicTtsPipeline(BaseSamplePipeline):
    """标准 TTS 样本处理器:把 adapter 产出的样本转成模型可吃的张量字典。

    职责 / Responsibility
    ---------------------
    实现 ``BaseSamplePipeline.process_sample`` 这一抽象方法,对单条样本做两条并行处理:
      - 文本侧:用类属性 ``template`` 拼模板并 tokenize,得到 ``input_ids/labels/loss_mask``;
      - 音频侧:读取波形 -> 高质量重采样到训练采样率 -> 边缘静音规整 -> 末尾补零对齐到
        token hop -> 由对齐后长度算出 audio-span 占位 token 数 -> 提取 speaker fbank。
    最终把这些字段写回样本 dict 一并返回。

    对应 ML 概念 / ML concept
    -------------------------
    这是 TTS 训练前的 per-sample preprocessing:它保证"波形采样点数"与"序列里 audio-span
    占位 token 个数"严格成整除关系(对齐到 token 边界),从而让连续 latent 能整齐地切片、
    AR 主干的 next-token 监督(loss_mask 只在音频段为 1)落在正确的位置上。

    注意 / Note
    -----------
    ``template`` 是类级常量;子类(如 ``InterleaveTtsPipeline``)只需覆盖它即可切换任务形态,
    本类其余逻辑无需改动。
    """

    template = DEFAULT_TRAIN_TEMPLATE  # 默认普通 TTS 模板;子类可覆写以切换任务 / overridable per subclass

    def __init__(self, tokenizer, data_cfg, *, profiler=None):
        """绑定 tokenizer 与数据超参,准备好 profiler。

        参数 / Args
            tokenizer: HF 风格 tokenizer,需具备 ``encode``/``eos_token_id`` 及音频 special token。
            data_cfg:  数据配置对象,读取两个关键超参:
                - ``train_audio_sample_rate``: 训练统一采样率(波形会被重采样到此值);
                - ``audio_samples_per_llm_token``: token hop——每个 audio-span 占位 token 对应的
                  采样点数,既用于对齐补零也用于换算 token 个数。
            profiler: 可选的数据侧性能采样器;为 None 时 ``ensure_data_profiler`` 给出空实现。
        """
        self.tokenizer = tokenizer
        self.train_audio_sample_rate = int(data_cfg.train_audio_sample_rate)
        self.audio_samples_per_llm_token = int(data_cfg.audio_samples_per_llm_token)
        self.profiler = ensure_data_profiler(profiler)

    @staticmethod
    def _load_waveform(audio_path: str) -> tuple[torch.Tensor, int]:
        """从磁盘读取音频并转成单声道 float32 波形张量。

        参数 / Args
            audio_path: 音频文件的文件系统路径(必须是 str;不接受内存 buffer)。

        返回 / Returns
            (waveform, sample_rate):
              - waveform: 形状 (1, T) 的单声道波形(float32),T 为采样点数;
              - sample_rate: 文件原始采样率(Hz),后续在 process 中再重采样到训练采样率。

        说明 / Notes
            ``always_2d=True`` 保证 soundfile 返回 (T, C) 二维数组;转置成 (C, T) 后,
            多声道时按通道求均值下混(downmix)成单声道。``contiguous()`` 让下游算子拿到
            连续内存,避免转置带来的非连续视图开销。
        """
        if not isinstance(audio_path, str):
            raise TypeError(
                f"Training audio must be a filesystem path, got {type(audio_path)}."
            )
        audio_data, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
        # sf 返回 (T, C);转置为 (C, T) 以符合 torchaudio/本仓库 (channel, time) 约定。
        waveform = torch.from_numpy(audio_data.T)
        if waveform.size(0) > 1:
            # 多声道下混为单声道:沿 channel 维求均值并保留该维 -> (1, T)。
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.contiguous(), int(sample_rate)

    @staticmethod
    def _validate_source_sample(sample: dict) -> None:
        """校验 adapter 是否产出了本 pipeline 必需的三个字段。

        参数 / Args
            sample: adapter 产出的原始样本 dict。
        异常 / Raises
            ValueError: 缺少 ``fid``(样本唯一 id)、``text``(待合成文本)或
                ``audio``(音频文件路径)中任意一个时抛出,并列出缺失字段与现有 keys
                便于排查数据源契约不一致。
        """
        missing = [field for field in ("fid", "text", "audio") if field not in sample]
        if missing:
            raise ValueError(
                "Source adapter must emit fid/text/audio. "
                f"Missing fields: {missing}. Sample keys: {sorted(sample.keys())}"
            )

    def process_sample(self, raw_sample: dict) -> dict:
        """单样本转换入口(实现基类抽象方法):校验 + 计时,委托给具体实现。

        参数 / Args
            raw_sample: adapter 产出的原始样本 dict(基类已传入浅拷贝)。
        返回 / Returns
            dict: 在原样本基础上新增/覆盖了波形、token、fbank 等字段的样本。

        说明:这里再做一次浅拷贝并把 ``fid`` 规范成字符串,然后用 profiler 包住整个
        重活,真正的处理逻辑在 ``_process_sample_impl`` 里。
        """
        sample = dict(raw_sample)
        self._validate_source_sample(sample)
        sample["fid"] = str(sample["fid"])  # fid 统一成 str,避免下游因类型(int/路径)不一致出错

        with self.profiler.measure("worker.process_sample_total"):
            return self._process_sample_impl(sample)

    def _process_sample_impl(self, sample: dict) -> dict:
        """样本处理的实际实现:波形清洗 + 长度对齐 + tokenize + fbank。

        参数 / Args
            sample: 已校验、已浅拷贝、``fid`` 已转 str 的样本 dict。
        返回 / Returns
            dict: 写入了下列字段的同一个 sample——
              - ``sample`` (Tensor (1, T)): 重采样+静音规整+对齐补零后的训练波形;
              - ``sample_rate`` (int): 训练采样率;
              - ``unpadded_sample_length`` / ``sample_length`` (int): 补零前/后采样点数;
              - ``input_ids`` / ``labels`` / ``loss_mask`` (list): teacher-forcing 监督序列;
              - ``num_text_tokens`` / ``num_audio_tokens`` / ``num_total_tokens`` (int): 计数;
              - ``fbank`` (Tensor (T_frames, 80)) / ``fbank_length`` (int): speaker x-vector 输入。

        关键不变式 / Key invariant
            波形先做边缘静音规整,再补零到 ``audio_samples_per_llm_token`` 的整数倍;由此
            对齐后长度 ÷ token hop = ``num_audio_tokens``(整除,无零头),保证序列里 audio-span
            占位 token 个数与连续 latent 的可切片帧数严格一一对应。每一步都用 profiler 计时,
            便于定位数据侧瓶颈(IO / 重采样 / tokenize)。
        """
        profiler = self.profiler
        with profiler.measure("worker.load_audio"):
            waveform, sample_rate = self._load_waveform(sample["audio"])
        with profiler.measure("worker.resample_audio"):
            # 统一到训练采样率(本仓库 24 kHz 一类);high_quality_resample 用 sinc+Kaiser 抗混叠。
            waveform = high_quality_resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.train_audio_sample_rate,
            )
        with profiler.measure("worker.normalize_edge_silence"):
            # 把两端 leading/trailing 静音规整到固定时长(默认 250 ms),消除"开头/结尾留白"
            # 这个可被模型误学的伪信号;此步可能裁剪也可能补零,故长度尚未对齐 token 边界。
            waveform = normalize_edge_silence_duration(
                waveform,
                sample_rate=self.train_audio_sample_rate,
            )
        sample["sample"] = waveform
        sample["sample_rate"] = self.train_audio_sample_rate
        sample["unpadded_sample_length"] = int(waveform.size(-1))  # 记录对齐补零前的真实长度(用于 mask/裁回)

        with profiler.measure("worker.pad_audio"):
            # 仅在末尾补零,把长度向上对齐到 token hop 的整数倍——只增不减,不动可听内容。
            waveform = pad_waveform_align_only(
                waveform,
                multiple_of=self.audio_samples_per_llm_token,
            )
        sample["sample"] = waveform
        sample["sample_length"] = int(waveform.size(-1))  # 对齐后的最终采样点数

        # 由对齐后长度整除 token hop 得到 audio-span 占位 token 数;若未对齐此函数会直接报错。
        num_audio_tokens = compute_num_audio_tokens(
            sample["sample_length"],
            audio_samples_per_llm_token=self.audio_samples_per_llm_token,
        )
        with profiler.measure("worker.tokenize"):
            # 用模板把 text 与 num_audio_tokens 个 audio-span 拼成完整序列,并生成 next-token
            # 监督:input_ids/labels 错一位,loss_mask 只在音频段为 1(只对音频部分回传 loss)。
            tokenized = build_tokenized_example(
                text=sample["text"],
                tokenizer=self.tokenizer,
                template=self.template,
                num_audio_tokens=num_audio_tokens,
            )
        sample["input_ids"] = tokenized["input_ids"]
        sample["labels"] = tokenized["labels"]
        sample["loss_mask"] = tokenized["loss_mask"]
        sample["input_ids_length"] = len(tokenized["input_ids"])
        sample["num_text_tokens"] = tokenized["text_token_count"]
        sample["num_audio_tokens"] = num_audio_tokens
        sample["num_total_tokens"] = sample["input_ids_length"]

        with profiler.measure("worker.extract_fbank"):
            # 从(已对齐的)波形提 speaker 分支 fbank;内部会重采样到 16 kHz 以喂 CAM++ x-vector,
            # 该 x-vector 作为音色条件喂给 flow-matching DiT 声学头。
            fbank = extract_speaker_fbank(
                sample["sample"],
                sample_rate=sample["sample_rate"],
            )
        sample["fbank"] = fbank
        sample["fbank_length"] = int(fbank.size(0))  # fbank 帧数 (T_frames),供下游做长度/padding 对齐
        return sample


class InterleaveTtsPipeline(BasicTtsPipeline):
    """流式(interleave / double-streaming)TTS 样本处理器。

    与 ``BasicTtsPipeline`` 唯一的区别是把 ``template`` 换成 interleave 变体:文本 token 与
    audio-span 不再"先文本后整段音频",而是按 token 逐字交错排布(每个文本 token 后紧跟一个
    audio span),以支持 double-streaming——边读文本边出音频的低延迟流式生成。波形读取/重采样/
    对齐/fbank 等全部逻辑都直接继承基类,无需改动。
    """

    template = DEFAULT_INTERLEAVE_TRAIN_TEMPLATE  # 改用 {interleave} 占位模板,触发文本/音频交错排布
