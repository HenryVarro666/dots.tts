"""波形(waveform)预处理阶段 / Waveform preprocessing stage.

本文件做什么 / What this file does
==================================
这是数据预处理 pipeline 中针对**原始音频波形**的清洗与规整工具集。dots.tts 是连续潜在
(continuous-latent) TTS:原始波形先经 BigVGAN AudioVAE 编码成连续 latent,再被自回归主干
(Qwen2.5-1.5B)与 flow-matching DiT 声学头消费。波形进入 AudioVAE 之前必须满足两个约束:
  1. 长度对齐 / length alignment —— 采样点数要能被某个 hop(每个 LLM token 对应的采样点数)整除,
     这样波形 latent 才能整齐地切成一个个 audio token,不留半个 token 的零头。
  2. 边缘静音规整 / edge-silence normalization —— 训练样本两端的留白(leading/trailing silence)
     应统一到一个目标时长,避免模型把"开头/结尾静音多长"当成可学习的伪信号,影响停顿与节奏。

在推理/训练数据流里的位置 / Position in the data flow
----------------------------------------------------
[读取音频] -> [本文件:边缘静音规整 + 长度对齐 + 补零] -> [AudioVAE 编码成 latent]
            -> [切成 audio token] -> [喂给 AR 主干 / DiT]
属于"数据落盘前/喂模型前"的纯张量清洗,不涉及任何模型权重,可在 CPU 上完成。

关键函数清单 / Key functions
---------------------------
- align_length:            把采样点数向上取整到 multiple_of 的整数倍(对齐 token hop)。
- pad_waveform_align_only: 仅按对齐需求在末尾补零(zero-pad),不做任何裁剪。
- normalize_edge_silence_duration: 把波形两端的静音时长规整到目标毫秒数(裁剪或补零)。
- compute_num_audio_tokens: 由对齐后的采样点数换算出 audio token 个数(整除关系)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 边缘静音规整的默认目标时长:两端静音都规整到 250 ms / default target edge-silence duration.
DEFAULT_EDGE_SILENCE_MS = 250.0
# 静音判定阈值(分贝):比峰值低 30 dB 以下的采样点视为"静音"/ "below peak by 30 dB" counts as silence.
DEFAULT_EDGE_SILENCE_TOP_DB = 30.0


def align_length(num_samples: int, multiple_of: int | None) -> int:
    """把采样点数向上取整到 multiple_of 的整数倍 / round sample count up to a multiple.

    为什么 / Why
        AudioVAE 把波形编码成 latent 后要整齐切成 audio token,每个 token 固定吃
        `multiple_of` 个采样点(token hop)。长度若不是其整数倍,最后会剩半个 token,
        故先把长度对齐(只增不减,向上取整 / ceil)。

    参数 / Args
        num_samples: 当前采样点数(标量 int)。
        multiple_of: 对齐基数;None 或 <=0 表示"不对齐",原样返回。

    返回 / Returns
        对齐后的采样点数(int),恒 >= num_samples。
    """
    if multiple_of is None or multiple_of <= 0:
        return int(num_samples)
    if num_samples % multiple_of == 0:
        return int(num_samples)
    # 标准向上取整到倍数的整数运算 / integer ceil-to-multiple: ⌈n/m⌉ * m,避免浮点误差。
    return int(((num_samples + multiple_of - 1) // multiple_of) * multiple_of)


def pad_waveform_align_only(
    waveform: torch.Tensor,
    *,
    multiple_of: int | None,
) -> torch.Tensor:
    """仅按对齐需求在末尾补零 / zero-pad the tail only to satisfy length alignment.

    职责 / Responsibility
        与 normalize_edge_silence_duration 不同,这里**只补零、不裁剪**:把波形末尾补到
        最近的 `multiple_of` 整数倍,保证后续能整切成完整 audio token。补的是静音(0.0),
        不改变可听内容。

    参数 / Args
        waveform: 形状 (..., T) 的波形张量,末维 T 是采样点数(time)。
        multiple_of: 对齐基数;None 或 <=0 时直接原样返回。

    返回 / Returns
        形状 (..., T') 的张量,T' 是 T 向上对齐到 multiple_of 的结果(T' >= T)。
    """
    if multiple_of is None or multiple_of <= 0:
        return waveform

    target_length = align_length(waveform.size(-1), multiple_of)
    delta = target_length - waveform.size(-1)
    if delta <= 0:
        return waveform

    # F.pad 的 (left, right) 作用在最后一维:(0, delta) 表示只在尾部补 delta 个零 / pad tail only.
    return F.pad(waveform, (0, delta), "constant", 0.0)


def normalize_edge_silence_duration(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    target_silence_duration_ms: float = DEFAULT_EDGE_SILENCE_MS,
    top_db: float = DEFAULT_EDGE_SILENCE_TOP_DB,
) -> torch.Tensor:
    """把波形两端的静音时长规整到目标值 / normalize leading & trailing silence to a target duration.

    职责 / Responsibility
        让每条训练样本的开头/结尾静音都恰好是 `target_silence_duration_ms`:不足则补零,
        过多则裁掉。这样模型学到的"句首/句尾停顿"长度一致,不被原始录音里随机的留白污染,
        有利于稳定的节奏(prosody)与流式(streaming)起止判定。

    参数 / Args
        waveform: 形状 (C, T) 的波形,第 0 维是声道(channel),末维是采样点(time)。
                  内部以第 0 声道判定静音,但补零/裁剪同时作用到所有声道。
        sample_rate: 采样率(Hz),用于把毫秒目标换算成采样点数。
        target_silence_duration_ms: 两端各自要规整到的目标静音时长(毫秒)。
        top_db: 静音判定的相对阈值(dB);振幅低于"峰值 - top_db"的采样点算静音。

    返回 / Returns
        形状 (C, T') 的波形,两端静音各约为 target_samples 个采样点。

    对应 ML 概念 / ML concept
        数据规整(data normalization)中的"端点对齐",是去除与内容无关的混淆变量
        (nuisance variable)的常见手段。
    """
    mono_waveform = waveform[0]  # 取第 0 声道做能量判定 / use channel-0 for silence detection.
    # 目标静音的采样点数 = sample_rate * ms / 1000,四舍五入到整数 / ms -> samples.
    target_samples = int(round(float(sample_rate) * float(target_silence_duration_ms) / 1000.0))
    amplitude = mono_waveform.abs()
    peak = float(amplitude.max().item())
    if peak <= 0.0:
        # 全静音(纯零)波形:无法定阈值,直接把长度对齐到目标——先截断再补零 / all-silence fast path.
        waveform = waveform[..., :target_samples]
        current_length = int(waveform.size(-1))
        if current_length < target_samples:
            waveform = F.pad(waveform, (0, target_samples - current_length), "constant", 0.0)
        return waveform

    # 相对峰值的振幅阈值:-top_db 分贝换算成幅值比 10**(-top_db/20) / dB -> linear amplitude ratio.
    threshold = peak * (10.0 ** (-float(top_db) / 20.0))
    # 找出所有"非静音"采样点的下标(已展平成 1D) / indices of samples above threshold.
    non_silent = torch.nonzero(amplitude > threshold, as_tuple=False).flatten()
    first_non_silent = int(non_silent[0].item())  # 第一个有声采样点 / start of speech.
    last_non_silent = int(non_silent[-1].item())  # 最后一个有声采样点 / end of speech.

    # 头部静音 = 第一个有声点之前的采样数;尾部静音 = 总长 - 最后有声点 - 1 / count edge silences.
    leading_silence_samples = first_non_silent
    trailing_silence_samples = int(mono_waveform.numel()) - last_non_silent - 1

    # 头部:delta>0 说明静音不足,前面补零;delta<0 说明静音过多,从开头裁掉 / pad-or-trim head.
    leading_delta = target_samples - leading_silence_samples
    if leading_delta > 0:
        waveform = F.pad(waveform, (leading_delta, 0), "constant", 0.0)
    else:
        # 裁剪量取 min,防止 -leading_delta 超过当前长度导致越界 / clamp trim to current length.
        trim_from_start = min(-leading_delta, int(waveform.size(-1)))
        waveform = waveform[..., trim_from_start:]

    # 尾部同理:不足则在末尾补零 / pad tail if trailing silence is too short.
    trailing_delta = target_samples - trailing_silence_samples
    if trailing_delta > 0:
        return F.pad(waveform, (0, trailing_delta), "constant", 0.0)

    trim_from_end = min(-trailing_delta, int(waveform.size(-1)))
    if trim_from_end <= 0:
        return waveform
    # 用负索引切掉末尾 trim_from_end 个采样点 / drop the last trim_from_end samples.
    return waveform[..., :-trim_from_end]


def compute_num_audio_tokens(
    num_samples: int, *, audio_samples_per_llm_token: int
) -> int:
    """由采样点数换算出 audio token 个数 / convert sample count to audio-token count.

    每个 LLM token 在波形侧对应固定的 `audio_samples_per_llm_token` 个采样点(token hop)。
    本函数要求输入长度必须已经对齐(用 align_length / pad_waveform_align_only 处理过),
    否则整除不尽,说明上游漏了对齐步骤,直接抛错而非默默丢弃零头 / fail loudly, not silently.

    参数 / Args
        num_samples: 已对齐的采样点数。
        audio_samples_per_llm_token: 每个 token 对应的采样点数(hop)。

    返回 / Returns
        audio token 个数 = num_samples // hop。

    异常 / Raises
        ValueError: 当 num_samples 不是 hop 的整数倍时(未对齐)。
    """
    if num_samples % audio_samples_per_llm_token != 0:
        raise ValueError(
            f"Waveform length {num_samples} is not aligned to token hop {audio_samples_per_llm_token}."
        )
    return num_samples // audio_samples_per_llm_token
