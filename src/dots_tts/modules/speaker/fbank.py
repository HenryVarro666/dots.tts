"""Speaker-branch fbank (Kaldi 风格 mel filter-bank) 特征提取小工具。

本文件做什么 / What this file does
-----------------------------------
为 **speaker / 音色条件** 分支统一产出 Kaldi 风格的 log mel filter-bank
(fbank) 特征。它把"采样率对齐 → mel 特征提取 → 均值归一化"这套 CAM++
x-vector 期望的固定预处理流程封装成一个函数，并把所有超参(16 kHz、80 个
mel bin、不加 dither、做 mean normalization)钉死成模块级常量，保证训练与
推理两侧拿到的 fbank 完全一致。

在数据流里的位置 / Where it sits
--------------------------------
参考音频 waveform
  → extract_speaker_fbank  (本文件: 重采样到 16 kHz + 提 fbank)
  → CAM++ speaker encoder  (modules/speaker/encoder.py / campplus.py)
  → x-vector (说话人 embedding)
  → 作为音色条件喂给 flow-matching DiT 声学头。
即:本文件是从原始波形通往 x-vector 的第一步预处理。调用方包括
``modules/speaker/encoder.py`` 和 ``data/pipelines/tts_pipeline.py``。

关键符号清单 / Key symbols
--------------------------
- 模块级常量 ``_SPEAKER_FBANK_*``: 锁定 speaker 分支的 fbank 超参,
  其中 ``_SPEAKER_FBANK_N_MELS`` 还被 ``campplus.py`` 复用以确定输入维度。
- ``extract_speaker_fbank``: 唯一对外函数, waveform → fbank 特征。
"""

from __future__ import annotations

import torch

from dots_tts.utils.audio import extract_fbank, high_quality_resample

# --- Speaker 分支 fbank 的固定超参 / fixed hyper-params for the speaker branch ---
# 这些值与 CAM++ x-vector 编码器的训练设定绑定, 不可随意改动: 一旦改了, 训练好的
# speaker encoder 权重就拿不到分布一致的输入。常量集中在此, 便于跨文件复用与对齐。

_SPEAKER_FBANK_SAMPLE_RATE = 16000  # CAM++ 在 16 kHz 上训练; 任何输入都先重采样到这里
_SPEAKER_FBANK_N_MELS = 80  # mel filter-bank 通道数 = fbank 特征维度 D (campplus.py 复用此值)
_SPEAKER_FBANK_MEAN_NORM = True  # 是否做 cepstral/utterance mean normalization (CMN), 见下方说明
_SPEAKER_FBANK_DITHER = 0.0  # 推理用 dither=0 关闭随机抖动, 保证特征可复现/确定性


def extract_speaker_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> torch.Tensor:
    """把一段任意采样率的波形转成 speaker 分支用的 Kaldi 风格 log-mel fbank。

    职责 / Responsibility
    ---------------------
    speaker 分支的"标准入口": 先把波形重采样到固定的 16 kHz, 再用本模块锁定
    的超参提取并(可选)均值归一化 fbank 特征, 喂给下游 CAM++ x-vector 编码器。
    所有超参都来自模块级 ``_SPEAKER_FBANK_*`` 常量, 不暴露给调用方, 以此强制
    训练 / 推理两侧的 fbank 完全一致 (避免 train-infer skew)。

    参数 / Args
    -----------
    waveform: 单通道波形张量。约定形状为 (T,) 或 (1, T) / (C, T);
        多通道时下游 ``extract_fbank`` 只取第 0 个通道。T 为采样点数。
    sample_rate: ``waveform`` 的实际采样率 (Hz)。只有当它 != 16 kHz 时才会触发
        重采样; 若上游已经是 16 kHz, 这里零开销直接透传。

    返回 / Returns
    --------------
    fbank 特征张量, 形状 (T_frames, n_mels) = (帧数, 80)。帧数由 Kaldi 的
    分帧逻辑 (帧长/帧移) 决定, 与输入时长近似成正比。

    设计要点 / Design notes
    -----------------------
    - **为什么先重采样**: CAM++ 在 16 kHz 上训练, 输入采样率必须对齐, 否则
      mel 滤波器组覆盖的频带和帧率都对不上, x-vector 会失真。
    - **mean_norm (CMN)**: 减去每个 mel 维度在时间轴上的均值, 抵消信道 /
      录音设备的恒定偏置, 让 speaker embedding 更鲁棒。
    - **dither=0**: 关闭 Kaldi 默认的随机抖动, 换取确定性/可复现的特征。
    """
    feature_input = waveform
    # 只有采样率不匹配时才重采样: 已是 16 kHz 则跳过, 避免不必要的滤波开销与精度损失。
    if sample_rate != _SPEAKER_FBANK_SAMPLE_RATE:
        # high_quality_resample 用 sinc + Kaiser 窗 (lowpass_filter_width=64) 做高质量重采样,
        # 尽量保留 speaker 相关的高频细节, 不引入混叠。
        feature_input = high_quality_resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=_SPEAKER_FBANK_SAMPLE_RATE,
        )
    # 把所有锁定的超参一次性传给通用 fbank 提取器; 关键字传参保证语义清晰、不会错位。
    return extract_fbank(
        feature_input,
        sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
        n_mels=_SPEAKER_FBANK_N_MELS,
        dither=_SPEAKER_FBANK_DITHER,
        mean_norm=_SPEAKER_FBANK_MEAN_NORM,
    )
