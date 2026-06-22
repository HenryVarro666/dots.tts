"""Audio helpers used by the retained train/infer pipeline.

本文件 / What this file does
============================
两个无状态的底层音频工具函数, 被 train 与 infer 两侧共用:

- ``high_quality_resample``: 高质量重采样 (sinc interpolation + Kaiser 窗低通)。
- ``extract_fbank``: 提取 Kaldi 风格 log-mel fbank 特征。

数据流位置 / Where it sits in the pipeline
------------------------------------------
这是 speaker (音色条件) 分支与数据预处理链路的最底层砖块, 不含任何模型权重:

1. ``high_quality_resample`` 把任意采样率的参考音频统一到目标采样率
   (speaker 分支 → 16 kHz CAM++ 输入; tts_pipeline 训练侧 → 24 kHz 一类),
   调用方见 ``modules/speaker/fbank.py`` / ``data/pipelines/tts_pipeline.py`` /
   ``runtime.py``。
2. ``extract_fbank`` 在重采样后的波形上抽 log-mel fbank, 喂给 CAM++ x-vector
   编码器换取 speaker embedding (音色/说话人条件)。

设计取向 / Design stance
------------------------
这里只做 "正确且确定性" 的信号处理, 一切可调超参 (n_mels / dither / mean_norm)
都由上层调用方显式传入, 本文件不锁死任何业务约定, 以保持复用性。
"""

from __future__ import annotations

import torch
import torchaudio.compliance.kaldi as Kaldi
import torchaudio.functional as AF


def high_quality_resample(x, orig_sr, target_sr):
    """高质量重采样: 用 sinc interpolation + Kaiser 窗低通把 ``x`` 从 ``orig_sr`` 转到 ``target_sr``。

    职责 / Responsibility
    ---------------------
    把波形从一个采样率转到另一个采样率, 同时尽量避免 aliasing (混叠) 与高频细节损失。
    重采样数学上 = 先理想低通滤波 (抗混叠) 再按新栅格取样; torchaudio 用一个有限长的
    windowed-sinc FIR kernel 来逼近这个理想低通。

    参数 / Args
    -----------
    x: 输入波形张量, 形状 (..., T_in), 最后一维是时间采样点。批/通道维原样保留。
    orig_sr: 输入采样率 (Hz)。
    target_sr: 目标采样率 (Hz)。

    返回 / Returns
    --------------
    重采样后的波形, 形状 (..., T_out), 其中 T_out ≈ T_in * target_sr / orig_sr。

    设计要点 / Design notes
    -----------------------
    - ``resampling_method="sinc_interp_kaiser"``: 用 Kaiser 窗加窗的 sinc 核 (而非默认的
      Hann 窗), 阻带衰减更可控, 适合对音质敏感的 speaker / TTS 特征提取。
    - ``lowpass_filter_width=64``: 单侧 sinc 旁瓣的截断宽度 (zero crossings 数)。越大 ⇒
      FIR 越长 ⇒ 过渡带越陡、抗混叠越好, 但计算更贵。64 是偏 "高质量" 的取值。
    - ``rolloff=0.95``: 抗混叠低通的截止频率定在 Nyquist 的 0.95 倍, 留一点过渡带余量,
      牺牲极少的最高频换取更干净的阻带 (减少混叠泄漏)。
    """
    return AF.resample(
        x,
        orig_freq=orig_sr,
        new_freq=target_sr,
        lowpass_filter_width=64,  # sinc 核单侧零交叉数: 越大滤波越锐利、抗混叠越好, 但更慢
        rolloff=0.95,  # 截止频率 = 0.95 * Nyquist, 留过渡带余量, 压低混叠泄漏
        resampling_method="sinc_interp_kaiser",  # Kaiser 窗 sinc, 阻带衰减更可控 (优于默认 Hann)
    )


def extract_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int,
    dither: float = 0.0,
    mean_norm: bool = False,
) -> torch.Tensor:
    """提取 Kaldi 风格的 log-mel filterbank (fbank) 特征。

    职责 / Responsibility
    ---------------------
    对单通道波形做短时分帧 → STFT → mel 滤波器组 → 取 log, 得到下游 speaker /
    声学分支常用的 log-mel fbank 表示。这里直接复用 ``torchaudio`` 包装的 Kaldi
    实现, 保证与经典 Kaldi/ESPnet 配方的特征数值对齐 (便于复用预训练的 CAM++ 等)。

    参数 / Args
    -----------
    waveform: 输入波形。接受 (T,) 单声道, 或 (C, T) 多声道; 多声道时**只取第 0 通道**。
    sample_rate: 波形采样率 (Hz)。须与上游重采样后的实际采样率一致, 否则 mel 滤波器
        覆盖的频带会错位。
    n_mels: mel 滤波器组的频带数 = 输出特征维度 D (如 speaker 分支用 80)。
    dither: Kaldi dithering 量级。>0 时给信号加极小随机噪声以避免 log(0)/数值死区;
        推理时通常设 0 以保证特征**确定性可复现**。
    mean_norm: 是否做 utterance-level cepstral mean normalization (CMN), 见下。

    返回 / Returns
    --------------
    fbank 特征张量, 形状 (T_frames, n_mels) = (帧数, D)。帧数由 Kaldi 的帧长/帧移
        默认配置和输入时长共同决定, 约与时长成正比。

    设计要点 / Design notes
    -----------------------
    - **强制单通道**: Kaldi.fbank 期望 (channel, time) 且按单声道处理, 这里把 1D 升成
      (1, T), 把多声道裁成第 0 通道, 统一输入形状, 避免把声道误当成 batch。
    - **mean_norm (CMN)**: 沿时间轴 (dim=0) 减去每个 mel 维的均值, 抵消信道/录音设备
      的恒定频谱偏置, 提升 speaker embedding 对录音条件的鲁棒性。
    """
    # 归一到 (1, T) 单通道: Kaldi.fbank 按 (channel, time) 取信号, batch 维不能混进来
    if waveform.ndim == 1:
        feature_input = waveform.unsqueeze(0)  # (T,) -> (1, T)
    elif waveform.ndim == 2:
        # 已是 (C, T): 单声道直接用, 多声道只保留第 0 通道 (用切片而非索引以保住 2D 形状)
        feature_input = waveform if waveform.size(0) == 1 else waveform[0:1, :]
    else:
        raise ValueError(
            f"FBank expects a 1D or 2D waveform, got shape {tuple(waveform.shape)}."
        )
    features = Kaldi.fbank(
        feature_input,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        dither=dither,
    )
    if mean_norm:
        # CMN: 沿时间轴减去逐 mel 维均值, 去掉信道恒定偏置 (keepdim 以便广播相减)
        features = features - features.mean(dim=0, keepdim=True)
    return features
