"""说话人组件 / Speaker modules.

中文：从参考音频中提取说话人/音色条件(speaker embedding / x-vector)，用于音色克隆与
条件控制。fbank(把波形转成 Mel-fbank 声学特征)、campplus_layers / campplus
(CAM++ 网络结构，TDNN + 通道注意力，输出 x-vector)、encoder(对外封装：音频 -> 定长
说话人向量)。该向量作为条件喂给主干，控制合成音色。

EN: Extracts speaker/timbre conditioning (a speaker embedding / x-vector) from reference
audio for voice cloning and conditional control — fbank (waveform to Mel-fbank features),
campplus_layers / campplus (the CAM++ TDNN + channel-attention network producing the
x-vector) and encoder (the public wrapper: audio -> fixed-length speaker vector).
"""
