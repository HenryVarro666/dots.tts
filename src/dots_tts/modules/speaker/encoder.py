"""Speaker x-vector 特征前端（speaker conditioning frontend）。

本文件把 3D-Speaker 的 **CAM++**（见 campplus.py）包成一个即插即用的模块
`SpeakerXVectorFeatures`：吃原始 waveform，吐出一个定长 **x-vector / speaker
embedding**（音色/声纹向量），供声学头（flow-matching DiT）做 speaker conditioning。
它负责 CAM++ 之外的全部「脏活」：重采样到 16 kHz、抽 80 维 fbank、把过长音频做
时间裁剪（random crop），并对外暴露「直接喂预算好的 fbank」这条快路。

在 dots.tts 推理/训练数据流里的位置（zero-shot 音色克隆）：
    参考音频 waveform (B, T)  或  (B, 1, T)
      └─> _crop_audio        : 截到 <= max_audio_seconds，训练时随机偏移（random offset）
            └─> _extract_fbank_batch : 重采样 16 kHz + 逐条抽 fbank -> (B, T_feat, 80)
                  └─> CAMPPlus(fbank, lengths) -> speaker embedding (B, 512)
                        └─> 作为条件注入 DiT，决定「用谁的嗓音」生成
    若调用方已自带 fbank，则跳过抽取，改用 _crop_fbank 在特征域做对齐裁剪。

设计要点：
    - CAM++ 全程冻结（requires_grad=False）+ forward 上 @torch.no_grad，只当固定特征
      提取器（feature extractor），不参与梯度更新。
    - autocast 显式关闭（device_type="cuda", enabled=False）：fbank/统计池化对数值精度
      敏感，强制走 fp32 以保证 x-vector 稳定（numerical stability）。

关键类/函数清单：
    - SpeakerXVectorFeatures : 顶层模块，封装 CAM++ + fbank 提取 + 重采样 + 裁剪。
    - _normalize_lengths     : 把可选的 lengths（或 None）规整成合法的 (B,) long 张量。
    - _crop_audio            : 时间域裁剪，长音频随机选起点，返回起点供特征域对齐。
    - _crop_fbank            : 把时间域的裁剪窗口按帧率折算到特征域，做同步裁剪。
    - _extract_fbank_batch   : 逐样本抽 fbank 后 pad 成 batch，得到 (B, T_feat, 80)。
"""

import math
import random

import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from dots_tts.modules.speaker.campplus import CAMPPlus
from dots_tts.modules.speaker.fbank import (
    _SPEAKER_FBANK_N_MELS,
    _SPEAKER_FBANK_SAMPLE_RATE,
    extract_speaker_fbank,
)


class SpeakerXVectorFeatures(nn.Module):
    """Speaker embedding extractor based on 3D-Speaker CAM++.

    职责：把任意采样率的参考 waveform 转成定长 **x-vector / speaker embedding**，
    作为 DiT 声学头的音色条件（speaker conditioning）。本类是 CAM++ 的「外壳」，
    把重采样、fbank 提取、时间裁剪都收口在这里，CAM++ 本身只负责 fbank -> embedding。

    关键参数：
        sample_rate            : 输入 waveform 的采样率；若 != 16 kHz 则建一个 Resample。
        campplus_embedding_size: 输出 x-vector 维度（默认 512）。
        max_audio_seconds      : 喂给 CAM++ 的最长秒数；超长则裁剪（<=0 表示不裁）。

    forward 返回：speaker embedding，形状 (B, campplus_embedding_size)。

    设计说明：CAM++ 参数在此被冻结（requires_grad=False），整个模块只做特征提取，
    不学习；这也是为什么 forward 用 @torch.no_grad 并关闭 autocast。
    """

    def __init__(
        self,
        sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
        campplus_embedding_size=512,
        max_audio_seconds=10.0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.max_audio_seconds = float(max_audio_seconds)
        self.model = CAMPPlus(
            feat_dim=_SPEAKER_FBANK_N_MELS,
            embedding_size=campplus_embedding_size,
        )
        self.resample = None
        # 仅当输入采样率与 CAM++ 期望的 16 kHz 不一致时才需要重采样器；相等则省掉这步开销。
        if self.sample_rate != _SPEAKER_FBANK_SAMPLE_RATE:
            self.resample = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=_SPEAKER_FBANK_SAMPLE_RATE,
            )

        # 冻结 CAM++：作为固定的 x-vector 提取器使用，不随主模型一起训练。
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _normalize_lengths(lengths, batch_size, max_length, device, *, min_length):
        """把可选的有效长度 lengths 规整成合法的 (B,) long 张量。

        约定：lengths 为 None 时，认为整个 batch 都「满长」（每条都等于 max_length），
        即没有 padding；否则把它搬到目标 device、转 long，并 clamp 到 [min_length, max_length]
        防止越界（如下游用它做切片或 mask）。min_length 由调用方按场景给：
        audio 维度允许 0（可能整段为空），fbank 维度至少 1（CAM++ 需要 >=1 帧）。
        """
        if lengths is None:
            return torch.full(
                (batch_size,),
                max_length,
                device=device,
                dtype=torch.long,
            )
        return lengths.to(device=device, dtype=torch.long).clamp(
            min=min_length,
            max=max_length,
        )

    def _crop_audio(self, audio, audio_lengths=None):
        """时间域裁剪：把每条音频截到 <= max_audio_seconds，长则随机选起点。

        参数：
            audio         : (B, T_audio) 单声道波形（已在 forward 里压成 2D）。
            audio_lengths : (B,) 各条有效采样点数，或 None（视为满长）。
        返回四元组：
            cropped_audio          : (B, T_crop) 裁剪 + pad 后的波形。
            original_audio_lengths : (B,) 裁剪前的原始有效长度（_crop_fbank 用它换算帧率）。
            cropped_audio_lengths  : (B,) 裁剪后的有效长度。
            starts                 : (B,) 每条的裁剪起点（采样点），供特征域对齐。

        为什么随机起点：训练时对超长参考音频做 **random crop** 是一种数据增强——
        同一条音频每次取不同片段，让 speaker embedding 不依赖固定位置、更鲁棒。
        max_audio_seconds<=0 时直接整段返回（起点全 0），相当于关闭裁剪。
        """
        original_lengths = self._normalize_lengths(
            audio_lengths,
            audio.size(0),
            audio.size(-1),
            audio.device,
            min_length=0,
        )
        if self.max_audio_seconds <= 0:
            # 不裁剪：cropped == original，起点全 0；保持四元组接口不变。
            return audio, original_lengths, original_lengths, torch.zeros_like(
                original_lengths
            )

        # 秒数 -> 最大采样点数（按输入采样率算，裁剪发生在重采样之前）。
        max_input_length = round(self.sample_rate * self.max_audio_seconds)
        cropped_audio = []
        cropped_lengths = []
        starts = []

        for index, total_length_tensor in enumerate(original_lengths):
            total_length = int(total_length_tensor.item())
            cropped_length = min(total_length, max_input_length)
            # 只有当原长严格大于裁剪长时才有可选起点；否则起点固定为 0（取整段）。
            start = (
                random.randint(0, total_length - cropped_length)
                if total_length > cropped_length
                else 0
            )
            cropped_audio.append(audio[index, start : start + cropped_length])
            cropped_lengths.append(cropped_length)
            starts.append(start)

        # 各条裁剪后长度不同，pad 到 batch 内最长，padding_value=0.0（静音）。
        return pad_sequence(
            cropped_audio,
            batch_first=True,
            padding_value=0.0,
        ), original_lengths, torch.tensor(
            cropped_lengths,
            device=audio.device,
            dtype=torch.long,
        ), torch.tensor(starts, device=audio.device, dtype=torch.long)

    def _crop_fbank(
        self,
        fbank,
        fbank_lengths,
        original_audio_lengths,
        cropped_audio_lengths,
        starts,
    ):
        """在特征域复刻 _crop_audio 的同一裁剪窗口（调用方自带 fbank 时走这里）。

        当 fbank 是外部预算好的（而非本类抽的），无法重新抽特征，只能把时间域的
        [start_audio, end_audio) 窗口按帧率（frame rate）线性折算到特征帧索引上，
        对 fbank 做同步裁剪，保证 embedding 对应的就是被裁后的那段音频。

        参数：
            fbank                  : (B, T_feat, F) 外部传入的 fbank。
            fbank_lengths          : (B,) 各条有效帧数，或 None。
            original_audio_lengths : (B,) 裁剪前音频采样点数（折算分母）。
            cropped_audio_lengths  : (B,) 裁剪后采样点数（用于算 end_audio）。
            starts                 : (B,) _crop_audio 选的裁剪起点（采样点）。
        返回：
            cropped_fbank          : (B, T_feat_crop, F) 裁剪 + pad 后的 fbank。
            cropped_fbank_lengths  : (B,) 各条裁剪后帧数。
        """
        original_fbank_lengths = self._normalize_lengths(
            fbank_lengths,
            fbank.size(0),
            fbank.size(1),
            fbank.device,
            min_length=1,
        )
        cropped_fbank = []
        cropped_fbank_lengths = []

        for index, total_feat_length_tensor in enumerate(original_fbank_lengths):
            total_audio_length = int(original_audio_lengths[index].item())
            total_feat_length = int(total_feat_length_tensor.item())
            start_audio = int(starts[index].item())
            end_audio = start_audio + int(cropped_audio_lengths[index].item())

            # 采样点索引 -> 特征帧索引：按「帧数/采样点数」的比例线性映射。
            # 起点 floor、终点 ceil，保证裁出的特征窗口完整覆盖目标音频段（不漏帧）。
            if total_audio_length > 0:
                start_feat = math.floor(
                    start_audio * total_feat_length / total_audio_length
                )
                end_feat = math.ceil(end_audio * total_feat_length / total_audio_length)
            else:
                # 原音频长度为 0（异常/空输入）时无从折算，退化成取 1 帧占位。
                start_feat = 0
                end_feat = 1

            # 钳制成合法区间：start 不超过倒数第二帧，end 至少比 start 大 1 且不越界，
            # 确保 [start_feat:end_feat) 非空（CAM++ 需要 >=1 帧）。
            start_feat = min(start_feat, total_feat_length - 1)
            end_feat = min(max(end_feat, start_feat + 1), total_feat_length)
            cropped_fbank.append(fbank[index, start_feat:end_feat])
            cropped_fbank_lengths.append(end_feat - start_feat)

        return pad_sequence(
            cropped_fbank,
            batch_first=True,
            padding_value=0.0,
        ), torch.tensor(
            cropped_fbank_lengths,
            device=fbank.device,
            dtype=torch.long,
        )

    def _extract_fbank_batch(self, audio, audio_lengths):
        """逐样本抽 80 维 fbank，再 pad 成 batch。

        参数：
            audio         : (B, T_crop) 已裁剪的波形（采样率为 self.sample_rate）。
            audio_lengths : (B,) 各条有效采样点数（裁剪后的长度）。
        返回：
            fbank         : (B, T_feat, 80) pad 后的 fbank（搬回 audio 的 device/dtype）。
            fbank_lengths : (B,) 各条有效帧数。

        实现细节：先重采样到 16 kHz（若需要），并同步把有效长度按比例放缩（向上取整，
        与下游 fbank 帧数对齐）；fbank 提取走 Kaldi，故移到 CPU 逐条抽——这也是为什么
        这里只在 forward 的 no_grad 上下文里调用（纯特征提取，不需要梯度）。
        """
        if self.resample is not None:
            audio = self.resample(audio)
            # 重采样后有效长度按采样率比缩放；ceil 避免把末尾不足一帧的尾巴丢掉。
            audio_lengths = torch.ceil(
                audio_lengths.float()
                * (_SPEAKER_FBANK_SAMPLE_RATE / self.sample_rate)
            ).long()

        audio_cpu = audio.detach().cpu()
        features = []

        for index, valid_length_tensor in enumerate(audio_lengths):
            valid_length = int(valid_length_tensor.item())
            waveform = audio_cpu[index, :valid_length]  # 切掉 padding，只对有效段抽特征
            # 空波形（valid_length==0）会让 Kaldi 报错，用一个零样本占位，至少出 1 帧。
            if waveform.numel() == 0:
                waveform = audio_cpu.new_zeros(1)
            features.append(
                extract_speaker_fbank(
                    waveform,
                    sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
                )
            )

        # extract_speaker_fbank 返回 (T_feat, 80)，故有效帧数取 size(0)。
        fbank_lengths = torch.tensor(
            [feature.size(0) for feature in features],
            device=audio.device,
            dtype=torch.long,
        )
        fbank = pad_sequence(
            features,
            batch_first=True,
            padding_value=0.0,
        ).to(device=audio.device, dtype=audio.dtype)  # 抽完搬回原 device/dtype 喂 CAM++
        return fbank, fbank_lengths

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def forward(
        self, audio, audio_lengths=None, fbank=None, fbank_lengths=None, **_kwargs
    ):
        """waveform -> x-vector / speaker embedding (B, embedding_size)。

        参数：
            audio         : (B, T) 或 (B, 1, T) 单声道波形。
            audio_lengths : (B,) 各条有效采样点数，或 None（满长）。
            fbank         : 可选 (B, T_feat, F) 预算好的 fbank；给了就跳过抽取走快路。
            fbank_lengths : 可选 (B,) fbank 有效帧数。
            **_kwargs     : 吞掉多余关键字参数，方便上层统一接口调用，本类不用。
        返回：speaker embedding (B, embedding_size)。

        装饰器说明：@torch.no_grad 因 CAM++ 冻结、仅做特征提取；@torch.autocast
        关闭混合精度（强制 fp32）以保证 fbank/统计池化的数值稳定（numerical stability）。
        """
        self.model.eval()  # 显式 eval：固定 BatchNorm 的 running stats，不更新
        audio = audio.float()
        # 统一形状到 (B, T)：接受 (B,1,T) 单声道并 squeeze 掉声道维；多声道/其它维度报错。
        if audio.dim() == 3:
            if audio.size(1) != 1:
                raise ValueError(
                    f"Speaker encoder expects mono audio, got shape {tuple(audio.shape)}."
                )
            audio = audio[:, 0]
        elif audio.dim() != 2:
            raise ValueError(
                f"Speaker encoder expects a 2D or 3D audio tensor, got shape {tuple(audio.shape)}."
            )

        # 先在时间域裁剪，拿到起点 starts；无论走哪条 fbank 路径都需要它做对齐。
        audio, original_audio_lengths, cropped_audio_lengths, starts = self._crop_audio(
            audio,
            audio_lengths=audio_lengths,
        )

        if fbank is None:
            # 路径 A：自己抽 fbank（已对裁剪后的 audio 抽，无需再裁特征）。
            fbank, fbank_lengths = self._extract_fbank_batch(
                audio,
                cropped_audio_lengths,
            )
        else:
            # 路径 B：调用方自带 fbank，需在特征域复刻同一裁剪窗口（见 _crop_fbank）。
            if not isinstance(fbank, torch.Tensor):
                raise TypeError("Speaker encoder expects `fbank` to be a torch.Tensor.")
            if fbank.dim() != 3 or fbank.size(0) != audio.size(0):
                raise ValueError(
                    f"Speaker encoder expects `fbank` with shape (B, T, F) and matching batch size, got {tuple(fbank.shape)}."
                )
            fbank, fbank_lengths = self._crop_fbank(
                fbank.to(device=audio.device, dtype=torch.float32),
                fbank_lengths,
                original_audio_lengths,
                cropped_audio_lengths,
                starts,
            )

        # CAM++ 用 fbank_lengths 做 masked statistics pooling，正确忽略 padding 帧后
        # 汇聚成定长向量；返回 (B, embedding_size) 的 x-vector。
        return self.model(fbank, lengths=fbank_lengths)
