"""Batch collator —— 把一批 per-sample dict 拼成一个 padded batch dict。

本文件 / What this file does
---------------------------
DataLoader 的 ``collate_fn``：上游 pipeline(见 ``pipelines/tts_pipeline.py``)
把每条样本处理成一个 dict(含 ``input_ids`` / ``labels`` / ``loss_mask`` /
连续波形 ``sample`` / speaker 分支用的 ``fbank`` 等),本文件负责把一个 list
of dict 折叠成单个张量化、按 batch 维度对齐(pad)的 batch dict,直接喂给训练
循环。处于"数据流水线的最后一站、模型 forward 的前一步"。

In the data flow / 数据流位置
    raw sample → (source adapter) → (tts_pipeline 处理) → per-sample dict
        → **PadCollator(本文件)** → padded batch dict → 模型 forward

关键概念 / Key ML concepts
    - ``input_ids`` / ``labels``: 自回归主干(Qwen2.5-1.5B)的输入 token 与
      训练目标;变长,需要按最长序列 pad。
    - ``loss_mask``: 标记哪些位置参与 loss(1.0)、哪些是 prompt/pad(0.0);
      pad 处补 0.0,天然不计 loss。
    - ``sample``: **连续潜在 TTS** 的原始波形(无离散音频 token),后续由
      AudioVAE(BigVGAN) 编码到连续 latent;变长,按最长波形 pad。
    - ``fbank``: 送进 CAM++ x-vector speaker encoder 的 log-mel 特征,提供
      音色/说话人条件;形状 (T_frames, 80),变长,按帧数 pad。
    - 长度向量(``*_lengths`` / ``num_*``): 记录每条样本 pad 前的真实长度,
      下游据此构造 attention mask / 还原有效区域。

关键类 / Key classes
    - :class:`PadCollator` —— 唯一的导出物,作为 ``DataLoader(collate_fn=...)``。
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence  # 变长序列按 batch 维右侧 pad 对齐


class PadCollator:
    """按长度 pad 的 collate 函数 / length-padding collate function。

    职责 / Responsibility
        从一批 per-sample dict 中抽取 ``input_ids`` / ``labels`` /
        ``loss_mask`` / 波形 ``sample`` / ``fbank`` 等字段,把变长字段沿 batch
        维做 right-padding 对齐,并附带每条样本 pad 前的真实长度,返回单个
        batch dict 供训练循环使用。

        额外做了一件提效的事:**按 ``sample_length`` 降序重排整个 batch**
        (见 :meth:`__call__`)。把长样本排在前面可以减少 pad 浪费,也便于下游
        某些按长度分组/裁剪的逻辑(e.g. 长度递减更利于 cache 友好的处理)。

    设计动机 / Why
        - 把"张量化 + pad + 收集长度"集中到一处,模型 forward 只面对规整张量。
        - ``pad_token_id`` 的回退链(pad → eos → 0)保证即使 tokenizer 未显式
          设置 pad token 也能拿到一个合法的填充 id,不让 pad 进入 loss(由
          ``loss_mask`` 控制),只用于占位。

    参数 / Args
        tokenizer: HuggingFace tokenizer,这里只用它的 ``pad_token_id`` /
            ``eos_token_id`` 来决定文本序列的填充值。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        # 回退链 / fallback: 有些 tokenizer 没单独的 pad token,退而用 eos;
        # 若 eos 也缺失或为 falsy,最后兜底用 0。pad 处不计 loss,id 取值不敏感。
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id or 0

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """把一批样本折叠成 padded batch dict / collapse samples into a batch.

        参数 / Args
            samples: 长度为 B 的 list,每个元素是上游 pipeline 产出的样本 dict,
                至少包含 ``sample_length`` / ``input_ids`` / ``labels`` /
                ``loss_mask`` / ``sample``(波形) / ``fbank`` / ``fid`` 以及各类
                计数字段。

        返回 / Returns
            一个 batch dict。变长字段已 right-pad 到本 batch 内的最大长度,
            并配有 pad 前真实长度的 ``*_lengths`` / ``num_*`` 向量(均 (B,)):
                - ``input_ids`` / ``labels``: (B, L_text),文本 pad 值用
                  ``pad_token_id``。
                - ``loss_mask``: (B, L_text),pad 值 0.0 → 填充位天然不计 loss。
                - ``sample``: (B, 1, T_wav),波形;先 squeeze 掉 mono 通道再 pad,
                  最后 ``unsqueeze(1)`` 补回通道维。
                - ``fbank``: (B, T_frames, 80),speaker 分支 log-mel 特征。

        说明 / Notes
            空 batch 直接报错——DataLoader 不应给出空 list,提前失败便于定位。
        """
        if not samples:
            raise ValueError("PadCollator received an empty sample list.")

        # 按波形长度 sample_length 降序重排 batch:长样本在前,减少 pad 浪费,
        # 也让下游按长度递减处理更高效。先排索引再据此取样本,保持稳定映射。
        order = sorted(
            range(len(samples)),
            key=lambda idx: samples[idx]["sample_length"],
            reverse=True,
        )
        ordered = [samples[idx] for idx in order]

        # 文本 token:list → LongTensor,后续 pad_sequence 会对齐到最长序列。
        input_ids = [
            torch.tensor(sample["input_ids"], dtype=torch.long) for sample in ordered
        ]
        labels = [
            torch.tensor(sample["labels"], dtype=torch.long) for sample in ordered
        ]
        # loss_mask 用 float(后续要与 logits/loss 逐元素相乘),pad 处补 0.0。
        loss_masks = [
            torch.tensor(sample["loss_mask"], dtype=torch.float32) for sample in ordered
        ]
        # 波形原始形状 (1, T) 单声道;squeeze(0) 去掉 mono 通道得到 (T,),
        # 以便 pad_sequence 沿时间维对齐,稍后再补回通道维。
        waveforms = [sample["sample"].squeeze(0) for sample in ordered]
        # fbank 每条形状 (T_frames, 80),已是张量,直接收集交给 pad_sequence。
        fbank = [sample["fbank"] for sample in ordered]

        return {
            "fids": [sample["fid"] for sample in ordered],
            "source_names": [sample.get("source_name") for sample in ordered],
            "input_ids": pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            # 真实长度取自 pad 前的原始 list 长度,而非 padded 张量,(B,)。
            "input_ids_lengths": torch.tensor(
                [len(sample["input_ids"]) for sample in ordered],
                dtype=torch.long,
            ),
            "labels": pad_sequence(
                labels,
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "loss_mask": pad_sequence(
                loss_masks,
                batch_first=True,
                padding_value=0.0,
            ),
            # 波形:(B, T_wav) 经 unsqueeze(1) → (B, 1, T_wav),补回 mono 通道维。
            "sample": pad_sequence(
                waveforms,
                batch_first=True,
                padding_value=0.0,
            ).unsqueeze(1),
            "sample_lengths": torch.tensor(
                [sample["sample_length"] for sample in ordered],
                dtype=torch.long,
            ),
            "num_text_tokens": torch.tensor(
                [sample["num_text_tokens"] for sample in ordered],
                dtype=torch.long,
            ),
            "num_audio_tokens": torch.tensor(
                [sample["num_audio_tokens"] for sample in ordered],
                dtype=torch.long,
            ),
            # fbank:沿帧维 pad → (B, T_frames_max, 80);下游用 fbank_lengths 还原。
            "fbank": pad_sequence(
                fbank,
                batch_first=True,
                padding_value=0.0,
            ),
            "fbank_lengths": torch.tensor(
                [sample["fbank_length"] for sample in ordered],
                dtype=torch.long,
            ),
        }
