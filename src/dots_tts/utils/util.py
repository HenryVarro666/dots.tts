"""杂项工具函数集合 / Miscellaneous utility helpers.

本文件汇集 dots.tts 训练与推理通用的小工具，与具体模型结构解耦，被框架各处复用：
This file gathers small, model-agnostic helpers reused across dots.tts training and
inference; none of them depend on the TTS architecture itself.

关键函数清单 / Key functions:
- ``get_dtype``            : 把字符串(如 "bf16")解析成 ``torch.dtype``，方便配置文件驱动精度选择。
                            Parse a string (e.g. "bf16") into a ``torch.dtype`` for config-driven precision.
- ``seed_everything``      : 统一设置 random / numpy / torch(含 CUDA) 随机种子并开启确定性，保证可复现。
                            Seed all RNGs (random / numpy / torch + CUDA) and force determinism for reproducibility.
- ``mask_data``            : 按布尔 mask 把张量中被屏蔽的位置填成常量(常用于 padding / CFG 条件丢弃)。
                            Fill masked positions of a tensor with a constant (padding / CFG condition dropout).
- ``get_mask_from_lengths``: 由各样本有效长度构造 padding mask，处理 batch 内变长序列。
                            Build a padding mask from per-sample valid lengths for variable-length batches.
- ``scalar_as_float``      : 把可能是 0-d tensor 的标量安全转成 Python ``float``(常用于日志记录)。
                            Safely convert a possibly-tensor scalar into a Python ``float`` (e.g. for logging).
"""

import random
from typing import Any

import numpy as np
import torch


def get_dtype(x):
    """把精度字符串解析成 ``torch.dtype`` / Resolve a precision string to a ``torch.dtype``.

    支持 bf16 / fp16 / fp32 三类(每类接受多种写法)，便于从配置文件或命令行用字符串指定计算精度。
    Accepts bf16 / fp16 / fp32 (several aliases each) so precision can be set via config/CLI strings.

    Args:
        x: 精度名字符串，大小写不敏感 / case-insensitive precision name.
    Returns:
        对应的 ``torch.dtype`` / the matching ``torch.dtype``.
    Raises:
        ValueError: 无法识别的精度名 / when the string is not a recognized precision.
    """
    if x.lower() in ("bf16", "torch.bfloat16", "bfloat16"):
        return torch.bfloat16
    if x.lower() in ("fp16", "torch.float16", "float16"):
        return torch.float16
    if x.lower() in ("fp32", "torch.float32", "float32"):
        return torch.float32
    raise ValueError("Unsupported dtype value.")


def seed_everything(seed: int = 42):
    """统一播种所有随机源并开启确定性 / Seed every RNG and enable deterministic mode.

    一次性固定 Python ``random``、NumPy、PyTorch(CPU 与所有 CUDA 设备) 的随机种子，
    使采样/初始化/数据增强可复现 —— 对调试 flow-matching 采样与 CFG 行为尤其重要。
    Pins the seeds of Python ``random``, NumPy and PyTorch (CPU + all CUDA devices) in one call,
    making sampling / init / augmentation reproducible — handy when debugging flow-matching
    sampling and CFG behavior.

    Args:
        seed: 随机种子，默认 42 / RNG seed, defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 当前 CUDA 设备 / current CUDA device
    torch.cuda.manual_seed_all(seed)  # 多卡场景下覆盖所有 GPU / cover every GPU in multi-device runs
    # cudnn 确定性: 牺牲部分速度换取可复现的卷积算法选择 / trade speed for reproducible cuDNN kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_data(x, mask, masking_value=0.0):
    """把 ``mask`` 为 True 的位置替换成常量 / Replace positions where ``mask`` is True with a constant.

    典型用途: 屏蔽 padding 帧、或在 classifier-free guidance (CFG) 训练时随机丢弃条件(置 0)。
    Typical uses: zero out padding frames, or drop conditioning to a constant during CFG training.

    Args:
        x:            待屏蔽张量，形状如 (B, T, D) / tensor to mask, e.g. (B, T, D).
        mask:         布尔张量，True 处会被替换；维度可少于 ``x`` / bool tensor, True positions get replaced.
        masking_value: 填充值，可为标量或可广播张量 / fill value, a scalar or a broadcastable tensor.
    Returns:
        与 ``x`` 同形状、被屏蔽位置替换后的张量 / tensor of ``x``'s shape with masked positions filled.
    """
    # mask 维度可能比 x 少(如 (B, T) vs (B, T, D))，末尾补轴以便沿特征维广播
    # broadcast mask up to x's rank by appending trailing dims (e.g. (B, T) -> (B, T, 1))
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    if isinstance(masking_value, torch.Tensor):
        # 张量填充值: 显式扩展到 x 形状再按 mask 选择 / expand tensor fill to x's shape, then select
        return torch.where(mask, masking_value.expand_as(x), x)
    # 标量填充值: 构造同 dtype/device 的常量张量供 where 选择 / build a constant tensor matching x's dtype/device
    return torch.where(
        mask, torch.full(x.shape, masking_value, dtype=x.dtype, device=x.device), x
    )


def get_mask_from_lengths(lengths, max_len=None):
    """由各样本有效长度构造 padding mask / Build a padding mask from per-sample valid lengths.

    batch 内序列长度不一时，用每条样本的真实长度生成一个布尔表: 有效帧为 True、padding 帧为 False。
    For variable-length batches, turn each sample's true length into a bool table:
    True for valid frames, False for padding.

    Args:
        lengths: 1-D 张量 (B,)，各样本有效长度 / 1-D tensor (B,) of valid lengths.
        max_len: mask 时间轴长度，缺省取 batch 内最大长度 / mask time length, defaults to the batch max.
    Returns:
        布尔 mask，形状 (B, max_len) / bool mask of shape (B, max_len).
    """
    if max_len is None:
        max_len = torch.max(lengths).item()
    # 在 lengths 所在设备上生成 [0, max_len) 的位置索引向量 / position indices [0, max_len) on lengths' device
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(lengths.device))
    # 广播比较: 位置 < 该样本长度即为有效帧 / broadcast compare: position < length marks valid frames
    return (ids < lengths.unsqueeze(1)).bool()


def scalar_as_float(value: Any) -> float:
    """把任意标量(可能是 0-d tensor) 安全转成 Python ``float`` / Coerce any scalar to a Python ``float``.

    常用于把 loss 等单值张量写进日志: 先 ``detach`` 脱离计算图、转 float32 再 ``item()``，
    避免梯度泄漏与半精度溢出/精度问题。非张量输入直接 ``float()``。
    Used e.g. to log scalar tensors like losses: ``detach`` from the graph, cast to float32, then
    ``item()`` — avoids holding gradients and dodges half-precision overflow/precision issues.
    Non-tensor inputs are passed straight to ``float()``.

    Args:
        value: 标量值，张量或 Python 数值均可 / a scalar, tensor or plain Python number.
    Returns:
        Python ``float`` 标量 / a Python ``float``.
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)
