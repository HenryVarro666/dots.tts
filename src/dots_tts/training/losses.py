"""Loss aggregation helpers shared by training and validation.

The model returns masked per-token / per-patch loss tensors. This module turns
them into numerators/denominators for logging, combines configured loss weights,
and provides distributed reduction helpers.

中文说明 / Chinese overview
==========================
本文件是分布式训练里的 **loss 聚合工具层**,本身不计算任何 loss——它只负责把
模型前向算出的"逐元素 loss + 同形 mask"折叠(collapse)成可累加、可跨 rank
求和、最终可加权平均的标量统计量。

在数据流里的位置:
    model.forward()  ->  每个 loss 项给出 (loss, mask) 两个同形张量
                         (loss 是 per-token / per-patch 的未 reduce 值;
                          mask 标记每个位置的有效性/权重,padding 处为 0)
    本文件           ->  把每项折叠成 numerator = Σ(loss*mask)、
                         normalizer = Σ(mask);跨 batch / 跨 rank 把这两个量
                         相加后再做 numerator/normalizer,得到无偏的加权平均。
    训练循环         ->  用 compute_gradient_loss() 拿到供 backward() 的标量;
                         用 reduce_loss_statistics() 拿到供日志记录的指标。

为什么要拆成"分子/分母"而不是直接取平均?
    因为不同 sample / batch / rank 的有效 token 数(mask 之和)不同。若各自先求
    平均再平均,等于给小样本过高权重,得到的是有偏估计。先各自累加分子与分母、
    最后一次性相除,才能得到按 token 数加权的、与切分方式无关的无偏平均;这也是
    跨 rank 聚合时必须遵守的契约:**先求和分子与分母,最后才相除**。

关键类 / 函数清单:
    LossTerm                              逐元素 loss 张量 + 同形 mask 的数据类
    collapse_loss_terms                   折叠成 (numerators, normalizers)
    collapse_loss_terms_by_source         按数据集来源(source)分组折叠
    reduce_loss_statistics                把聚合后的分子/分母还原成加权平均指标
    compute_gradient_loss                 组装供 backward() 的标量 loss
    sum_named_scalars_across_ranks        跨 rank all-reduce 求和(含键的并集对齐)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
import torch.distributed as dist

from dots_tts.utils.util import scalar_as_float


@dataclass(frozen=True)
class LossTerm:
    """A loss tensor paired with a same-shape mask.

    ``loss`` stores unreduced per-element values.
    ``mask`` stores the weighting/validity for the same positions.

    中文:一个 loss 项 = 未 reduce 的逐元素 loss + 与之**同形**的 mask。
    - loss: 形状如 (B, T) 或 (B, T, ...) 的逐 token / 逐 patch loss,尚未求和或平均。
    - mask: 同形的有效性 / 权重张量;padding、被 mask 掉的位置取 0,有效位置取 1
            (或软权重)。后续 Σ(loss*mask)/Σ(mask) 才把它折叠成标量平均。
    设计成 frozen dataclass 是为了让一个 loss 项作为不可变值在聚合管线里流转,
    并通过 __post_init__ 在构造时就强制 loss 与 mask 形状一致——这是后续所有
    逐元素相乘 / 求和操作正确的前提。
    """

    loss: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        # 形状不一致会让 loss*mask 触发广播或静默错位,因此在构造期直接拒绝。
        if self.loss.shape != self.mask.shape:
            raise ValueError(
                "LossTerm expects loss and mask to have the same shape, "
                f"but got {tuple(self.loss.shape)} and {tuple(self.mask.shape)}."
            )


LossTerms: TypeAlias = dict[str, LossTerm]
LossMasks: TypeAlias = dict[str, torch.Tensor]


def _safe_average(numerator: Any, denominator: Any) -> Any:
    """Average safely when a mask may produce a zero denominator.

    中文:做 numerator/denominator 的"安全除法"。当某个 loss 项在本 batch / 本
    rank 完全没有有效位置时,denominator(=Σmask)会是 0,直接除会得到 NaN/Inf
    并污染梯度。这里在分母 <= 0 时返回 0,否则正常相除。
    同时支持两条路径:numerator 为 Tensor 时保留计算图(供 backward),为标量时
    走纯 Python 浮点(供日志)。
    """
    if isinstance(numerator, torch.Tensor):
        denom = denominator
        if not isinstance(denom, torch.Tensor):
            # 分母是标量时,用 numerator 派生同 device/dtype 的张量,避免跨设备运算。
            denom = numerator.new_tensor(float(denominator))
        if float(denom.detach().item()) <= 0.0:
            # 无有效位置:返回 0 但保留计算图(乘 0.0 而非 return 0.0),
            # 这样该项在 backward 时贡献 0 梯度而不是断开图 / 报错。
            return numerator * 0.0
        # clamp_min(1.0) 仅作数值兜底:能走到这分支说明 denom>0,
        # clamp 只防极小正值导致的放大,不改变正常情形的结果。
        return numerator / denom.clamp_min(1.0).to(numerator.dtype)

    # 纯标量路径:用于日志聚合,不涉及梯度。
    denom = float(denominator)
    if denom <= 0.0:
        return 0.0
    return float(numerator) / denom


def _as_weight_map(loss_config) -> dict[str, float]:
    """Extract ``*_weight`` fields from config into ``*_loss`` weights.

    中文:把 loss 配置里的 ``<x>_weight`` 字段映射成 ``<x>_loss`` 这个键的权重。
    约定:配置用 ``flow_weight=1.0`` 形式,而 loss 项在管线里的名字是 ``flow_loss``;
    这里统一把 ``_weight`` 后缀(7 个字符)替换成 ``_loss``,使权重表能直接按
    loss 名查权重。没在表里的项默认权重 1.0(见调用处 ``weights.get(name, 1.0)``)。
    """
    weights = {}
    for name, value in loss_config.model_dump().items():
        if name.endswith("_weight"):
            # name[:-7] 去掉结尾的 "_weight"(共 7 字符),再拼上 "_loss"。
            weights[f"{name[:-7]}_loss"] = float(value)
    return weights


def accumulate_named_scalars_(
    target: dict[str, float],
    source: dict[str, float],
) -> dict[str, float]:
    """In-place add ``source`` scalar values into ``target`` by key.

    中文:把 source 的标量按键就地累加进 target(target 须为 defaultdict(float)
    或已含同名键)。用于跨 micro-batch / 跨梯度累加步把分子、分母逐步攒起来。
    函数名末尾的下划线沿用 PyTorch 约定,表示**原地(in-place)修改**。
    """
    for name, value in source.items():
        target[name] += float(value)
    return target


def to_host_named_scalars(values: dict[str, Any]) -> dict[str, float]:
    """Convert scalar tensors into plain Python floats for logging/serialization.

    中文:把 ``{名字: 标量张量 或 数}`` 统一转成 ``{名字: float}``。标量张量会被
    ``.detach().float().item()`` 搬回 host(脱离计算图),便于写日志 / 序列化,
    避免把 GPU 张量带进日志系统造成隐式同步或显存泄漏。
    """
    return {name: scalar_as_float(value) for name, value in values.items()}


def collapse_loss_masks(
    loss_masks: LossMasks,
) -> dict[str, Any]:
    """Reduce each loss mask to its total effective weight.

    中文:把每个 mask 求和成"总有效权重"标量(= 该项的归一化分母 normalizer)。
    用于只需要分母、不需要分子的场景(例如单独统计各项有多少有效 token)。
    """
    return {name: mask.sum() for name, mask in loss_masks.items()}


def collapse_loss_terms(
    loss_terms: LossTerms,
    *,
    indices: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert masked per-sample loss terms into ``sum(loss*mask)`` statistics.

    Returns ``(numerators, normalizers)`` so the caller can aggregate them across
    batches or ranks before taking the final average.

    中文:把每个 LossTerm 折叠成两个标量统计量,这是整个聚合管线的核心一步。
        numerator   = Σ(loss * mask)   分子:有效位置上的 loss 加权和
        normalizer  = Σ(mask)          分母:有效位置的总权重(有效 token 数)
    刻意**只返回分子和分母、不就地相除**:这样调用方能先跨 batch / 跨 rank 把
    两者分别累加,最后才做 numerator/normalizer,从而得到与切分方式无关的无偏
    加权平均(见模块 docstring 的契约)。

    参数:
        loss_terms: {名字: LossTerm};每个 term 的 loss/mask 形如 (B, ...),
                    第 0 维是 batch / sample。
        indices:    可选的样本下标列表;给定时只对这些样本(沿第 0 维)折叠,
                    用于"按数据集来源分组"统计(见 collapse_loss_terms_by_source)。
    返回:
        (numerators, normalizers),两个 {名字: 标量张量};分子保留计算图。
    """
    index = None
    if indices is not None:
        # 取任意一个 term 的 device,把 Python 下标列表搬成同设备的 long 索引张量,
        # 供下面 index_select 使用(所有 term 的第 0 维 batch 对齐,故共用一个索引)。
        first = next(iter(loss_terms.values()))
        index = torch.tensor(indices, device=first.loss.device, dtype=torch.long)

    numerators = {}
    normalizers = {}
    for name, term in loss_terms.items():
        loss = term.loss
        mask = term.mask
        if index is not None:
            # 沿 batch 维(dim=0)挑出指定样本子集,再在子集上做折叠。
            loss = loss.index_select(0, index)
            mask = mask.index_select(0, index)
        # mask 可能是 bool / 整型;转成与 loss 同 dtype 才能正确做逐元素相乘与求和。
        mask = mask.to(loss.dtype)
        numerators[name] = (loss * mask).sum()
        normalizers[name] = mask.sum()
    return numerators, normalizers


def collapse_loss_terms_by_source(
    loss_terms: LossTerms,
    *,
    source_names: list[str | None],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Group collapsed loss statistics by dataset/source name within a batch.

    中文:一个 batch 里可能混合多个数据来源(source,如不同数据集 / 语种)。本函数
    按 source 把样本分组,分别折叠,得到每个来源各自的分子 / 分母,便于分来源监控
    loss。source_names[i] 必须对应 batch 第 i 个样本。

    参数:
        source_names: 长度 = batch_size 的来源名列表(不允许 None);
    返回:
        (numerators_by_source, normalizers_by_source),均为
        {source 名: {loss 名: float}} 的嵌套字典(已转 host float,供日志聚合)。
    """
    first = next(iter(loss_terms.values()))
    batch_size = int(first.loss.size(0))
    if len(source_names) != batch_size:
        # 来源名必须与 batch 一一对应,否则下面按下标分组会错位。
        raise RuntimeError(
            "source_names must align with the batch size for source loss statistics. "
            f"Expected {batch_size}, got {len(source_names)}."
        )

    # 把"样本下标"按来源名归桶:source_indices[来源] = [属于该来源的样本下标...]
    source_indices: dict[str, list[int]] = defaultdict(list)
    for index, source_name in enumerate(source_names):
        if source_name is None:
            raise RuntimeError("source_names must not contain None.")
        source_indices[str(source_name)].append(index)

    numerators_by_source = {}
    normalizers_by_source = {}
    for source_name, indices in source_indices.items():
        # 复用 collapse_loss_terms,仅传入该来源的样本下标子集做折叠。
        numerators, normalizers = collapse_loss_terms(loss_terms, indices=indices)
        numerators_by_source[source_name] = to_host_named_scalars(numerators)
        normalizers_by_source[source_name] = to_host_named_scalars(normalizers)
    return numerators_by_source, normalizers_by_source


def reduce_loss_statistics(
    numerators: dict[str, Any],
    normalizers: dict[str, Any],
    *,
    loss_config,
) -> dict[str, Any]:
    """Turn aggregated numerators/normalizers into averaged metrics.

    The returned mapping includes each individual loss plus a weighted ``loss``
    field assembled from ``loss_config``.

    中文:聚合管线的"收尾"步骤——把已经跨 batch / 跨 rank 累加好的分子与分母,
    逐项做安全除法得到各 loss 的加权平均值,再按配置权重加权求和成一个总 ``loss``
    指标。注意这里产出的是**供日志记录**的指标;供 backward 的标量在
    compute_gradient_loss() 里另算(缩放方式不同)。

    返回 dict 既含每个单项 loss(如 ``flow_loss``),也含汇总键 ``"loss"``。
    """
    weights = _as_weight_map(loss_config)
    reduced = {}
    total_loss: Any = 0.0
    # 按名字排序遍历,保证多 rank / 多次运行的加法顺序一致(可复现、浮点稳定)。
    for name, numerator in sorted(numerators.items()):
        value = _safe_average(numerator, normalizers[name])
        reduced[name] = value
        # 未在权重表里的项默认权重 1.0。
        total_loss = total_loss + value * weights.get(name, 1.0)
    reduced["loss"] = total_loss
    return reduced


def reduce_loss_statistics_by_source(
    numerators_by_source: dict[str, dict[str, float]],
    normalizers_by_source: dict[str, dict[str, float]],
    *,
    loss_config,
) -> dict[str, dict[str, Any]]:
    """Apply :func:`reduce_loss_statistics` independently for each source.

    中文:对每个数据来源(source)各自调用 reduce_loss_statistics,得到分来源的
    加权平均指标 {source 名: {loss 名: 平均值, ..., "loss": 加权和}}。用于在日志里
    分数据集对比 loss。按 source 名排序保证遍历顺序确定。
    """
    return {
        source_name: reduce_loss_statistics(
            numerators,
            normalizers_by_source[source_name],
            loss_config=loss_config,
        )
        for source_name, numerators in sorted(numerators_by_source.items())
    }


def compute_gradient_loss(
    loss_terms: LossTerms,
    *,
    global_normalizers: dict[str, float],
    loss_config,
    ddp_world_size: int,
    gradient_accumulation_steps: int,
) -> torch.Tensor:
    """Build the scalar loss used for ``backward()``.

    ``global_normalizers`` is expected to already include cross-rank totals. The
    final scaling by world size and accumulation steps compensates for the mean
    reduction that DDP/Accelerate applies during gradient synchronization.

    中文:组装真正喂给 ``backward()`` 的标量 loss,是本文件唯一保留计算图、参与
    反传的产出(其余函数都是给日志算指标)。

    关键点:这里用的分母是 **global_normalizers——已经跨所有 rank 求和过的全局
    有效 token 总数**,而不是本 rank 的局部分母。原因是我们想要的是"按全局 token
    数加权的平均梯度",但本 rank 的 backward 只用本地分子 Σ(loss*mask)_local。

    末尾乘 ``ddp_world_size * gradient_accumulation_steps`` 是**反缩放补偿**:
    - DDP / Accelerate 在梯度同步(all-reduce)时对各 rank 梯度取 **均值**(即除以
      world_size),且梯度累加会把 accumulation_steps 次结果相加;
    - 我们用的是全局分母,本身已等价于"求和后再除以全局总量"的口径;
    - 故先在这里乘回 world_size 和 accumulation_steps,抵消框架后续的均值化,
      使最终等效梯度恰好等于"全局 Σ(loss*mask) / 全局 Σmask"对参数的梯度,
      与单卡、不累加时数值一致。
    """
    # 只取本 rank 的局部分子(分母由外部传入的全局量替代,故第二个返回值丢弃)。
    numerators, _ = collapse_loss_terms(loss_terms)
    weights = _as_weight_map(loss_config)

    total_loss: Any = 0.0
    # 排序保证各项相加顺序确定,跨 rank 数值一致。
    for name, numerator in sorted(numerators.items()):
        total_loss = total_loss + _safe_average(
            numerator,
            global_normalizers[name],  # 全局分母:本 rank 局部分子 / 全局总分母
        ) * weights.get(name, 1.0)
    # 反缩放:抵消 DDP 均值化(/world_size)与梯度累加(/accumulation_steps)。
    return total_loss * float(ddp_world_size) * float(gradient_accumulation_steps)


def sum_named_scalars_across_ranks(
    values: dict[str, float],
    *,
    device: torch.device,
) -> dict[str, float]:
    """All-reduce a dict of scalar values and return summed host floats.

    中文:把一个 {名字: 标量} 字典跨所有 rank 求和,返回求和后的 host float 字典。
    典型用途:把各 rank 的局部分子 / 分母累加成全局量(供 compute_gradient_loss 的
    global_normalizers 使用)。

    难点在于**不同 rank 的键集合可能不一致**(例如某 rank 这步没遇到某来源 / 某 loss
    项)。因此先用 _gather_string_union_across_ranks 求出全体 rank 键的**并集并排序**,
    再按这个统一顺序把值打包成同长度向量做 all_reduce——保证每个 rank 在张量同一位置
    放的是同一个键的值,缺失的键补 0。用 float64 打包以减小求和时的精度损失。
    """
    # 先对齐键空间:取所有 rank 键的并集(已排序),保证打包顺序在各 rank 一致。
    names = _gather_string_union_across_ranks(values, device=device)
    if not names:
        return {}

    # 按统一键顺序打包;本 rank 没有的键补 0.0,这样求和后即为全局总量。
    packed = torch.tensor(
        [float(values.get(name, 0.0)) for name in names],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        name: float(value)
        for name, value in zip(names, packed.tolist(), strict=True)
    }


def sum_grouped_named_scalars_across_ranks(
    values: dict[str, dict[str, float]],
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """All-reduce nested ``group -> metric -> value`` scalar mappings.

    中文:sum_named_scalars_across_ranks 的二维版本,用于按来源分组的统计
    {group(来源): {metric(loss 名): 值}}。同样要先对齐键空间——分别求出 group 名
    和 metric 名在所有 rank 上的**并集**,把二维表展平成 (len(group)*len(metric),)
    的向量做一次 all_reduce,再 reshape 回 (group, metric) 二维。缺失项补 0。
    """
    # 第一维键(来源)的全局并集。
    group_names = _gather_string_union_across_ranks(values, device=device)
    if not group_names:
        return {}

    # 第二维键(metric / loss 名)的全局并集:展平遍历所有 group 的内层键。
    metric_names = _gather_string_union_across_ranks(
        (
            metric_name
            for group_values in values.values()
            for metric_name in group_values
        ),
        device=device,
    )
    if not metric_names:
        return {group_name: {} for group_name in group_names}

    # 按 (group, metric) 行优先顺序展平打包成一维;本 rank 缺失的格子补 0.0。
    packed = torch.tensor(
        [
            float(values.get(group_name, {}).get(metric_name, 0.0))
            for group_name in group_names
            for metric_name in metric_names
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    # 一维求和结果 reshape 回二维网格,行=group、列=metric。
    packed = packed.view(len(group_names), len(metric_names))
    return {
        group_name: {
            metric_name: float(packed[group_index, metric_index].item())
            for metric_index, metric_name in enumerate(metric_names)
        }
        for group_index, group_name in enumerate(group_names)
    }


def accumulate_grouped_named_scalars_(
    target: dict[str, dict[str, float]],
    source: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """In-place add nested ``group -> metric -> value`` scalar mappings.

    中文:accumulate_named_scalars_ 的二维版本,把分组统计就地累加进 target。
    若 target 还没有某个 group,先用全 0 初始化该 group 的内层字典再累加;否则
    逐个 metric 加上去。函数名末尾下划线表示**原地修改**(PyTorch 约定)。
    """
    for group_name, values in source.items():
        group_target = target.get(group_name)
        if group_target is None:
            # 首次见到该 group:用与 source 同键、初值 0.0 的字典初始化。
            group_target = {name: 0.0 for name in values}
            target[group_name] = group_target
        for name, value in values.items():
            group_target[name] += float(value)
    return target


def _gather_string_union_across_ranks(
    values: Iterable[str],
    *,
    device: torch.device,
) -> list[str]:
    """Gather the sorted union of string keys across all ranks.

    中文:求"键名集合"在所有 rank 上的**并集并排序**。这是前面那些 all_reduce 求和
    函数的前置步骤——只有先让每个 rank 拿到完全一致、顺序一致的键列表,后续按位置
    打包成张量做 all_reduce 才不会错位。

    实现要点(为什么不直接用 all_gather_object?):这里走纯张量集合通信,避免依赖
    pickle / NCCL 对 object 的支持,更稳健。做法是把字符串列表编码成 UTF-8 字节:
        1. all_gather 各 rank 的字节长度,取 max 作为统一缓冲区长度(集合通信要求
           各 rank 张量等长);
        2. 把本地字节放进 max_size 长的 uint8 buffer(右侧补 0),all_gather 收齐;
        3. 按各自真实长度(size)切片解码,合并进并集,最后排序返回。
    """
    strings = sorted({str(value) for value in values})
    # 单进程 / 未初始化分布式:无需通信,本地键集合即全集。
    if not (dist.is_available() and dist.is_initialized()):
        return strings

    payload = _encode_string_list(strings)
    world_size = dist.get_world_size()
    # 先交换各 rank 的字节长度,以便确定统一的 buffer 尺寸(all_gather 需等长张量)。
    local_size = torch.tensor([len(payload)], device=device, dtype=torch.int64)
    size_tensors = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_tensors, local_size)

    max_size = max(int(size.item()) for size in size_tensors)
    if max_size <= 0:
        # 所有 rank 都没有任何键。
        return []

    # 本地字节放进定长 buffer,不足 max_size 的尾部保持 0(padding)。
    local_bytes = torch.zeros(max_size, device=device, dtype=torch.uint8)
    if payload:
        local_bytes[: len(payload)] = torch.tensor(
            list(payload),
            device=device,
            dtype=torch.uint8,
        )

    gathered_bytes = [
        torch.empty(max_size, device=device, dtype=torch.uint8)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_bytes, local_bytes)

    union = set(strings)
    for size_tensor, byte_tensor in zip(size_tensors, gathered_bytes, strict=True):
        size = int(size_tensor.item())
        if size <= 0:
            continue
        # 按该 rank 的真实字节数切片(丢弃 padding),搬回 host 解码成字符串并入并集。
        union.update(
            _decode_string_list(bytes(byte_tensor[:size].cpu().tolist()))
        )
    return sorted(union)


def _encode_string_list(values: list[str]) -> bytes:
    # 用 NUL(\0)作分隔符把字符串列表拼成一段 UTF-8 字节,供跨 rank 字节传输。
    # 因此键本身不能含 NUL,否则解码会被错误切分——这里提前检查并报错。
    if any("\0" in value for value in values):
        raise ValueError("Distributed scalar keys must not contain NUL characters.")
    return "\0".join(values).encode("utf-8")


def _decode_string_list(payload: bytes) -> list[str]:
    # _encode_string_list 的逆操作:按 NUL 分隔符还原字符串列表。空串返回空列表。
    if not payload:
        return []
    return payload.decode("utf-8").split("\0")


__all__ = [
    "accumulate_grouped_named_scalars_",
    "LossMasks",
    "LossTerm",
    "LossTerms",
    "accumulate_named_scalars_",
    "collapse_loss_masks",
    "collapse_loss_terms",
    "collapse_loss_terms_by_source",
    "compute_gradient_loss",
    "reduce_loss_statistics",
    "reduce_loss_statistics_by_source",
    "sum_grouped_named_scalars_across_ranks",
    "sum_named_scalars_across_ranks",
    "to_host_named_scalars",
]
