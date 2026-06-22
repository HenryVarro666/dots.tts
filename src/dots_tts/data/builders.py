"""数据集 / dataloader 的装配工厂 (assembly factory)。

本文件做什么 / What this file does
------------------------------------
把配置 (``DataConfig``) 翻译成可迭代的训练/验证数据流：
读取 source 列表 → 为每个 source 实例化 ``source_adapter`` + ``pipeline``
→ 用多源适配器 (multi-source adapter) 把它们组装成一个 ``BaseSourceAdapter``
→ 包进 ``StreamingSampleDataset`` (按 rank/world_size 做分布式分片 sharding)
→ 再套上 PyTorch ``DataLoader`` 与 ``BatchedDataStream`` 做在线 batching。

在数据流里的位置 / Where it sits in the pipeline
------------------------------------------------
config 层 (``DataConfig``)  →  **本文件 (builders)**  →  streaming 层
(``StreamingSampleDataset`` / ``BatchedDataStream``)  →  batchers / collator。
本文件只负责"装配"，不实现任何采样/分片/batching 逻辑本身——那些在
``streaming.py`` / ``source_adapters`` / ``pipelines`` 里。

关键函数清单 / Key functions
----------------------------
- ``_build_source_pipeline`` : pipeline 名 → pipeline 实例 (basic / interleave)。
- ``_build_source_specs``    : 把每个 source 配置组装成 ``SourceSpec``。
- ``_resolve_rank_info``     : 从 accelerator 取 (rank, world_size)。
- ``_local_num_tokens_per_epoch`` : 把全局 token 预算切到本 rank 上。
- ``_build_dataset``         : 选顺序/加权多源适配器，建 ``StreamingSampleDataset``。
- ``build_training_dataset`` / ``build_validation_dataset`` : 对外的数据集入口。
- ``build_training_dataloader`` / ``build_validation_dataloader`` : 对外的 loader 入口。

设计要点 / Design notes
-----------------------
- 训练用 **加权无限流** (``WeightedMultiSourceAdapter``)，会无限循环并按
  ``weight`` 抽 source；验证用 **顺序有限流** (``SequentialMultiSourceAdapter``)，
  按配置顺序把各 source 拼接一遍即停。这是 train/val 的核心区别。
- 分布式语义：``num_tokens_per_epoch`` 是 **全局** token 预算 (across all ranks)，
  到 dataloader 这一层才用 ``_local_num_tokens_per_epoch`` 切成 **本 rank** 的份额。
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from dots_tts.config.data import DataConfig
from dots_tts.data.pipelines.base import BaseSamplePipeline
from dots_tts.data.pipelines.tts_pipeline import BasicTtsPipeline, InterleaveTtsPipeline
from dots_tts.data.source_adapters.jsonl_manifest_adapter import (
    JsonlManifestSourceAdapter,
)
from dots_tts.data.source_adapters.multi_source_adapter import (
    SequentialMultiSourceAdapter,
    SourceSpec,
    WeightedMultiSourceAdapter,
)
from dots_tts.data.streaming import (
    BatchedDataStream,
    StreamingSampleDataset,
    identity_collate,
)

# source_adapter 注册表：配置里的 class_name 字符串 → 真正的适配器类。
# 当前只支持从 JSONL manifest 读样本；新增数据后端时在这里注册即可。
_SOURCE_ADAPTER_CLASSES = {
    "JsonlManifestSourceAdapter": JsonlManifestSourceAdapter,
}


def _build_source_pipeline(
    tokenizer, data_cfg, pipeline_name: str, *, profiler=None
) -> BaseSamplePipeline:
    """按名字实例化一条样本处理 pipeline。

    pipeline 负责把 source_adapter 吐出的原始样本 (文本+音频路径等)
    转成模型可吃的张量样本：文本 tokenize、音频读入并编码到连续 latent、
    x-vector 等条件的拼接。

    参数 / Args
    -----------
    tokenizer     : 文本 tokenizer，pipeline 内部 tokenize 文本时用。
    data_cfg      : ``DataConfig``，提供采样率、token 换算等超参。
    pipeline_name : ``"basic"`` 或 ``"interleave"``。
                    - basic      : 标准 TTS 样本 (一条文本 → 一段音频)。
                    - interleave : 文本/音频交错排布的样本组织方式。
    profiler      : 可选的耗时打点器，透传给 pipeline。

    返回 / Returns
    --------------
    ``BaseSamplePipeline`` 实例 (callable，接收原始样本迭代器，产出处理后样本)。

    未知 ``pipeline_name`` 直接抛 ``ValueError``，避免静默走错分支。
    """
    if pipeline_name == "basic":
        return BasicTtsPipeline(tokenizer, data_cfg, profiler=profiler)
    if pipeline_name == "interleave":
        return InterleaveTtsPipeline(tokenizer, data_cfg, profiler=profiler)
    raise ValueError(f"Unsupported data pipeline: {pipeline_name!r}")


def _build_source_specs(data_cfg, tokenizer, *, profiler=None) -> list[SourceSpec]:
    """把每个数据源配置组装成一个 ``SourceSpec`` 列表。

    一个 ``SourceSpec`` = (name, weight, adapter, pipeline)，是多源适配器
    (multi-source adapter) 调度的最小单元：adapter 负责"从哪读、读什么、
    断点续训状态"，pipeline 负责"把读到的原始样本变成模型样本"，weight 只在
    加权采样 (``WeightedMultiSourceAdapter``) 时生效。

    返回 / Returns
    --------------
    ``list[SourceSpec]``，顺序与 ``data_cfg.sources`` 一致 (顺序流会按此拼接)。
    """
    specs = []
    for source_cfg in data_cfg.sources:
        # 用注册表把配置里的类名字符串解析成具体适配器类，再用 params 字典实例化。
        adapter_cls = _SOURCE_ADAPTER_CLASSES[source_cfg.adapter.class_name]
        adapter = adapter_cls(**source_cfg.adapter.params)
        specs.append(
            SourceSpec(
                name=source_cfg.name,
                weight=float(source_cfg.weight),
                adapter=adapter,
                pipeline=_build_source_pipeline(
                    tokenizer, data_cfg, source_cfg.pipeline, profiler=profiler
                ),
            )
        )
    return specs


def _resolve_rank_info(accelerator=None) -> tuple[int, int]:
    """从 (HuggingFace Accelerate 的) accelerator 里取出分布式拓扑信息。

    返回 / Returns
    --------------
    ``(rank, world_size)``：当前进程序号 (process_index) 与总进程数。
    无 accelerator (单卡/无分布式) 时退化为 ``(0, 1)``，让单机也能跑通同一套
    分片逻辑。用 ``getattr`` 做容错，兼容不同 accelerator 实现/缺字段的情况。
    """
    rank = (
        int(getattr(accelerator, "process_index", 0)) if accelerator is not None else 0
    )
    world_size = (
        int(getattr(accelerator, "num_processes", 1)) if accelerator is not None else 1
    )
    return rank, world_size


def _local_num_tokens_per_epoch(
    global_num_tokens_per_epoch: int, *, rank: int, world_size: int
) -> int:
    """把"全局 token 预算"切成"本 rank 的 token 预算"。

    一个 epoch 的训练量用 token 数 (而非样本数) 来定，且这个数是 **全局** 的
    (所有 rank 加起来)。每个 rank 只负责其中一份，份额尽量平均：

        base = global // world_size              # 每 rank 至少分到的份额
        remainder = global % world_size          # 余下无法整除的部分
        本 rank 份额 = base + (1 if rank < remainder else 0)

    即把余数 ``remainder`` 个 token 依次多分给前 ``remainder`` 个 rank，
    保证各 rank 份额最多相差 1，且总和精确等于全局预算 (无丢失/无重复)。

    会先校验 ``world_size > 0`` 且 ``rank ∈ [0, world_size)``，越界直接报错。
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be in [0, {world_size}), but got rank={rank}."
        )

    base, remainder = divmod(int(global_num_tokens_per_epoch), int(world_size))
    # bool→int：前 remainder 个 rank 各多分 1 个 token，把余数均摊掉。
    return base + int(rank < remainder)


def _build_dataset(
    data_cfg: DataConfig,
    *,
    tokenizer,
    seed: int,
    accelerator=None,
    sequential: bool,
    profiler=None,
):
    """组装一个 ``StreamingSampleDataset`` (train/val 共用的核心装配逻辑)。

    参数 / Args
    -----------
    sequential : 选哪种多源适配器，这是 train 与 val 唯一的行为差异——
                 - ``False`` → ``WeightedMultiSourceAdapter``：无限流，按 weight
                   随机抽 source，每个子源各自循环 (用于训练)。
                 - ``True``  → ``SequentialMultiSourceAdapter``：有限流，按配置
                   顺序把各 source 跑一遍即停 (用于验证，需要可数、可复现)。
    seed       : 决定加权抽样与 (可选的) shuffle 的随机序列，保证可复现。
    accelerator: 提供 rank/world_size；交给 ``StreamingSampleDataset`` 做分片。

    返回 / Returns
    --------------
    ``StreamingSampleDataset``：一个 ``IterableDataset``，迭代时按
    (rank, world_size, worker) 做确定性分片 sharding，每条样本带上断点续训状态。
    """
    rank, world_size = _resolve_rank_info(accelerator)
    # 训练=加权无限流；验证=顺序有限流。仅此一处分叉决定 train/val 的数据语义。
    source_cls = SequentialMultiSourceAdapter if sequential else WeightedMultiSourceAdapter
    source = source_cls(
        sources=_build_source_specs(data_cfg, tokenizer, profiler=profiler)
    )
    return StreamingSampleDataset(
        source=source,
        rank=rank,
        world_size=world_size,
        seed=int(seed),
    )


def build_training_dataset(
    data_cfg: DataConfig,
    tokenizer,
    *,
    seed: int,
    accelerator=None,
    profiler=None,
):
    """对外入口：构建 **训练** 数据集 (加权无限流)。

    训练流是无限的，必须靠 ``num_tokens_per_epoch`` 这个 token 预算来界定
    "一个 epoch"，否则迭代永不停止——因此这里强制要求它非空，缺失即报错。
    其余装配工作委托给 ``_build_dataset`` (``sequential=False``)。
    """
    if data_cfg.num_tokens_per_epoch is None:
        raise ValueError("Training data requires num_tokens_per_epoch.")
    return _build_dataset(
        data_cfg,
        tokenizer=tokenizer,
        seed=seed,
        accelerator=accelerator,
        sequential=False,
        profiler=profiler,
    )


def build_validation_dataset(
    data_cfg: DataConfig,
    tokenizer,
    *,
    seed: int,
    accelerator=None,
    profiler=None,
):
    """对外入口：构建 **验证** 数据集 (顺序有限流)。

    验证流按配置顺序把各 source 拼接跑一遍即自然结束，因此 **不需要**
    ``num_tokens_per_epoch`` (不强制校验)。装配委托给 ``_build_dataset``
    (``sequential=True``)。有限+顺序保证验证集每轮内容固定、指标可比。
    """
    return _build_dataset(
        data_cfg,
        tokenizer=tokenizer,
        seed=seed,
        accelerator=accelerator,
        sequential=True,
        profiler=profiler,
    )


def _build_sample_loader(dataset, data_cfg: DataConfig) -> DataLoader:
    """把 ``StreamingSampleDataset`` 包成一个产出 **单条样本** 的 ``DataLoader``。

    关键设计 / Key choices
    -----------------------
    - ``batch_size=None`` + ``identity_collate``：**关闭** DataLoader 自带的
      batching，让它只做"多进程预取单条样本"。真正的成批 (按 token 数动态
      bucketing) 交给下游的 ``BatchedDataStream`` / ``OnlineBatcher`` 完成——
      因为变长音频/文本无法用固定 batch_size 高效打包。
    - ``persistent_workers`` 仅在多 worker 时开，避免每个 epoch 重启 worker 进程
      (重启会丢掉 ``IterableDataset`` 的流式状态、拖慢启动)。
    - ``prefetch_factor`` 只有在 ``num_workers > 0`` 时才合法，故条件性写入；
      单进程 (num_workers==0) 传它会被 PyTorch 拒绝。
    """
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "collate_fn": identity_collate,
        "num_workers": data_cfg.num_workers,
        "pin_memory": data_cfg.pin_memory,
        "persistent_workers": data_cfg.num_workers > 0,
    }
    if data_cfg.num_workers > 0:
        # prefetch_factor 是"每 worker 预取多少条"，单进程模式下传它会报错。
        loader_kwargs["prefetch_factor"] = int(data_cfg.prefetch_factor)
    sample_loader = DataLoader(**loader_kwargs)
    return sample_loader


def build_training_dataloader(
    dataset, data_cfg: DataConfig, tokenizer, *, profiler=None
):
    """对外入口：基于训练 ``dataset`` 构建 **训练** 数据流 (``BatchedDataStream``)。

    流程 / Flow
    -----------
    1. 把全局 token 预算切成本 rank 的份额 (``_local_num_tokens_per_epoch``)，
       下游 ``BatchedDataStream`` 用它来判定本 rank 的 epoch 何时结束。
    2. 建单样本 loader (多进程预取)。
    3. 建 ``BatchedDataStream`` (在线动态 batching) 并 attach 上 loader。

    返回 / Returns
    --------------
    ``BatchedDataStream``：可迭代，每次产出一个 collate 好的 batch (dict of tensors)。
    """
    # 全局 → 本地：rank 间已天然并行，每个 rank 只需消费自己那份 token 预算。
    local_num_tokens_per_epoch = _local_num_tokens_per_epoch(
        int(data_cfg.num_tokens_per_epoch),
        rank=int(dataset.rank),
        world_size=int(dataset.world_size),
    )
    sample_loader = _build_sample_loader(dataset, data_cfg)
    batched_stream = BatchedDataStream(
        sample_dataset=dataset,
        data_cfg=data_cfg,
        tokenizer=tokenizer,
        num_tokens_per_epoch=local_num_tokens_per_epoch,
        profiler=profiler,
    )
    batched_stream.attach_loader(sample_loader)
    return batched_stream


def build_validation_dataloader(
    dataset, data_cfg: DataConfig, tokenizer, *, profiler=None
):
    """对外入口：基于验证 ``dataset`` 构建 **验证** 数据流 (``BatchedDataStream``)。

    与训练版的唯一区别：``num_tokens_per_epoch=None``。验证用的是顺序有限流，
    数据 **跑完即停**，不靠 token 预算来截断，所以这里显式传 ``None`` (不设上限)。
    """
    sample_loader = _build_sample_loader(dataset, data_cfg)
    batched_stream = BatchedDataStream(
        sample_dataset=dataset,
        data_cfg=data_cfg,
        tokenizer=tokenizer,
        num_tokens_per_epoch=None,  # 无 token 预算：数据源耗尽即结束 epoch。
        profiler=profiler,
    )
    batched_stream.attach_loader(sample_loader)
    return batched_stream


__all__ = [
    "build_training_dataloader",
    "build_training_dataset",
    "build_validation_dataloader",
    "build_validation_dataset",
]
