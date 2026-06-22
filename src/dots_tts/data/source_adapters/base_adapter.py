"""数据源 adapter 抽象基类 / Abstract base classes for data-source adapters.

本文件定义 dots.tts 训练数据流最上游的「数据源」抽象层。一个 source adapter 的职责是：
把某种底层语料（JSONL manifest、多源混合……）转成一个 **可恢复（resumable）、可分片
（shardable）** 的样本流，供下游 pipeline（文本/音频预处理）和 DataLoader 消费。

This module sits at the very head of the training data pipeline. A source adapter turns some
underlying corpus (a JSONL manifest, a mixture of sources, …) into a **resumable** and
**shardable** stream of raw samples that downstream pipelines and the DataLoader consume.

关键设计 / Key design ideas:
- **可恢复（resumable）**: adapter 不直接持有「读到哪了」的可变游标，而是把进度编码进一个
  纯 dict 形式的 ``state``（如 ``{"cycle": .., "cursor": ..}``）。每条产出的样本都会带上下一个
  ``state``（约定写在 ``sample["_adapter_state"]``），训练崩溃后可从 checkpoint 里的 state 精确续跑，
  不重复也不漏样本。This makes mid-epoch checkpoint/resume exact.
- **可分片（shardable）**: 多 GPU（rank/world_size）× 多 DataLoader worker（worker_id/num_workers）
  的笛卡尔积构成全局 worker 网格；每个全局 worker 只取属于自己的样本子集，做到无重叠、无遗漏的
  确定性切分（见 ``ShardableSourceAdapter``）。
- **可循环（cycling）**: 有限源读完会 ``StopIteration``；无限/加权采样器靠 ``advance_cycle`` 让
  子源「翻篇」重新开始，并用 ``is_cycle_start_state`` 识别「子源一上来就空」这种死循环风险。

关键类型 / Key types:
- ``SourceContext``  —— 一次迭代的不可变运行上下文（epoch / rank / worker / seed）。
- ``BaseSourceAdapter`` —— 所有 adapter 的抽象接口（state 生命周期 + 样本迭代）。
- ``ShardableSourceAdapter`` —— 提供确定性 rank/worker 分片的 mixin 工具。
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeVar


@dataclass(frozen=True)
class SourceContext:
    """一次 adapter 迭代的不可变运行上下文 / Immutable per-iterator execution context.

    描述「谁、在什么 epoch、用什么随机种子」来跑这条数据流。冻结（``frozen=True``）保证它在迭代
    过程中不可变，从而让分片/洗牌完全由这些字段确定（determinism），便于精确续跑与复现实验。

    Describes *who* iterates, in *which* epoch, with *which* seed. Being frozen guarantees the
    sharding/shuffle decisions are a pure function of these fields — i.e. fully deterministic.

    字段 / Fields:
        epoch:        当前训练轮次；参与 shuffle 种子，使不同 epoch 顺序不同。
        rank:         分布式训练里本进程的全局编号（0..world_size-1）。
        world_size:   分布式总进程数（通常 = GPU 数）。
        worker_id:    DataLoader 内本 worker 的编号（0..num_workers-1）。
        num_workers:  单个 rank 的 DataLoader worker 数。
        seed:         全局随机种子基准；各 worker 在其上叠加 epoch/offset 得到自己的种子。
    """

    epoch: int
    rank: int
    world_size: int
    worker_id: int
    num_workers: int
    seed: int

    @property
    def global_worker_count(self) -> int:
        # 全局 worker 总数 = world_size × num_workers，即所有 GPU × 所有 DataLoader worker 的网格大小。
        # 用 max(1, ..) 兜底：单进程 + num_workers=0 时仍返回 1，避免后续取模出现除零。
        return max(1, self.world_size * self.num_workers)

    @property
    def global_worker_id(self) -> int:
        # 把 (rank, worker_id) 这对二维坐标线性化成全局唯一的一维 id（行主序展平）。
        # 配合 global_worker_count 做模运算，即可把样本无重叠地分配到每个全局 worker。
        return self.rank * self.num_workers + self.worker_id


class BaseSourceAdapter(ABC):
    """带状态的流式数据源接口 / State-aware streaming source interface.

    所有具体 adapter（``JsonlManifestSourceAdapter``、``Sequential/WeightedMultiSourceAdapter``）
    都要实现这个接口。核心契约是把「读取进度」外置成一个可序列化的 ``state`` dict，从而支持
    精确的 checkpoint / resume。

    Every concrete adapter implements this interface. The core contract: externalise the read
    *progress* into a serialisable ``state`` dict, enabling exact checkpoint/resume.

    state 生命周期 / state lifecycle:
        - ``initial_state()`` 给出全新（从头开始）的 state；
        - ``iter_samples()`` 从给定 state 续读，并把「下一步的 state」写进每条样本的
          ``_adapter_state`` 字段（下游据此更新 checkpoint）；
        - ``normalize_state()`` / ``clone_state()`` 负责补全缺省字段并做深拷贝，避免别名共享；
        - ``advance_cycle()`` 让有限源在被无限采样器复用时「翻篇」重新开始。
    """

    @abstractmethod
    def initial_state(self) -> dict[str, Any]:
        """返回一个全新 worker/epoch 的默认迭代状态 / Default state for a fresh worker/epoch.

        Returns: 形如 ``{"cycle": 0, "cursor": 0}`` 的纯 dict（具体字段由子类决定），代表「从头读」。
        """

    @abstractmethod
    def iter_samples(
        self,
        context: SourceContext,
        *,
        state: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        """从 ``state`` 处续读样本，并给每条样本附上「下一步」状态 / Yield samples, each carrying next state.

        Args:
            context: 本次迭代的运行上下文（用于分片与洗牌）。
            state:   续读起点；``None`` 表示从 ``initial_state()`` 开始。

        约定：每条 yield 出的样本必须带 ``sample["_adapter_state"]``（= 读完这条之后应保存的 state）。
        Contract: every yielded sample must carry ``sample["_adapter_state"]`` (the state to persist
        *after* consuming it). Downstream code relies on this for crash-safe resume.
        """

    @abstractmethod
    def is_cycle_start_state(self, state: dict[str, Any] | None) -> bool:
        """判断 ``state`` 是否正指向某一轮的起点 / Whether ``state`` points at a cycle start.

        被无限采样器用来检测「子源一开始就空」的退化情况：若刚 advance_cycle 到起点却立刻
        StopIteration，说明该 worker 分到的样本为空，应当报错而非无限空转。
        """

    def normalize_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        # 把外部传入的（可能不完整的）state 与 initial_state() 合并补全：缺的键用默认值，存在的键覆盖。
        # deepcopy 传入值，避免把调用方的 dict 直接塞进返回结果造成别名共享/意外修改。
        merged = self.initial_state()
        if state:
            merged.update(deepcopy(state))
        return merged

    def clone_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        # 先归一化补全，再整体深拷贝，得到一份与外界完全隔离、可安全持久化的 state 快照。
        return deepcopy(self.normalize_state(state))

    def advance_cycle(self, state: dict[str, Any] | None) -> dict[str, Any]:
        # 默认实现：拒绝循环。有限源（如顺序拼接）天然不应被反复 cycling；
        # 只有真正可循环的源（如 JSONL manifest）才覆写此方法返回 cursor 归零、cycle+1 的新 state。
        raise RuntimeError(
            f"{self.__class__.__name__} does not support repeated cycling."
        )


_T = TypeVar("_T")


class ShardableSourceAdapter(BaseSourceAdapter):
    """确定性 rank/worker 分片的工具 mixin / Helper mixin for deterministic sharding.

    提供两个静态方法，把一个可索引序列按全局 worker 网格无重叠地切给当前 worker。
    「确定性」是关键：任何 worker 在相同 ``context`` 下都会算出相同的归属判断，因此分片结果
    不依赖运行时通信，天然无重叠、无遗漏，且崩溃续跑时可复现。

    Provides static helpers that partition an indexable sequence across the global worker grid
    with no overlap. Determinism is the point: the assignment is a pure function of ``context``,
    so it needs no cross-worker communication and reproduces exactly on resume.
    """

    @staticmethod
    def is_assigned_index(index: int, context: SourceContext) -> bool:
        # 取模分片（round-robin / strided）：第 index 个元素归属于 (index mod 全局worker数) 号全局 worker。
        # 当且仅当这个余数等于当前 worker 的 global_worker_id 时，本 worker 才处理该元素。
        return index % context.global_worker_count == context.global_worker_id

    @staticmethod
    def shard_items(
        items: Sequence[_T],
        context: SourceContext,
        *,
        shuffle: bool = False,
        seed_offset: int = 0,
    ) -> list[_T]:
        # 物化成 list（不就地改入参），可选先做一次「所有 worker 一致」的洗牌，再按 index 取模分片。
        assigned = list(items)
        if shuffle:
            # 种子 = seed + epoch + seed_offset：不含 rank/worker，确保各 worker 洗出完全相同的顺序，
            # 这样后续按 index 取模才能保证整体无重叠、无遗漏；epoch 入种子使逐 epoch 顺序不同。
            random.Random(context.seed + context.epoch + seed_offset).shuffle(assigned)
        return [
            item
            for index, item in enumerate(assigned)
            if ShardableSourceAdapter.is_assigned_index(index, context)
        ]
