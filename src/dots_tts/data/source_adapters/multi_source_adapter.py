"""多数据源组合 adapter / Multi-source composition adapters.

本文件做什么 (What this file does)
------------------------------------
在 dots.tts 的训练数据流水线里，每个"数据源"(source) 由一个
:class:`BaseSourceAdapter` (负责"读出原始样本 + 维护可恢复的迭代状态")
和一条 :class:`BaseSamplePipeline` (负责对原始样本做变换/过滤) 组成。
本文件提供两种把多个数据源**组合**成单一 adapter 的方式，让上层只看到一个统一的样本流:

- :class:`SequentialMultiSourceAdapter` —— **顺序拼接**: 把各源按配置顺序首尾相接，
  逐源耗尽后再进入下一源；是一个**有限**(finite) 序列(总会在最后一个源耗尽后结束)。
- :class:`WeightedMultiSourceAdapter` —— **加权采样**: 按 ``weight`` 比例在各源之间
  随机挑选下一个样本，并让每个源**独立循环**(一个源跑完一轮就 ``advance_cycle`` 重开)，
  因此是一个**无限**(infinite) 流，永不自然结束。

在训练数据流里的位置 (Position in the data flow)
------------------------------------------------
config → 每个 source 实例化出 (adapter, pipeline) → 本文件把它们组合成一个
顶层 adapter → DataLoader 通过 :meth:`iter_samples` 拉取样本喂给模型训练。
关键在于**可断点续训** (resumable / checkpointable): 每个 yield 出去的样本都
通过 ``_adapter_state`` 字段携带"产出该样本之后的完整迭代状态"，写进 checkpoint 后
即可在任意位置精确恢复 (deterministic resume)，无需从头重跑 epoch。

关键类/函数清单 (Key classes / functions)
-----------------------------------------
- :class:`SourceSpec` —— 单个数据源的配置打包 (name / weight / adapter / pipeline)。
- :func:`_mix_uint64` / :func:`_stable_seed` —— 与平台/进程无关的确定性哈希，
  给加权采样提供**可复现**的伪随机抽取(取代有状态的 RNG，从而便于断点续训)。
- :class:`SequentialMultiSourceAdapter` —— 顺序拼接(有限)。
- :class:`WeightedMultiSourceAdapter` —— 加权采样(无限、各源独立循环)。
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from dots_tts.data.pipelines.base import BaseSamplePipeline
from dots_tts.data.source_adapters.base_adapter import (
    BaseSourceAdapter,
    SourceContext,
)


@dataclass(frozen=True)
class SourceSpec:
    """单个数据源的不可变配置 / Immutable spec of one data source.

    把"一个源"所需的四件东西打包在一起，供下面两个组合 adapter 统一处理。
    Bundles everything the composing adapters need to drive one source.

    字段 (Fields):
        name:     源的唯一名字；既做状态字典里的 key，也会写进每个样本的
                  ``source_name``，便于训练时按源统计/调试。
        weight:   仅 :class:`WeightedMultiSourceAdapter` 使用的采样权重(必须 > 0)；
                  顺序拼接模式忽略它。
        adapter:  真正读数据并维护可恢复迭代状态的底层 source adapter。
        pipeline: 套在 adapter 原始样本流之上的变换/过滤管线 (callable: iter→iter)。
    """

    name: str
    weight: float
    adapter: BaseSourceAdapter
    pipeline: BaseSamplePipeline


# 64-bit 掩码: 用按位与模拟 uint64 回卷, 因为 Python int 是任意精度、不会自动溢出。
# 64-bit mask to emulate uint64 wrap-around (Python ints are arbitrary precision).
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF


def _mix_uint64(value: int) -> int:
    """SplitMix64 的 finalizer 雪崩混合 / SplitMix64 finalizer (avalanche mix).

    把一个 64-bit 整数充分"打散": 输入哪怕只差 1 bit, 输出也近似均匀随机变化。
    这是确定性哈希的核心一步, 配合 :func:`_stable_seed` 用于加权采样的可复现抽样。
    那两个魔数是 SplitMix64 论文给出的固定乘子, 无需理解其取值——只需知道它们
    经过验证能产生良好的 bit 扩散 (good avalanche)。
    """
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    value &= _UINT64_MASK  # 每次乘法后立刻掩码回 64-bit, 防止 Python int 无限增长
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB
    value &= _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _stable_seed(*parts: int) -> int:
    """把若干整数确定性地折叠成一个 64-bit 种子 / Fold ints into a stable 64-bit seed.

    为什么不用 ``random``/``hash``: 训练要求**跨进程、跨平台、可断点续训**地复现
    完全相同的采样序列。Python 内置 ``hash`` 受 PYTHONHASHSEED 影响、``random`` 是有状态的,
    都不满足。这里用无状态的纯函数: 同样的 ``parts`` 永远得到同样的输出, 与机器无关。

    实现: 初值取黄金比例常数 0x9E3779B9..., 把每个 part 依次累加进去再做一轮
    SplitMix64 混合, 让"输入顺序 + 每个值"都共同决定最终种子。
    """
    value = 0x9E3779B97F4A7C15
    for part in parts:
        # 累加黄金比例增量再混合: 这一步保证各 part 不可交换 (顺序敏感) 且充分扩散
        value = (value + int(part) + 0x9E3779B97F4A7C15) & _UINT64_MASK
        value = _mix_uint64(value)
    return value


class SequentialMultiSourceAdapter(BaseSourceAdapter):
    """顺序拼接多源 (有限) / Finite adapter that concatenates sources in order.

    把各源按配置顺序首尾相接: 先把第 0 个源完整跑完, 再跑第 1 个, 直到最后一个源耗尽,
    整个迭代随之结束 —— 因此它是**有限**的。常用于"先小数据集预热再切大数据集"之类
    的固定课程 (curriculum) 安排, 或把若干已切分好的子集当作一个逻辑 epoch 串起来。

    状态 (state) 结构: ``{"source_index": <当前正在跑第几个源>, "sources": {名字: 子状态}}``。
    ``source_index`` 让断点续训能跳过已耗尽的前面的源; 每个子源的进度则各存在
    ``sources`` 里, 由各自的 adapter 维护。
    """

    def __init__(self, *, sources: list[SourceSpec]):
        if not sources:
            raise ValueError(
                "SequentialMultiSourceAdapter requires at least one source."
            )
        self.sources = list(sources)  # 拷一份, 避免外部后续改动影响内部顺序

    def initial_state(self) -> dict:
        """全新迭代的初始状态 / Fresh iterator state: 从第 0 个源、各子源各自的初态开始。"""
        return {
            "source_index": 0,
            "sources": {
                source.name: source.adapter.initial_state() for source in self.sources
            },
        }

    def is_cycle_start_state(self, state: dict | None) -> bool:
        """是否处于一轮的起点 / Whether ``state`` sits exactly at a cycle start.

        仅当"还没开始跑任何源"(source_index==0) 且"每个子源也都在各自的起点"时才为真。
        上层据此判断该 worker 这一轮是否一个样本都没产出 (用于报错而非静默卡死)。
        """
        normalized = self.normalize_state(state)
        if int(normalized["source_index"]) != 0:
            return False
        return all(
            source.adapter.is_cycle_start_state(normalized["sources"][source.name])
            for source in self.sources
        )

    def normalize_state(self, state: dict | None) -> dict:
        """把(可能残缺/来自旧 checkpoint 的)外部 state 补全成规范结构 / Normalize state.

        基类先用 ``initial_state`` 兜底再 update 外部值; 这里进一步把每个子源的状态
        交给对应 adapter 的 ``clone_state`` 规范化, 并强制 ``source_index`` 为 int。
        这样即便外部传入只含部分源、或字段类型不对的 state, 也能安全恢复。
        """
        normalized = super().normalize_state(state)
        source_states = normalized.get("sources") or {}
        normalized["sources"] = {
            source.name: source.adapter.clone_state(source_states.get(source.name))
            for source in self.sources
        }
        normalized["source_index"] = int(normalized.get("source_index", 0))
        return normalized

    def clone_state(self, state: dict | None) -> dict:
        """规范化并深拷贝一份独立 state / Normalize + deep-copy an isolated snapshot.

        深拷贝是关键: yield 出去附在样本上的状态必须与内部正在演进的 ``live_state``
        完全脱钩, 否则后续迭代会就地修改已经发出的快照, 破坏断点续训的正确性。
        """
        return deepcopy(self.normalize_state(state))

    def iter_samples(
        self,
        context: SourceContext,
        *,
        state: dict | None = None,
    ) -> Iterable[dict]:
        """按顺序逐源产出样本, 每个样本带可续训状态 / Yield samples source-by-source.

        Args:
            context: 本次迭代的执行环境 (epoch / rank / worker / seed), 用于子源分片。
            state:   续训时传入的上次快照; ``None`` 表示从头开始。

        Yields:
            dict: 经各源 pipeline 处理后的样本, 额外注入两个字段——
                  ``source_name`` (来源名) 与 ``_adapter_state`` (产出本样本后的完整顶层状态)。

        恢复逻辑: 从 ``source_index`` 起跳过已耗尽的前面的源; 当前源用其子状态从断点续起,
        后面的源则各自从头开始 (它们的子状态仍是初态)。
        """
        live_state = self.normalize_state(state)
        start_index = int(live_state["source_index"])  # 断点续训: 跳过已耗尽的前序源
        for index in range(start_index, len(self.sources)):
            source = self.sources[index]
            child_state = live_state["sources"][source.name]
            # 先让底层 adapter 产原始样本流, 再套上该源自己的 pipeline 做变换/过滤
            raw_iter = source.adapter.iter_samples(context, state=child_state)
            for sample in source.pipeline(raw_iter):
                item = dict(sample)  # 拷一份, 避免就地修改 pipeline 内部对象
                # 子 adapter 必须把"产出此样本后的子状态"放进 _adapter_state; 取出并从样本里剥掉
                next_child_state = item.pop("_adapter_state", None)
                if next_child_state is None:
                    raise RuntimeError(
                        f"{source.adapter.__class__.__name__} must attach '_adapter_state' to samples."
                    )
                live_state["source_index"] = index  # 记录"当前仍停在第 index 个源"
                live_state["sources"][source.name] = source.adapter.clone_state(
                    next_child_state
                )
                item["source_name"] = source.name
                # 把整个顶层状态深拷贝后附给样本, 作为可写入 checkpoint 的续训锚点
                item["_adapter_state"] = self.clone_state(live_state)
                yield item
            # 本源耗尽: 把指针推进到下一个源, 这样续训会从下一个源的起点开始
            live_state["source_index"] = index + 1


class WeightedMultiSourceAdapter(BaseSourceAdapter):
    """加权采样多源 (无限) / Infinite weighted sampler cycling each source independently.

    每抽一个样本: 先按各源 ``weight`` 的比例**确定性地**(见 :func:`_stable_seed`)选中一个源,
    再从该源取下一个样本。每个源各自维护一个迭代器并**独立循环**——某个源跑完一轮
    (``StopIteration``)就 ``advance_cycle`` 重新开一轮, 因此整体永不结束 (infinite)。
    这正是大规模混合语料训练想要的: 小数据集会被反复循环, 大数据集慢慢推进,
    而每个样本来自各源的长期期望比例恰好等于 ``weight`` 占比。

    状态 (state) 结构: ``{"draw_count": <已抽样本计数>, "sources": {名字: 子状态}}``。
    ``draw_count`` 同时是采样序号 (喂给 ``_stable_seed`` 决定第 N 次抽哪个源) 与续训锚点;
    各源在循环中的进度存于 ``sources``。
    """

    def __init__(self, *, sources: list[SourceSpec]):
        if not sources:
            raise ValueError("WeightedMultiSourceAdapter requires at least one source.")
        # 权重必须为正: 0 或负权重在累积权重上无法表示"可被选中", 直接报错列出违规源
        invalid = [source.name for source in sources if float(source.weight) <= 0.0]
        if invalid:
            raise ValueError(f"Source weights must be positive: {invalid}")
        self.sources = list(sources)
        # 预计算**累积权重** (cumulative weights): 把权重折成区间 [0, total) 上的分段,
        # 抽样时只需把一个 [0, total) 的随机值落到哪一段, 即选中对应源 (O(N) 线性扫描)。
        self._cumulative_weights = []
        total = 0.0
        for source in self.sources:
            total += float(source.weight)
            self._cumulative_weights.append(total)
        self._total_weight = total

    def initial_state(self) -> dict:
        """全新迭代的初始状态 / Fresh state: 抽样计数为 0, 各子源各自从初态开始。"""
        return {
            "draw_count": 0,
            "sources": {
                source.name: source.adapter.initial_state() for source in self.sources
            },
        }

    def is_cycle_start_state(self, state: dict | None) -> bool:
        """是否处于整体起点 / Whether ``state`` is the very beginning.

        仅当一个样本都还没抽 (draw_count==0) 且每个子源也都在起点时为真。
        """
        normalized = self.normalize_state(state)
        if int(normalized["draw_count"]) != 0:
            return False
        return all(
            source.adapter.is_cycle_start_state(normalized["sources"][source.name])
            for source in self.sources
        )

    def normalize_state(self, state: dict | None) -> dict:
        """补全/规范化 state, 并强制 ``draw_count`` 为 int / Normalize state.

        语义同 :meth:`SequentialMultiSourceAdapter.normalize_state`, 只是顶层计数字段
        从 ``source_index`` 换成 ``draw_count`` (抽样序号)。
        """
        normalized = super().normalize_state(state)
        source_states = normalized.get("sources") or {}
        normalized["sources"] = {
            source.name: source.adapter.clone_state(source_states.get(source.name))
            for source in self.sources
        }
        normalized["draw_count"] = int(normalized.get("draw_count", 0))
        return normalized

    def clone_state(self, state: dict | None) -> dict:
        """规范化并深拷贝, 给 yield 的样本一份与内部状态脱钩的快照 / Normalize + deep-copy."""
        return deepcopy(self.normalize_state(state))

    def _source_draw_value(self, context: SourceContext, draw_count: int) -> float:
        """为第 ``draw_count`` 次抽样生成 [0, total_weight) 上的确定性随机值。

        把"全局上下文 + 抽样序号"喂给无状态哈希 :func:`_stable_seed`, 得到一个 64-bit 整数,
        归一化到 [0,1) 后再乘以总权重, 落到累积权重区间里即决定选哪个源。
        关键: 同样的 (seed, epoch, rank, worker_id, draw_count) 永远得到同一个值,
        故采样序列**与机器无关、可断点续训**地复现; 不同 rank/worker 因 id 不同而得到不同序列,
        从而各自采到不同样本 (隐式分片去重)。
        """
        raw = _stable_seed(
            context.seed,
            context.epoch,
            context.rank,
            context.worker_id,
            draw_count,
        )
        # raw 是 uint64; 除以 2**64 映射到 [0,1), 再线性放大到 [0, total_weight)
        return (raw / float(1 << 64)) * self._total_weight

    def _pick_source(self, context: SourceContext, draw_count: int) -> SourceSpec:
        """按累积权重把抽样值定位到具体源 / Pick a source by cumulative-weight bucket.

        线性扫描找到第一个"上界 > draw_value"的源即命中。末尾的 ``return self.sources[-1]``
        是数值兜底: 当 draw_value 因浮点取整恰好等于 total_weight 时, 落回最后一个源。
        """
        draw_value = self._source_draw_value(context, draw_count)
        for source, upper in zip(self.sources, self._cumulative_weights, strict=True):
            if draw_value < upper:
                return source
        return self.sources[-1]  # 浮点边界兜底 (draw_value == total_weight 的极端情形)

    def iter_samples(
        self,
        context: SourceContext,
        *,
        state: dict | None = None,
    ) -> Iterable[dict]:
        """无限按权重抽样产出, 每个样本带可续训状态 / Infinitely yield weighted samples.

        Args:
            context: 执行环境; 与 ``draw_count`` 一起决定每次抽哪个源 (见 :meth:`_source_draw_value`)。
            state:   续训快照; ``None`` 表示从头开始。

        Yields:
            dict: 经源 pipeline 处理的样本, 注入 ``source_name`` 与 ``_adapter_state``。

        结构是双层循环:
        - 外层每轮决定第 ``draw_count`` 次抽样选中的源;
        - 内层从该源取样, 若该源迭代器耗尽则 ``advance_cycle`` 重开一轮并重试,
          直到拿到一个样本才 ``break`` 回外层、把 ``draw_count`` 自增。
        """
        live_state = self.normalize_state(state)
        # 进程内缓存各源的活动迭代器: 跨多次抽样保持各源的"读取游标"不丢, 避免每抽一次都重建。
        # 注意它不进 checkpoint —— 续训靠 sources 子状态重建迭代器, 这里只是运行期缓存。
        iterators: dict[str, object] = {}

        while True:
            draw_count = int(live_state["draw_count"])
            source = self._pick_source(context, draw_count)  # 本次抽样命中的源

            while True:
                child_state = live_state["sources"][source.name]
                child_iter = iterators.get(source.name)
                if child_iter is None:
                    # 首次用到该源 (或它刚 advance_cycle 后被清掉): 用当前子状态重建迭代器
                    raw_iter = source.adapter.iter_samples(context, state=child_state)
                    child_iter = iter(source.pipeline(raw_iter))
                    iterators[source.name] = child_iter

                try:
                    sample = dict(next(child_iter))  # 拷一份, 不就地改 pipeline 产物
                except StopIteration:
                    # 该源本轮耗尽。若它还停在"起点"却没产出任何样本, 说明这个 worker
                    # 分到的该源数据为空 —— 无限循环会变成死循环, 故直接报错而非静默卡住。
                    if source.adapter.is_cycle_start_state(child_state):
                        raise RuntimeError(
                            "Weighted source yielded no samples for this worker. "
                            f"source={source.name!r}, worker={context.global_worker_id}, "
                            f"epoch={context.epoch}"
                        )
                    # 否则: 丢弃旧迭代器, 让该源进入下一轮循环 (advance_cycle 通常会改 epoch/偏移
                    # 以换不同 shuffle), 然后 continue 用新子状态重建迭代器再取一次。
                    iterators.pop(source.name, None)
                    live_state["sources"][source.name] = source.adapter.advance_cycle(
                        child_state
                    )
                    continue

                # 取到样本: 剥出子状态 (子 adapter 必须挂载), 更新该源进度
                next_child_state = sample.pop("_adapter_state", None)
                if next_child_state is None:
                    raise RuntimeError(
                        f"{source.adapter.__class__.__name__} must attach '_adapter_state' to samples."
                    )
                live_state["sources"][source.name] = source.adapter.clone_state(
                    next_child_state
                )
                # 仅在**成功产出一个样本后**才递增 draw_count: 保证 draw_count 与"已 yield 数"
                # 严格对齐, 续训时第 N 次抽样的随机种子才能精确复现 (耗尽重试不计入)。
                live_state["draw_count"] = draw_count + 1
                sample["source_name"] = source.name
                sample["_adapter_state"] = self.clone_state(live_state)  # 深拷贝快照, 与内部脱钩
                yield sample
                break  # 跳回外层, 进入下一次加权抽样
