"""在线动态批构造器 (online dynamic batcher) —— 训练数据流的"成批"环节。

本文件做什么 / What this file does
-----------------------------------
把一条**逐样本**的数据流 (sample iterator) 实时聚合成一批批 (batch) 喂给训练。
不同于静态 batch_size，这里用的是 **token 预算 (token budget) 动态批**策略：
不固定每批样本数，而是约束 "padding 后的 token 总量"，从而把显存利用率压满、
同时让同一批内样本长度尽量接近 (length bucketing) 以减少 padding 浪费。

为什么 TTS 需要这个 / Why TTS cares
------------------------------------
dots.tts 是连续潜在 (continuous-latent) 自回归 TTS：每个样本既有文本侧 token
(input_ids)，又有声学侧的 audio latent token，二者长度差异都很大。若按固定样本数
成批，长短样本混在一起会产生大量 padding；按 token 预算 + 长度桶化 (length
bucketing) 成批可显著减少浪费。注意预算用的是 **padded** 口径：一批的代价 ≈
该批最长样本长度 × 批内样本数 (即 padding 到等长后的矩形面积)，而非各样本长度之和。

在数据流中的位置 / Position in the pipeline
--------------------------------------------
上游: dataset / sample stream 产出 dict 样本 (含 num_audio_tokens、input_ids_length
等字段) →  **本文件 OnlineBatcher.build_decisions** 产出一连串 BatchDecision →
下游: collate / DataLoader 按 BatchDecision.batch_samples 真正拼成张量批。
本批构造器只决定"哪些样本进同一批/哪些样本被丢弃"，不接触张量本身。

关键类 / 函数清单 / Key classes & functions
--------------------------------------------
- BatchDecision     : 单次决策的结果 (本批选中的样本 + 被丢弃的样本)。
- _PoolSample       : 候选池里的一个样本及其缓存的长度/到达时刻元数据。
- OnlineBatcher     : 主体；维护有界候选池 (sample pool)、排序、选锚点、贪心打包。
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from dots_tts.utils.profiling import ensure_data_profiler


@dataclass(slots=True)
class BatchDecision:
    """一次批构造决策的产物 / Result of one batching step.

    属性 / Attributes:
        dropped_samples: 本步被永久丢弃的样本 (单条就超过任何预算、无法成批)。
            Samples discarded this step (too large to ever fit any batch).
        batch_samples : 本步组成一个训练批的样本列表 (下游 collate 用)。
            Samples chosen to form one training batch.

    设计说明 / Note: 用 slots=True 的 dataclass 是为了降低这种高频小对象的内存/
    创建开销 (训练时每个 batch 都会产生一个)。dropped 与 batch 分开返回，便于上层
    分别统计"被丢弃样本数"这一数据质量指标。
    """

    dropped_samples: list[dict]
    batch_samples: list[dict]


@dataclass(slots=True)
class _PoolSample:
    """候选池中的一个样本及其缓存元数据 / A pooled sample with cached metadata.

    把长度等字段在入池时算好缓存下来 (而不是每次排序/打包都从 dict 里取并 int()
    转换)，因为同一个样本在出池前会被反复读取 (排序键、预算判断)。下划线前缀表示
    这是模块内部实现细节，不对外暴露。

    属性 / Attributes:
        sample          : 原始样本 dict (透传给下游，本类不解析其张量内容)。
        num_audio_tokens: 声学侧 latent token 数 (用于 audio 预算)。
        num_text_tokens : 文本侧 token 数 = input_ids 长度 (用于 text 预算)。
        arrival_step    : 该样本进入候选池时的 decision_step，用于"等待时长"
            老化判断 (anti-starvation)。Step at which the sample entered the pool.
    """

    sample: dict
    num_audio_tokens: int
    num_text_tokens: int
    arrival_step: int


class OnlineBatcher:
    """在线 token-预算动态批构造器 / Online token-budget dynamic batcher.

    核心思想 / Core idea
    --------------------
    维护一个有界候选池 (sample pool)，每一步:
      1. 把池排序，让长度相近的样本聚到一起 (length bucketing，减少 padding)。
      2. 选一个"锚点 (anchor)"样本作为本批的最长样本/起点。
      3. 从池里贪心 (greedy) 地往这一批里塞其它样本，只要塞进去后 **padded** 预算
         (最长长度 × 批内样本数) 仍不超 audio / text / batch_size 上限。
      4. 把选中的样本移出池，返回 BatchDecision。

    为什么是"padded"预算 / Why padded budget
    ------------------------------------------
    训练时同一批会被 pad 到等长，真实显存占用 ≈ 最长样本长度 × 批大小 (一个矩形)，
    而不是各样本长度之和。所以预算判断用 `proposed_longest * proposed_batch_size`，
    这也正是先按长度排序的原因: 让矩形尽量"瘦",padding 浪费最小。

    防饿死 / Anti-starvation
    ------------------------
    纯按长度排序会让某些长度的样本长期排不到队首。_choose_anchor_index 会优先挑
    "等待步数 ≥ sample_pool_size"的最老样本当锚点，给长尾样本一个被成批的机会，
    以此在"打包效率"和"样本延迟/公平性"之间折中。

    参数 / Args:
        max_audio_tokens_in_batch: 一批 audio latent token 的 padded 预算上限。
        max_text_tokens_in_batch : 一批 text token 的 padded 预算上限。
        max_batch_size           : 批内最大样本数 (None 表示不另设上限)。
        sample_pool_size         : 候选池容量;同时也是上面老化阈值的步数。
            池越大 → 排序/桶化越充分、打包越紧，但前瞻越多、单样本延迟越高。
        profiler                 : 可选的 DataProfiler，用于埋点排序/决策耗时。
    """

    def __init__(
        self,
        *,
        max_audio_tokens_in_batch: int,
        max_text_tokens_in_batch: int,
        max_batch_size: int | None,
        sample_pool_size: int,
        profiler=None,
    ):
        # max(1, ...) 兜底: 预算/池容量至少为 1，避免 0 或负值导致死循环或无法成批。
        self.max_audio_tokens_in_batch = max(1, int(max_audio_tokens_in_batch))
        self.max_text_tokens_in_batch = max(1, int(max_text_tokens_in_batch))
        self.max_batch_size = max_batch_size
        self.sample_pool_size = max(1, int(sample_pool_size))
        self.profiler = ensure_data_profiler(profiler)

    @staticmethod
    def _sort_pool(pool: list[_PoolSample]) -> None:
        """按长度降序对候选池排序 (length bucketing) / Sort pool by length, desc.

        排序键 = (audio token 数, text token 数, -到达步) 且整体 reverse=True:
          - 主键 audio、次键 text 都取降序 → 长度相近的样本被排到一起，相邻打包时
            padding 浪费最小 (这正是 length bucketing 的效果)。
          - 末键用 -arrival_step 配合 reverse=True ⇒ 实际按 arrival_step **升序**，
            即同长度下先到先排前面 (稳定的先来先服务，配合防饿死逻辑)。
        原地排序 (in-place)，无返回值。
        """
        pool.sort(
            key=lambda item: (
                item.num_audio_tokens,
                item.num_text_tokens,
                -item.arrival_step,  # reverse=True 下取负 ⇒ arrival_step 升序 (老的排前)
            ),
            reverse=True,
        )

    def _choose_anchor_index(
        self,
        pool: list[_PoolSample],
        *,
        decision_step: int,
    ) -> int:
        """选本批锚点样本的下标 / Pick the anchor sample index for this batch.

        锚点 = 决定本批"最长长度"维度的起点样本。两种策略二选一:
          - 防饿死优先: 若池里存在已等待 ≥ sample_pool_size 步的样本，则挑其中
            **最老**的那个当锚点，保证长尾样本不会被无限期推迟 (anti-starvation)。
          - 否则: 返回 0，即排好序后的"最长样本"当锚点 (length bucketing 默认行为)。

        参数 / Args:
            pool         : 已由 _sort_pool 排过序的候选池 (此处不再依赖其有序性)。
            decision_step: 当前决策步,用于计算每个样本的等待步数。
        返回 / Returns: 锚点在 pool 中的下标。
        """
        oldest_waiting_index = -1
        oldest_waiting_step = decision_step  # 初值=当前步;只有更老 (更小) 的才会替换它

        for index, item in enumerate(pool):
            waited_steps = decision_step - item.arrival_step
            if waited_steps < self.sample_pool_size:
                continue  # 还没等够 sample_pool_size 步,不触发防饿死,跳过
            # 在"等够了"的候选中取 arrival_step 最小 (到达最早=等待最久) 的那个。
            # <= 保证同样老时取后出现的那个 (这里取谁差别不大,关键是命中最老批次)。
            if item.arrival_step <= oldest_waiting_step:
                oldest_waiting_index = index
                oldest_waiting_step = item.arrival_step

        # 没有任何样本等待超阈值 → 退回下标 0 (排序后最长样本) 作为锚点。
        return 0 if oldest_waiting_index < 0 else oldest_waiting_index

    def _build_next_decision(
        self,
        pool: list[_PoolSample],
        *,
        decision_step: int,
    ) -> BatchDecision:
        """从当前池贪心构造一个批 / Greedily build one batch from the pool.

        步骤 / Steps:
          1. 选锚点 anchor (见 _choose_anchor_index)。
          2. 若锚点**单条**就超出任一预算 → 无法成批,丢弃并告警,本步只返回 dropped。
          3. 否则以锚点起批,遍历池中其余样本,凡加入后 padded 预算
             (最长长度 × 批大小) 仍达标的就纳入,直到达到 max_batch_size 或塞不下。
          4. 把选中样本从池中移除 (倒序 pop 以免下标错位),返回 BatchDecision。

        参数 / Args:
            pool         : 已排序的候选池,本方法会**原地修改** (弹出被选/被丢样本)。
            decision_step: 当前决策步,透传给锚点选择做老化判断。
        返回 / Returns: 本步的 BatchDecision (batch_samples 与 dropped_samples 二者
            至少一非空;若都空,调用方会判定"未推进"并报错)。
        """
        dropped_samples: list[dict] = []
        batch_samples: list[dict] = []
        selected_indices: list[int] = []  # 记录被选入本批的池下标,末尾统一移除
        anchor_index = self._choose_anchor_index(pool, decision_step=decision_step)
        anchor = pool[anchor_index]

        # 单条样本若已撑爆任一预算,则它永远无法和别人成批 → 直接丢弃 (而非死循环)。
        exceed_audio_budget = anchor.num_audio_tokens > self.max_audio_tokens_in_batch
        exceed_text_budget = anchor.num_text_tokens > self.max_text_tokens_in_batch
        # max_batch_size < 1 是配置非法 (一批连一个都放不下),也归入"单条即超限"分支。
        exceed_batch_size = self.max_batch_size is not None and self.max_batch_size < 1
        if exceed_audio_budget or exceed_text_budget or exceed_batch_size:
            skipped = pool.pop(anchor_index).sample  # 把这条超限样本移出池并丢弃
            dropped_samples.append(skipped)
            warnings.warn(
                "Skipping sample that exceeds batching limits on its own: "
                f"fid={skipped.get('fid')!r}, "
                f"num_audio_tokens={anchor.num_audio_tokens}, "
                f"input_ids_length={anchor.num_text_tokens}, "
                f"max_audio_tokens_in_batch={self.max_audio_tokens_in_batch}, "
                f"max_text_tokens_in_batch={self.max_text_tokens_in_batch}, "
                f"max_batch_size={self.max_batch_size}",
                RuntimeWarning,
                stacklevel=2,
            )
            return BatchDecision(
                dropped_samples=dropped_samples,
                batch_samples=batch_samples,
            )

        # 用锚点起批,并以锚点长度初始化"本批当前最长长度"(padding 维度的当前值)。
        longest_audio_tokens = anchor.num_audio_tokens
        longest_text_tokens = anchor.num_text_tokens
        batch_samples.append(anchor.sample)
        selected_indices.append(anchor_index)

        for index, item in enumerate(pool):
            if index == anchor_index:
                continue  # 锚点已入批,跳过自身
            if (
                self.max_batch_size is not None
                and len(batch_samples) >= self.max_batch_size
            ):
                break  # 已达样本数上限,停止再纳入

            # 试探把 item 加入后的"新批大小"与"新最长长度"——预算按 padded 口径估算。
            proposed_batch_size = len(batch_samples) + 1
            proposed_longest_audio_tokens = max(
                longest_audio_tokens,
                item.num_audio_tokens,
            )
            proposed_longest_text_tokens = max(
                longest_text_tokens,
                item.num_text_tokens,
            )
            # padded 总量 = 最长长度 × 批大小 (pad 到等长后的矩形面积)。任一维超预算
            # 就跳过这条 (continue 而非 break: 池按长度降序,后面更短的样本可能反而塞得下)。
            if (
                proposed_longest_audio_tokens * proposed_batch_size
                > self.max_audio_tokens_in_batch
            ):
                continue
            if (
                proposed_longest_text_tokens * proposed_batch_size
                > self.max_text_tokens_in_batch
            ):
                continue

            # 通过预算检查 → 正式纳入本批,并更新当前最长长度。
            batch_samples.append(item.sample)
            selected_indices.append(index)
            longest_audio_tokens = proposed_longest_audio_tokens
            longest_text_tokens = proposed_longest_text_tokens

        # 倒序 (从大到小) pop 被选中的下标,避免前面元素被移除后导致后面下标失效。
        for index in sorted(set(selected_indices), reverse=True):
            pool.pop(index)

        return BatchDecision(
            dropped_samples=dropped_samples,
            batch_samples=batch_samples,
        )

    def build_decisions(self, sample_iter: Iterable[dict]) -> Iterator[BatchDecision]:
        """主循环: 把逐样本流转成 BatchDecision 流 / Stream samples → batch decisions.

        这是对外的生成器入口。每轮:
          1. **补池**: 从源 iterator 取样本填到候选池,直到池满 (sample_pool_size)
             或源耗尽;每条样本入池时缓存其长度并记下 arrival_step。
          2. **排序 + 决策**: 排序池 → _build_next_decision 产出一批 (或一次丢弃)。
          3. 只要本步有进展 (产出了 batch 或 dropped) 就 yield 并推进 decision_step。
        当源耗尽且池清空时结束。这是流式 (online) 设计: 不需要把整个数据集读进内存,
        只在内存里维护一个大小受限的池。

        参数 / Args:
            sample_iter: 产出样本 dict 的可迭代对象,样本需含 num_audio_tokens /
                input_ids_length 字段 (缺失则按 0 计)。
        产出 / Yields: 一连串 BatchDecision。
        异常 / Raises: RuntimeError —— 池非空却一步都推进不了 (理论上不该发生,
            作为防御性断言: 防止 bug 导致的静默死循环)。
        """
        pool: list[_PoolSample] = []
        source_exhausted = False  # 源 iterator 是否已 StopIteration
        decision_step = 0  # 单调递增的决策步计数,兼作样本 arrival 时间戳
        iterator = iter(sample_iter)

        # 主循环: 源没耗尽,或池里还有残留样本,就继续成批。
        while not source_exhausted or pool:
            # —— 补池阶段: 尽量把池填到 sample_pool_size,给排序/打包足够的前瞻空间 ——
            while not source_exhausted and len(pool) < self.sample_pool_size:
                try:
                    sample = next(iterator)
                except StopIteration:
                    source_exhausted = True  # 源到底,跳出补池;池里已有的样本仍要处理
                    break
                pool.append(
                    _PoolSample(
                        sample=sample,
                        # 长度字段缺失时退化为 0;int() 兜底防止字段是 numpy/字符串型。
                        num_audio_tokens=int(sample.get("num_audio_tokens", 0)),
                        num_text_tokens=int(sample.get("input_ids_length", 0)),
                        arrival_step=decision_step,  # 记录到达步,供老化/防饿死判断
                    )
                )

            if not pool:
                break  # 源耗尽且池已空 → 全部处理完毕,正常收尾

            profiler = self.profiler
            # 用 profiler 给两段热点 (排序、决策) 埋点计时,便于数据流水线性能分析。
            with profiler.measure("main.sort_pool", count=len(pool)):
                self._sort_pool(pool)
            with profiler.measure("main.build_batch_decision"):
                decision = self._build_next_decision(
                    pool,
                    decision_step=decision_step,
                )
            # 有进展 (产出批或丢弃了样本) 才推进步数并交付;否则视为异常 (见下)。
            if decision.dropped_samples or decision.batch_samples:
                decision_step += 1
                yield decision
                continue
            # 防御性断言: 池非空却既没成批也没丢弃,说明逻辑有 bug,宁可崩也别死循环。
            raise RuntimeError("OnlineBatcher failed to make progress on a non-empty pool.")
