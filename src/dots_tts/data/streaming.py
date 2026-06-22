"""流式训练数据管线 / Streaming training data pipeline.

本文件做什么 / What this file does:
    把一个"状态可恢复"(state-aware / resumable)的样本源 (``BaseSourceAdapter``)
    封装成 PyTorch 的 ``IterableDataset``，再叠加在线分桶 (online bucketing/batching)、
    pad-collate 和 token 级进度跟踪，最终对训练循环吐出一批批 padded 张量 batch。

    数据流位置 / Position in the data flow（自下而上）:
        source adapter (读分片 shard、解码音频/文本，产出 raw sample dict)
          -> StreamingSampleDataset (IterableDataset: 按 rank/worker 分片，逐样本 yield)
          -> DataLoader (多进程 worker，identity_collate 不做 batch，单样本透传)
          -> BatchedDataStream:
               OnlineBatcher.build_decisions  (按 audio/text token 预算在线攒 batch)
               PadCollator                    (把变长样本 pad 成 (B, T, ...) 张量)
               _DataStateTracker              (统计已发样本/各类 token 数，记录每个
                                               worker 的 adapter_state 以便断点续训)
          -> 训练循环消费 batch（peek/commit/discard 三段式提交协议）

    为什么要这么复杂 / Why so much machinery:
        TTS 训练样本是变长音频 latent，必须按 token 预算动态分桶才能填满 GPU；
        同时大规模流式训练要能从 checkpoint 精确续训(resume)，所以每个样本都携带
        其来源 adapter 的"下一步状态"，由 tracker 在 batch 真正提交时才推进，
        保证"已写入 checkpoint 的进度 == 已真正训练过的样本"。

关键类/函数清单 / Key classes & functions:
    identity_collate           —— DataLoader 的 collate_fn，单样本直通(不拼 batch)。
    StreamingSampleDataset     —— IterableDataset，负责分片 + 续训定位 + 逐样本产出。
    _DataStateTracker          —— 跨 worker 的进度/token 统计与 resume 状态聚合器。
    BatchedDataStream          —— 顶层封装：分桶 + collate + 三段式 batch 提交协议。
"""

from __future__ import annotations

import math
import multiprocessing as mp
from collections.abc import Iterable
from copy import deepcopy

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dots_tts.data.batchers import OnlineBatcher
from dots_tts.utils.profiling import ensure_data_profiler
from dots_tts.data.source_adapters.base_adapter import BaseSourceAdapter, SourceContext

# 样本 dict 内挂载续训元数据的私有 key；双下划线 + 私有命名避免与真实样本字段冲突。
# Private key used to attach resume/tracking metadata onto a sample dict.
_TRACKING_KEY = "__tracking_state__"
# resume state 顶层的"worker 拓扑"key：续训时必须和保存时的 world_size/worker 数一致，
# 否则分片切分不同、续训定位会错位。Key holding the saved worker topology for resume.
_RESUME_TOPOLOGY_KEY = "resume_topology"


def identity_collate(sample):
    """DataLoader 的 collate_fn：单样本直通，不做 batch 拼接。

    Identity collate: per-sample passthrough (no batching here).

    DataLoader 的 ``batch_size`` 设为 1 / batch 逻辑全部下放到 ``OnlineBatcher``，
    所以这里只需原样返回单个样本 dict，真正的分桶与 pad 在主进程侧完成。
    Batching is deferred to ``OnlineBatcher`` in the main process, so the loader
    just forwards each raw sample dict unchanged.
    """
    return sample


class StreamingSampleDataset(IterableDataset):
    """逐样本流式数据集 / Per-sample streaming ``IterableDataset``.

    职责 / Responsibility:
        在每个 DataLoader worker 内构造一个 ``SourceContext``（携带 epoch / rank /
        world_size / worker_id / seed），交给底层 ``source`` 适配器去做确定性分片
        (sharding) 和样本产出，并把 worker 标识写回样本以便上游做进度跟踪。

    设计要点 / Design notes:
        - 没有 ``__len__``：流式无限/未知长度数据集，靠 token 预算或 StopIteration 收尾。
        - epoch 用 ``multiprocessing.Value`` 跨进程共享：DataLoader 的多个 worker 是
          fork 出来的子进程，主进程调用 ``set_epoch`` 后子进程读到的必须是同一个值，
          因此用共享内存而非普通 int（"q" = signed long long）。
        - 续训 (resume): 主进程把整份 ``resume_state`` 暂存在 ``_pending_resume_state``，
          每个 worker 进入 ``__iter__`` 时按自己的 ``global_worker_id`` 取出对应分片的
          adapter_state，从断点处继续，而不是从头重放。

    Each DataLoader worker builds a ``SourceContext`` and lets the underlying
    ``source`` adapter shard deterministically and emit samples; resume state is
    looked up per global worker id so each worker continues from its checkpoint.
    """

    def __init__(
        self,
        *,
        source: BaseSourceAdapter,
        rank: int,
        world_size: int,
        seed: int,
    ):
        self.source = source
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        # 跨进程共享的 epoch 计数器（"q" = signed 64-bit）；DataLoader worker 是子进程，
        # 必须用共享内存才能看到主进程 set_epoch 的更新。Shared-memory epoch counter.
        self._epoch = mp.Value("q", 0)
        # 主进程暂存的整份续训状态；worker 进 __iter__ 时按需取走自己那份分片。
        # Resume state staged by the main process, consumed per worker on iteration.
        self._pending_resume_state: dict | None = None

    def load_state_dict(self, state: dict | None) -> None:
        """暂存续训状态 / Stage resume state (deep-copied) for later per-worker lookup."""
        # deepcopy 防止外部后续修改 state 影响到我们暂存的快照。
        self._pending_resume_state = deepcopy(state) if state else None

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch（跨进程可见）/ Set epoch, visible to all forked workers."""
        with self._epoch.get_lock():  # 共享 Value 需加锁保证读写原子性
            self._epoch.value = int(epoch)

    def _current_epoch(self) -> int:
        with self._epoch.get_lock():
            return int(self._epoch.value)

    def _take_resume_state(self, epoch: int) -> dict | None:
        """取走并清空与目标 epoch 匹配的续训状态；不匹配返回 None。

        Take (and clear) the staged resume state iff it belongs to ``epoch``.

        续训只在"保存时所处的那个 epoch"生效：进入下一个 epoch 时数据应从头重排，
        所以这里用 epoch 做匹配；取走后置空，保证一份 resume 状态只被消费一次。
        """
        if (
            self._pending_resume_state is None
            or int(self._pending_resume_state.get("epoch", -1)) != int(epoch)
        ):
            return None
        state = deepcopy(self._pending_resume_state)
        self._pending_resume_state = None
        return state

    @staticmethod
    def _validate_resume_topology(
        resume_state: dict,
        *,
        context: SourceContext,
        loader_num_workers: int,
    ) -> None:
        """校验续训拓扑一致性 / Guard that resume uses the same worker topology.

        为什么必须一致 / Why it must match:
            分片是按 ``index % global_worker_count == global_worker_id`` 切的（见
            ``ShardableSourceAdapter``）。一旦 world_size 或 每 rank 的 worker 数变了，
            同一个 global_worker_id 对应的样本子集就完全不同，续训定位会错乱、
            导致重复或漏样本。因此拓扑不一致直接抛错，宁可拒绝也不静默错配。

        Sharding is keyed on the global worker grid; changing it invalidates every
        saved per-worker offset, so we hard-fail instead of silently misaligning.
        """
        resume_topology = resume_state.get(_RESUME_TOPOLOGY_KEY)
        if not isinstance(resume_topology, dict):
            raise RuntimeError(
                "Resume state is missing required worker topology metadata."
            )
        expected_world_size = int(resume_topology["world_size"])
        expected_num_workers = int(resume_topology["loader_num_workers"])
        expected_global_worker_count = int(resume_topology["global_worker_count"])
        current_num_workers = int(loader_num_workers)
        current_global_worker_count = int(context.global_worker_count)
        if (
            expected_world_size != int(context.world_size)
            or expected_num_workers != current_num_workers
            or expected_global_worker_count != current_global_worker_count
        ):
            raise RuntimeError(
                "Resume requires the same data worker topology as the saved state. "
                f"saved(world_size={expected_world_size}, "
                f"num_workers_per_rank={expected_num_workers}, "
                f"global_worker_count={expected_global_worker_count}), "
                f"current(world_size={context.world_size}, "
                f"num_workers_per_rank={current_num_workers}, "
                f"global_worker_count={current_global_worker_count})."
            )

    def __iter__(self) -> Iterable[dict]:
        """构造本 worker 的迭代器并逐样本 yield / Build this worker's sample iterator.

        每个 DataLoader worker（含主进程单 worker 情形）调用一次。流程：
            1) 探测自身 worker 身份 -> 组装 SourceContext（分片所需的全部坐标）；
            2) 若本 epoch 有 resume 状态，校验拓扑并取出本 worker 对应的 adapter_state；
            3) 从该状态（或从头）让 source 产出样本，逐个打上 worker 标识后 yield。

        关键区分 / Subtle distinction:
            ``loader_num_workers`` 是 DataLoader 实际配置的 worker 数（单进程时为 0，
            用于拓扑校验/记录）；``effective_num_workers`` 是用于分片计算的"有效"数
            （单进程时取 1，避免除 0 / 让单进程也能正确切到全量数据）。
        """
        worker_info = get_worker_info()
        if worker_info is None:
            # 单进程加载（num_workers=0）：本"worker"就是主进程，id=0、有效 worker 数为 1。
            worker_id = 0
            loader_num_workers = 0
            effective_num_workers = 1
        else:
            worker_id = worker_info.id
            loader_num_workers = worker_info.num_workers
            effective_num_workers = worker_info.num_workers

        epoch = self._current_epoch()
        # SourceContext 把分片所需坐标打包成不可变 dataclass，交给 adapter 做确定性切分。
        context = SourceContext(
            epoch=epoch,
            rank=self.rank,
            world_size=self.world_size,
            worker_id=worker_id,
            num_workers=effective_num_workers,
            seed=self.seed,
        )
        resume_state = self._take_resume_state(epoch)
        if resume_state is not None:
            # 有续训状态才校验拓扑：拓扑不符直接抛错（见 _validate_resume_topology）。
            self._validate_resume_topology(
                resume_state,
                context=context,
                loader_num_workers=loader_num_workers,
            )
        # 从整份 resume 状态里挑出"本 global worker"那一份；其它 worker 的状态与我无关。
        # "workers" 以字符串化的 global_worker_id 为键（JSON/序列化友好）。
        worker_state = (
            None
            if resume_state is None
            else (resume_state.get("workers") or {}).get(str(context.global_worker_id))
        )
        # 把本 worker 的 adapter_state 传给 source：None 表示从该分片头部开始，
        # 非 None 表示从断点处继续。Resume from saved adapter_state, else start fresh.
        sample_iter = self.source.iter_samples(
            context,
            state=None if worker_state is None else worker_state.get("adapter_state"),
        )
        for sample in sample_iter:
            # 给每个样本盖上 worker 标识戳，供下游 tracker 按 worker 聚合进度/续训状态。
            sample["data_worker_id"] = context.worker_id
            sample["data_global_worker_id"] = context.global_worker_id
            yield sample


class _DataStateTracker:
    """主进程侧的进度/续训状态聚合器 / Main-process progress & resume tracker.

    职责 / Responsibility:
        - 统计本 epoch 已发出的样本数与各类 token 数（text / audio / total），用于
          判断 epoch 是否按 token 预算结束 (``should_stop``)。
        - 为每个 (global) worker 记录"最新已提交样本对应的 adapter_state + 顺序号"，
          这就是 checkpoint 里的可续训状态 (``state_dict`` 的 ``workers`` 字段)。

    为什么进度只在 commit 时推进 / Why progress advances only on commit:
        batch 是"先 peek 后 commit"两段式的：训练步可能因梯度异常等丢弃整个 batch
        (``discard``)。只有真正 commit 的样本才计入进度并推进 worker 的 adapter_state，
        从而保证 checkpoint 记录的 == 实际训练过的，续训不重不漏 (exactly-once)。

    Per-worker adapter states + token counters; progress is only advanced when a
    batch is actually committed, giving exactly-once resume semantics.
    """

    def __init__(self, *, num_tokens_per_epoch: int | None):
        # None 表示不以 token 数封顶 epoch（靠 source StopIteration 自然结束）。
        self.num_tokens_per_epoch = (
            None if num_tokens_per_epoch is None else int(num_tokens_per_epoch)
        )
        self._pending_state: dict | None = None
        self._reset_for_epoch(epoch=0)

    def _reset_for_epoch(self, *, epoch: int) -> None:
        """清零本 epoch 的所有计数与 per-worker 状态 / Reset all counters for a fresh epoch."""
        self.epoch = int(epoch)
        self.samples_emitted = 0
        self.num_text_tokens = 0
        self.num_audio_tokens = 0
        self.num_total_tokens = 0
        # global_worker_id(str) -> {adapter_state, sample_order}：每 worker 的续训游标。
        self.workers: dict[str, dict] = {}
        # global_worker_id(str) -> 下一个待分配的样本顺序号，用于给样本编序、判断新旧。
        self._next_sample_order_by_worker: dict[str, int] = {}

    def load_state_dict(self, state: dict | None) -> None:
        """暂存待恢复的进度状态 / Stage progress state, applied later by ``set_epoch``."""
        self._pending_state = deepcopy(state) if state else None

    def set_epoch(self, epoch: int) -> None:
        """切到目标 epoch：能续训则恢复进度，否则清零重来。

        Switch to ``epoch``: restore staged progress if it matches, else reset.

        只有当暂存状态的 epoch 与目标 epoch 相同才执行恢复（同 dataset 的设计，
        续训只在保存时那个 epoch 内生效）；恢复后清空 pending，保证只消费一次。
        """
        if self._pending_state is not None and int(
            self._pending_state.get("epoch", -1)
        ) == int(epoch):
            state = deepcopy(self._pending_state)
            self._pending_state = None
            self.epoch = int(state.get("epoch", epoch))
            self.samples_emitted = int(state.get("samples_emitted", 0))
            self.num_text_tokens = int(state.get("num_text_tokens", 0))
            self.num_audio_tokens = int(state.get("num_audio_tokens", 0))
            self.num_total_tokens = int(state.get("num_total_tokens", 0))
            self.workers = deepcopy(state.get("workers") or {})
            # 从每 worker 已提交的 sample_order 推回"下一个顺序号"(= 已提交 + 1)，
            # 这样续训后新样本的编序与保存前无缝衔接。Rebuild next-order cursors (+1).
            self._next_sample_order_by_worker = {
                worker_key: int((worker_state or {}).get("sample_order", -1)) + 1
                for worker_key, worker_state in self.workers.items()
            }
            return
        self._reset_for_epoch(epoch=int(epoch))

    def should_stop(self) -> bool:
        """是否已达到本 epoch 的 token 预算 / Whether the per-epoch token budget is hit."""
        return (
            self.num_tokens_per_epoch is not None
            and self.num_total_tokens >= self.num_tokens_per_epoch
        )

    def stage_sample(self, sample: dict) -> dict:
        """登记一个刚从 loader 取到的样本，挂上续训元数据后返回。

        Stage a freshly-loaded sample: strip transport fields, attach tracking meta.

        做了什么 / What it does:
            - 取出并移除样本上的 worker 标识戳和 source 携带的 ``_adapter_state``
              （这些是"运输用"字段，不应进入模型 batch）；
            - 给该 worker 分配一个递增的 ``sample_order``（用于后续判断新旧、去重推进）；
            - 把续训所需信息打包进私有 ``_TRACKING_KEY``，随样本一路带到 commit 阶段。

        注意 / Note: 这里只是"登记/暂存"，尚未计入 epoch 进度——进度在 commit 才推进。
        """
        item = dict(sample)  # 浅拷贝，避免就地修改 loader 产出的原 dict
        worker_key = str(item.pop("data_global_worker_id"))
        item.pop("data_worker_id", None)
        # adapter_state 是 source 给出的"产出该样本后的下一步状态"，续训定位的关键。
        adapter_state = item.pop("_adapter_state", None)
        sample_order = int(self._next_sample_order_by_worker.get(worker_key, 0))
        self._next_sample_order_by_worker[worker_key] = sample_order + 1
        item[_TRACKING_KEY] = {
            "worker_key": worker_key,
            "adapter_state": deepcopy(adapter_state),  # deepcopy 防止后续被原地改动
            "sample_order": sample_order,
            "num_text_tokens": int(item["num_text_tokens"]),
            "num_audio_tokens": int(item["num_audio_tokens"]),
            # total_tokens 缺省时退化为 input_ids_length（纯文本/无显式 total 的情形）。
            "num_total_tokens": int(
                item.get("num_total_tokens", item["input_ids_length"])
            ),
        }
        return item

    def _pop_tracking(self, sample: dict) -> tuple[dict, dict]:
        """从样本剥离续训元数据，返回 (干净样本, tracking)。

        Split a staged sample into (clean item, tracking metadata).

        干净样本即可送入 collator / 模型；tracking 用于推进 worker 续训游标。
        缺失元数据视为内部协议被破坏，直接抛错而非静默吞掉。
        """
        item = dict(sample)
        tracking = item.pop(_TRACKING_KEY, None)
        if not isinstance(tracking, dict):
            raise RuntimeError("Tracked sample is missing internal resume metadata.")
        return item, tracking

    def _advance_worker(self, tracking: dict) -> None:
        """把某 worker 的续训游标前推到该样本之后 / Advance a worker's resume cursor.

        关键不变量 / Key invariant:
            只在 ``sample_order`` 严格大于已记录值时才更新（单调推进）。同一 batch 内
            样本来自多个 worker、顺序可能被分桶打乱，用 sample_order 比较可保证每个
            worker 的 adapter_state 只朝前走、不会被乱序的旧样本回退覆盖。
            adapter_state 为 None（如样本不携带状态）则跳过，不破坏已有游标。
        """
        adapter_state = tracking.get("adapter_state")
        if adapter_state is None:
            return
        worker_key = str(tracking["worker_key"])
        sample_order = int(tracking.get("sample_order", -1))
        current_state = self.workers.get(worker_key)
        current_order = int((current_state or {}).get("sample_order", -1))
        if current_order >= sample_order:  # 乱序到来的旧样本：不回退，直接忽略
            return
        self.workers[worker_key] = {
            "adapter_state": deepcopy(adapter_state),
            "sample_order": sample_order,
        }

    def mark_samples_dropped(self, samples: list[dict]) -> None:
        """处理被 batcher 丢弃的样本：推进 worker 游标但不计入 token 进度。

        Account for batcher-dropped samples: advance resume cursor, but do NOT
        count their tokens.

        为什么仍要推进游标 / Why still advance:
            被 batcher 丢弃（如超长、无法装桶）的样本已经"消费"过了，续训时不应再产出，
            所以要把 adapter_state 前推越过它们；但它们没进入任何 batch、未被训练，
            因此不累加 token 计数。
        """
        for sample in samples:
            _, tracking = self._pop_tracking(sample)
            self._advance_worker(tracking)

    def commit_batch(self, samples: list[dict]) -> list[dict]:
        """正式提交一批样本：推进游标 + 累加 token 进度，返回干净样本列表。

        Commit a batch: advance per-worker cursors AND accumulate token counters.

        与 ``mark_samples_dropped`` 的区别：commit 的样本确实被训练，故既推进续训
        游标、又计入 epoch 的样本/Token 进度，这才是 ``should_stop`` 判停的依据。
        """
        committed: list[dict] = []
        for sample in samples:
            item, tracking = self._pop_tracking(sample)
            self._advance_worker(tracking)
            self.samples_emitted += 1
            self.num_text_tokens += int(tracking["num_text_tokens"])
            self.num_audio_tokens += int(tracking["num_audio_tokens"])
            self.num_total_tokens += int(tracking["num_total_tokens"])
            committed.append(item)
        return committed

    def state_dict(self) -> dict:
        """导出可写入 checkpoint 的进度/续训状态 / Serialize progress for checkpointing.

        deepcopy ``workers`` 防止外部持有的快照随后续训练被原地修改。
        """
        return {
            "epoch": int(self.epoch),
            "samples_emitted": int(self.samples_emitted),
            "num_text_tokens": int(self.num_text_tokens),
            "num_audio_tokens": int(self.num_audio_tokens),
            "num_total_tokens": int(self.num_total_tokens),
            "workers": deepcopy(self.workers),
            "num_tokens_per_epoch": self.num_tokens_per_epoch,
        }


class BatchedDataStream:
    """顶层批数据流 / Top-level batched data stream.

    把 ``StreamingSampleDataset`` 经 DataLoader 产出的单样本，串联成：
        在线分桶 (``OnlineBatcher``) -> pad-collate (``PadCollator``) ->
        三段式 batch 提交协议 (peek / commit / discard) + 进度跟踪。

    三段式提交协议 / Three-phase batch protocol（``peek_batch`` 文档详述）:
        1) ``peek_batch``  —— 攒出并 collate 一个 batch，但**不**计入进度，可重复 peek；
        2) ``commit_batch``—— 训练步成功后调用，正式计入进度并推进续训游标；
        3) ``discard_batch``—— 训练步失败/跳过时调用，丢弃 pending、不动进度。
        这让"checkpoint 进度 == 实际训练步数"，配合 ``state_dict`` 实现精确续训。

    设计要点 / Design notes:
        - 不能在有 pending batch 时序列化（见 ``state_dict``），否则进度会处于半提交态。
        - 迭代状态（各 iterator / pending）可整体重置 (``_reset_iteration_state``)，
          用于 set_epoch / load_state_dict / close 后重新建流。
    """

    def __init__(
        self,
        *,
        sample_dataset: StreamingSampleDataset,
        data_cfg,
        tokenizer,
        num_tokens_per_epoch: int | None,
        profiler=None,
    ):
        from dots_tts.data.collator import PadCollator

        self.sample_dataset = sample_dataset
        self.profiler = ensure_data_profiler(profiler)
        # 由"音频采样率 / 每个 LLM token 对应的音频采样点数"算出每秒音频折合多少 LLM token，
        # 用来把"按秒计的 batch 预算"换算成"按 token 计的预算"。
        llm_token_rate = (
            float(data_cfg.train_audio_sample_rate)
            / float(data_cfg.audio_samples_per_llm_token)
        )
        self.batcher = OnlineBatcher(
            # 秒预算 × token 速率 -> 每 batch 音频 token 上限；ceil 且至少为 1，防止取到 0。
            max_audio_tokens_in_batch=max(
                1,
                math.ceil(float(data_cfg.max_audio_seconds_in_batch) * llm_token_rate),
            ),
            max_text_tokens_in_batch=data_cfg.max_text_tokens_in_batch,
            max_batch_size=data_cfg.max_samples_per_batch,
            sample_pool_size=data_cfg.bucketing_pool_size,  # 分桶池大小：越大越能凑齐等长样本
            profiler=self.profiler,
        )
        self.sample_loader = None  # 由 attach_loader 注入的 DataLoader
        self.collator = PadCollator(tokenizer)
        self.data_state = _DataStateTracker(
            num_tokens_per_epoch=num_tokens_per_epoch
        )
        # 以下为迭代期的瞬时状态，全部可被 _reset_iteration_state 清空：
        self._decision_iterator = None  # batcher.build_decisions 的迭代器
        self._sample_iterator = None    # DataLoader 的迭代器
        self._pending_batch = None      # 已 collate 但未 commit 的张量 batch
        self._pending_samples = None    # 与 _pending_batch 对应的原始样本（带 tracking）

    def attach_loader(self, loader: DataLoader) -> None:
        """注入 DataLoader（其底层 dataset 应为本流的 sample_dataset）。Attach the loader."""
        self.sample_loader = loader

    def close(self) -> None:
        """关闭数据流：重置迭代状态并断开 loader / Tear down iteration and detach loader."""
        self._reset_iteration_state()
        self.sample_loader = None

    def load_state_dict(self, state: dict | None) -> None:
        """加载续训状态：同步下发给 tracker 与 dataset，并重置迭代器。

        Load resume state into both the tracker and the dataset, then reset iter.
        """
        self.data_state.load_state_dict(state)
        self.sample_dataset.load_state_dict(state)
        self._reset_iteration_state()

    def state_dict(self) -> dict:
        """导出可续训的完整状态（进度 + worker 拓扑）/ Export full resumable state.

        额外写入 ``_RESUME_TOPOLOGY_KEY``：world_size / 每 rank worker 数 / 全局 worker
        总数。续训时 dataset 会据此校验拓扑必须一致（见 _validate_resume_topology）。

        前置条件 / Preconditions:
            - 必须已 attach loader（要读 num_workers）；
            - 不能存在 pending batch：半提交态序列化会让进度记账不自洽，故直接抛错。
        """
        if self.sample_loader is None:
            raise RuntimeError("BatchedDataStream has no attached sample loader.")
        if self._pending_batch is not None or self._pending_samples is not None:
            raise RuntimeError(
                "Cannot serialize BatchedDataStream while a batch is pending commit."
            )
        loader_num_workers = int(getattr(self.sample_loader, "num_workers", 0))
        # 与 dataset.__iter__ 一致：单进程(num_workers=0)时有效 worker 数按 1 计。
        effective_num_workers = max(1, loader_num_workers)
        state = self.data_state.state_dict()
        state[_RESUME_TOPOLOGY_KEY] = {
            "world_size": int(self.sample_dataset.world_size),
            "loader_num_workers": loader_num_workers,
            "global_worker_count": int(self.sample_dataset.world_size)
            * effective_num_workers,
        }
        return state

    def set_epoch(self, epoch: int) -> None:
        """切换 epoch：同步 dataset 与 tracker，并重建迭代流。Switch epoch on both ends."""
        self.sample_dataset.set_epoch(epoch)
        self.data_state.set_epoch(epoch)
        self._reset_iteration_state()

    def _reset_iteration_state(self) -> None:
        """重置所有迭代期瞬时状态 / Reset all transient iteration state.

        先尝试关闭 decision 迭代器（generator.close()，会触发 _iter_staged_samples 的
        finally 清理并间接释放 DataLoader 的 worker），再把所有 pending 句柄置空。
        """
        close_iterator = getattr(self._decision_iterator, "close", None)
        if callable(close_iterator):
            close_iterator()  # 关闭生成器，确保底层 loader/worker 资源被释放
        self._decision_iterator = None
        self._sample_iterator = None
        self._pending_batch = None
        self._pending_samples = None

    def _iter_staged_samples(self):
        """从 loader 拉样本，登记 tracking 后逐个 yield（喂给 batcher 的上游）。

        Pull samples from the loader, stage tracking metadata, yield to the batcher.

        终止条件 / Termination:
            - ``should_stop()`` 为真（达到 token 预算）即停；
            - loader 耗尽 (``StopIteration``) 即停。
        ``sample is None`` 表示该 worker 本次无可用样本（如分片为空），跳过即可。
        finally 块清空 ``_sample_iterator``，配合生成器 close 释放 loader worker。
        """
        if self.sample_loader is None:
            raise RuntimeError("BatchedDataStream has no attached sample loader.")
        self._sample_iterator = iter(self.sample_loader)
        profiler = self.profiler
        try:
            while True:
                if self.data_state.should_stop():  # 已达 token 预算，提前收尾
                    return
                try:
                    with profiler.measure("main.loader_wait_next_sample"):
                        sample = next(self._sample_iterator)
                except StopIteration:
                    return
                if sample is None:
                    continue  # 空样本占位：跳过，不计数也不中断
                with profiler.measure("main.stage_sample"):
                    staged = self.data_state.stage_sample(sample)
                yield staged
        finally:
            self._sample_iterator = None

    def _decision_stream(self):
        """惰性构造并缓存 batcher 的决策迭代器 / Lazily build & cache the decision stream.

        ``OnlineBatcher.build_decisions`` 消费 staged 样本流，产出一系列
        ``BatchDecision``（含 ``dropped_samples`` 和 ``batch_samples``）。
        缓存到 ``_decision_iterator`` 以便跨多次 ``peek_batch`` 连续推进同一个流。
        """
        if self._decision_iterator is None:
            self._decision_iterator = iter(
                self.batcher.build_decisions(self._iter_staged_samples())
            )
        return self._decision_iterator

    def peek_batch(self) -> tuple[dict | None, bool]:
        """攒出下一个 batch 并 collate，但不提交进度 / Produce next batch without committing.

        返回 / Returns:
            (batch_dict_or_None, has_batch_bool)。
            - 已有 pending（上次 peek 未 commit/discard）时直接复用，幂等可重复 peek；
            - 否则推进 decision 流：丢弃样本先登记 (mark_samples_dropped)，遇到首个
              非空 ``batch_samples`` 就 collate 成张量并暂存为 pending；
            - decision 流耗尽则返回 ``(None, False)`` 表示本 epoch 数据已尽。

        注意 / Note: 进度（token/样本计数、worker 游标）在此**不**推进，必须等
        ``commit_batch`` 调用；这正是支持"训练步失败可 discard 重来"的关键。
        """
        if self._pending_batch is not None:
            return self._pending_batch, True  # 幂等：已有 pending 直接返回

        for decision in self._decision_stream():
            if decision.dropped_samples:
                # 被 batcher 丢弃的样本：推进续训游标但不计 token（见 mark_samples_dropped）。
                self.data_state.mark_samples_dropped(decision.dropped_samples)
            if not decision.batch_samples:
                continue  # 本次决策只丢样本、未凑出 batch，继续拉下一个决策
            self._pending_samples = decision.batch_samples
            with self.profiler.measure(
                "main.collate_batch",
                count=len(decision.batch_samples),
            ):
                # PadCollator 把变长样本 pad 成 (B, T, ...) 张量 batch。
                self._pending_batch = self.collator(decision.batch_samples)
            return self._pending_batch, True
        return None, False

    def commit_batch(self) -> dict:
        """提交当前 pending batch：计入进度、清空 pending，返回该 batch。

        Commit the pending batch: advance progress + clear pending. Must have peeked.
        """
        if self._pending_batch is None or self._pending_samples is None:
            raise RuntimeError("BatchedDataStream has no pending batch to commit.")
        pending_batch = self._pending_batch
        # 用带 tracking 的原始样本推进进度/续训游标；干净张量 batch 直接返回给调用方。
        self.data_state.commit_batch(self._pending_samples)
        self._pending_batch = None
        self._pending_samples = None
        return pending_batch

    def discard_batch(self) -> None:
        """丢弃当前 pending batch，不计入任何进度 / Drop pending batch, no progress change.

        用于训练步失败/被跳过时回滚：下次 ``peek_batch`` 会攒一个新 batch。
        """
        if self._pending_batch is None or self._pending_samples is None:
            raise RuntimeError("BatchedDataStream has no pending batch to discard.")
        self._pending_batch = None
        self._pending_samples = None

    def __iter__(self):
        """便捷迭代：peek -> 立即 commit -> yield，直到数据尽或达 token 预算。

        Convenience iteration: peek, immediately commit, yield, until exhausted.

        这是"自动提交"的简化用法（每个 batch 都直接计入进度）；需要"失败可丢弃"的
        精细控制时，调用方应改用 peek_batch / commit_batch / discard_batch 三段式接口。
        """
        while True:
            batch, has_batch = self.peek_batch()
            if not has_batch:
                return
            self.commit_batch()
            yield batch
            if self.data_state.should_stop():  # 提交后再次检查 token 预算
                return
