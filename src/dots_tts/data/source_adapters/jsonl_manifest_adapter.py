"""JSONL manifest source adapter / JSONL manifest 数据源 adapter。

本文件做什么 / What this file does:
    实现 ``JsonlManifestSourceAdapter`` —— 一个从 *line-delimited JSON*
    (JSONL，每行一个 JSON 对象) manifest 文件读取训练样本的有限 (finite)
    数据源 adapter。每条记录至少含 fid / text / audio 三个字段(键名可配置),
    被转成统一的 sample dict 交给上层 data pipeline。
    Reads one JSON object per line and yields normalized sample dicts.

在训练数据流里的位置 / Where it sits in the pipeline:
    source adapter 处于训练数据流的**最上游**:它只负责"产出一条条原始样本",
    并不做音频解码、tokenize、batching。下游会把 ``audio`` 路径喂给
    BigVGAN AudioVAE 编码成连续 latent、把 ``text`` 喂给 Qwen2.5 backbone 的
    tokenizer。本 adapter 继承自 ``ShardableSourceAdapter`` 以支持多 rank /
    多 worker 的确定性分片 (deterministic sharding),以及可断点续训的
    state(cycle + cursor),这样训练中断后能从上次 yield 的位置恢复。

关键类 / Key class:
    - ``JsonlManifestSourceAdapter`` —— 唯一的对外类,实现 ``BaseSourceAdapter``
      要求的 ``initial_state`` / ``iter_samples`` / ``is_cycle_start_state``,
      并按需覆写 ``advance_cycle`` 以支持数据集多轮循环 (repeated cycling)。

相关概念 / Concepts:
    - sharding:按 ``index % global_worker_count == global_worker_id`` 把样本
      切给不同 rank/worker,保证每条样本只被一个 worker 处理一次。
    - resumable state:每个 yield 出去的 sample 都挂上 ``_adapter_state``
      (下一个游标位置),checkpoint 时存下它即可精确续训。
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from dots_tts.data.source_adapters.base_adapter import (
    BaseSourceAdapter,
    ShardableSourceAdapter,
    SourceContext,
)


class JsonlManifestSourceAdapter(ShardableSourceAdapter, BaseSourceAdapter):
    """Finite adapter for line-delimited JSON manifests.

    职责 / Responsibility:
        把一个 JSONL manifest 文件变成可分片、可断点续训的样本流。
        "finite" 指 manifest 行数固定;若需多轮重复使用,靠 ``advance_cycle``
        递增 ``cycle`` 来开启下一轮(每轮可用不同 shuffle 顺序)。

    构造参数 / Constructor args:
        manifest_path: JSONL 文件路径。
        fid_key / text_key / audio_key: 记录里 file-id / 文本 / 音频字段的键名,
            可配置以适配不同来源的 manifest schema。
        shuffle: 是否在每个 cycle 内打乱样本顺序(打乱用确定性种子,见下)。
        encoding: 读文件的字符编码。

    设计要点 / Design notes:
        - 记录一次性读入内存并缓存在 ``self._records``(见 ``_base_records``),
          因为是 finite 数据集,后续每个 cycle 复用同一份内存数据,避免重复 IO。
        - shuffle 用 ``seed + epoch + 1009*cycle`` 作种子 → 同一 (seed, epoch,
          cycle) 在所有 rank/worker 上得到**一致**的打乱顺序,保证分片不重不漏。
        - state = ``{"cycle", "cursor"}``;cursor 是"下一条要产出的样本"在
          *已分片索引列表* 中的位置,断点续训时直接从该位置继续。
    """

    def __init__(
        self,
        *,
        manifest_path: str,
        fid_key: str = "fid",
        text_key: str = "text",
        audio_key: str = "audio",
        shuffle: bool = False,
        encoding: str = "utf-8",
    ):
        self.manifest_path = Path(manifest_path)
        self.fid_key = fid_key
        self.text_key = text_key
        self.audio_key = audio_key
        self.shuffle = shuffle
        self.encoding = encoding
        # 记录缓存:None 表示尚未读取;首次访问时 lazy-load 整个 manifest。
        # Lazy cache of all records; None until first read.
        self._records: list[dict[str, Any]] | None = None

    def initial_state(self) -> dict[str, Any]:
        """新迭代器的默认 state:第 0 轮、游标在 0 / Default fresh state."""
        return {"cycle": 0, "cursor": 0}

    def is_cycle_start_state(self, state: dict[str, Any] | None) -> bool:
        """state 是否指向某一 cycle 的起点 / Is the state at a cycle boundary.

        cursor == 0 即表示该 cycle 还没产出过任何样本(起点)。上层据此判断
        能否安全地 ``advance_cycle`` 进入下一轮。
        """
        normalized = self.normalize_state(state)
        return int(normalized["cursor"]) == 0

    def advance_cycle(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """进入下一轮:cycle+1、cursor 归零 / Move to the next cycle.

        覆写 base 的"默认不支持循环"行为,使该有限数据集可被反复遍历;
        cycle 递增会改变 shuffle 种子,从而下一轮得到不同的样本顺序。
        """
        normalized = self.normalize_state(state)
        return {"cycle": int(normalized["cycle"]) + 1, "cursor": 0}

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        """逐行解析 JSONL,产出原始 dict / Stream-parse the JSONL file.

        逐行读取以避免一次性把超大文件载入字符串;空行跳过;解析失败时把
        出错的文件名+行号包进 ``ValueError`` 抛出(``from exc`` 保留原始
        ``JSONDecodeError`` 便于定位坏数据)。
        """
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path!s}")
        with self.manifest_path.open("r", encoding=self.encoding) as fin:
            # line_no 从 1 起算 → 报错信息里的行号与编辑器一致,方便人工排查。
            for line_no, raw_line in enumerate(fin, start=1):
                line = raw_line.strip()
                if not line:
                    continue  # 跳过空行 / skip blank lines
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {self.manifest_path}:{line_no}"
                    ) from exc

    def _base_records(self) -> list[dict[str, Any]]:
        """返回(并缓存)全部记录列表 / Return all records, caching them.

        memoize:首次调用把 ``_iter_records`` 物化成 list 存进 ``self._records``,
        之后每个 cycle / worker 都复用这份内存数据,不再重读磁盘。
        """
        if self._records is None:
            self._records = list(self._iter_records())
        return self._records

    def _build_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        """把一条 manifest 记录归一化成统一的 sample dict / Normalize a record.

        - 强校验三个必需字段(缺任意一个就 ``KeyError``,并打印缺失键+整条记录)。
        - 输出 dict 的键名固定为 ``fid`` / ``text`` / ``audio``(下游按这套
          canonical 键名取值,屏蔽各 manifest 的原始键名差异);``fid`` 强转
          ``str`` 以兼容整数型 id。
        - 记录里其余的额外字段原样透传(如 speaker、lang 等条件信息),供
          下游 CAM++ x-vector / 多语种条件等使用。
        """
        missing = [
            key
            for key in (self.fid_key, self.text_key, self.audio_key)
            if key not in record
        ]
        if missing:
            raise KeyError(
                f"Manifest record is missing required keys {missing}: {record}"
            )

        sample = {
            "fid": str(record[self.fid_key]),
            "text": record[self.text_key],
            "audio": record[self.audio_key],
        }
        for key, value in record.items():
            # 已映射到 canonical 键的三个原始键跳过,避免重复/覆盖。
            if key in {self.fid_key, self.text_key, self.audio_key}:
                continue
            sample[key] = value  # 透传其余 metadata / pass through extra fields
        return sample

    def _indices_for_cycle(
        self,
        context: SourceContext,
        *,
        cycle: int,
    ) -> list[int]:
        """算出本 worker 在该 cycle 应处理的记录下标列表 / This worker's indices.

        返回 (list[int]):指向 ``_base_records()`` 的全局下标子集 —— 即经过
        shuffle(可选)再做 rank/worker 分片后,**只属于当前 worker** 的那批。

        关键在于 shuffle 与分片的顺序 / Why shuffle-then-shard:
            - shuffle 用 ``seed + epoch + 1009*cycle`` 作种子,所有 rank/worker
              用同一种子打乱 → 得到**完全一致**的全局排列 ``indices``;1009
              是个质数偏移,让不同 cycle 的排列互不相同。
            - 然后按"打乱后的位置"(``shuffled_index``)做分片判定,而不是按
              原始 record 下标 —— 这样每条样本在所有 worker 间不重不漏,且各
              worker 拿到的是打乱后均匀分布的一段。
            - 不 shuffle 时则直接按原始 record 下标分片,保持文件顺序。
        """
        indices = list(range(len(self._base_records())))
        if self.shuffle:
            random.Random(context.seed + context.epoch + 1009 * int(cycle)).shuffle(
                indices
            )
            # 注意:分片用打乱后的位置 shuffled_index 判定,留下的 value 仍是
            # 原始 record 下标 record_index → 既打乱又正确分片。
            indices = [
                record_index
                for shuffled_index, record_index in enumerate(indices)
                if self.is_assigned_index(shuffled_index, context)
            ]
        else:
            # 顺序模式:直接按原始下标分片,样本保持 manifest 中的行顺序。
            indices = [
                record_index
                for record_index in indices
                if self.is_assigned_index(record_index, context)
            ]
        return indices

    def iter_samples(
        self,
        context: SourceContext,
        *,
        state: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        """从 ``state`` 处续流,逐条产出本 worker 的样本 / Resume-able sample stream.

        参数 / Args:
            context: 当前 epoch / rank / worker / seed 等执行上下文。
            state: 续训用的 ``{"cycle", "cursor"}``;None 则从初始 state 开始。

        产出 / Yields:
            归一化 sample dict,且每条都额外挂上 ``_adapter_state`` = 产出该
            样本**之后**的游标 (cursor=position+1)。上层 checkpoint 存下它,
            重启时传回 ``state`` 即可精确从下一条继续,不重不漏。
        """
        live_state = self.normalize_state(state)
        cycle = int(live_state["cycle"])
        cursor = int(live_state["cursor"])
        records = self._base_records()
        # 先按 cycle 还原"本 worker 的下标列表"(含 shuffle+分片)。
        indices = self._indices_for_cycle(context, cycle=cycle)

        # 从 cursor 续起:跳过本 cycle 已产出过的前 cursor 条,实现断点续训。
        for position in range(cursor, len(indices)):
            sample = self._build_sample(records[indices[position]])
            sample["_adapter_state"] = {
                "cycle": cycle,
                "cursor": position + 1,  # 下一条的游标 / next-item cursor
            }
            yield sample
