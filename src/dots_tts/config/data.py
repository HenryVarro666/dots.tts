"""数据相关配置 / Data-pipeline configuration schema.

本文件用 pydantic 定义训练数据加载层的全部配置项 (schema)，它们在训练启动时
从 YAML/JSON 反序列化校验，再驱动 DataLoader、source adapter、batch 分桶
(bucketing) 等逻辑。属于"配置 (config)"层，本身不读音频、不做张量运算，只描述
"该怎么读、怎么组 batch"。

This module is the typed configuration layer for the training-time data pipeline.
All fields are validated by pydantic on load, then consumed downstream to build
DataLoaders, instantiate source adapters, and drive token-budget bucketing. It
performs no I/O and no tensor work — it only declares *how* data should be read
and batched.

在数据流中的位置 / Position in the data flow:
    YAML/JSON 配置 → (本文件做校验/类型化) → DataConfig 实例
                  → SourceAdapter (按 manifest 取样) → 分桶/打包 batch → 模型训练

关键类清单 / Key classes:
    - SourceAdapterConfig: 单个数据源适配器 (source adapter) 的类名 + 参数。
    - DataSourceConfig:    一个命名数据源 (含采样权重 weight 与 pipeline 模式)。
    - DataConfig:          顶层数据配置 (多数据源 + batch/分桶/DataLoader 预算)。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from dots_tts.config.base import StrictConfigBase

# 默认的 source adapter 实现类名；目前只支持基于 JSONL manifest 的取样器。
# Default source-adapter implementation; currently the only supported one
# reads samples from a JSONL manifest file.
DEFAULT_SOURCE_ADAPTER_CLASS_NAME = "JsonlManifestSourceAdapter"


class SourceAdapterConfig(StrictConfigBase):
    """数据源适配器配置 / Config for one source adapter.

    描述"用哪个类、传哪些参数"去把磁盘上的数据 (如 JSONL manifest) 解析成样本。
    用 ``Literal`` 把 ``class_name`` 锁死成已注册的实现，避免反序列化时注入任意类
    (一种安全/可控性约束)；具体路径、字段名等可调项通过自由字典 ``params`` 透传给
    该适配器构造函数。

    Declares which adapter class to instantiate and what kwargs to pass it when
    turning on-disk data (e.g. a JSONL manifest) into samples. ``class_name`` is
    constrained to a ``Literal`` whitelist so config can't request an arbitrary
    class; per-adapter knobs flow through the open ``params`` dict.

    字段 / Fields:
        class_name: 适配器实现类名 (当前仅 JsonlManifestSourceAdapter)。
        params:     透传给适配器的关键字参数；缺省为空 dict。
    """

    class_name: Literal["JsonlManifestSourceAdapter"] = (
        DEFAULT_SOURCE_ADAPTER_CLASS_NAME
    )
    params: dict[str, Any] = Field(default_factory=dict)


class DataSourceConfig(StrictConfigBase):
    """单个命名数据源配置 / Config for one named data source.

    一次训练通常混合多个数据源 (不同语料库/语种/录音质量)。每个源有唯一 ``name``、
    一个采样权重 ``weight`` (>0)，以及一个 ``adapter`` 决定如何读取。``weight`` 控制
    多源混采 (interleaving) 时该源被抽到的相对概率——权重越大，样本越频繁出现。

    A training run usually mixes several corpora. Each source has a unique
    ``name``, a sampling ``weight`` (>0) governing its relative draw probability
    when interleaving multiple sources, and an ``adapter`` describing how to read
    it. Larger weight ⇒ this source's samples appear more often.

    字段 / Fields:
        name:     数据源唯一名 (DataConfig 会校验全局不重名)。
        weight:   混采采样权重，必须 > 0；默认 1.0 即等权。
        pipeline: 该源的处理流水线模式——"basic" 逐样本，"interleave" 多源交织。
        adapter:  该源使用的 SourceAdapterConfig。
    """

    name: str
    weight: float = Field(default=1.0, gt=0.0)
    pipeline: Literal["basic", "interleave"] = "basic"
    adapter: SourceAdapterConfig = Field(default_factory=SourceAdapterConfig)


class DataConfig(StrictConfigBase):
    """顶层数据配置 / Top-level data-pipeline config.

    汇总一次训练所需的全部数据侧设置：多数据源混采、音频/文本 token 的换算关系、
    DataLoader 性能旋钮，以及 batch 打包与分桶 (bucketing) 的预算上限。

    Aggregates everything the data side of a training run needs: the set of mixed
    sources, the audio↔LLM-token correspondence, DataLoader performance knobs, and
    the budget caps used when packing/bucketing variable-length samples into
    batches.

    关键字段释义 / Notable fields:
        train_audio_sample_rate:   训练音频采样率 (Hz)，数据须按此重采样。
        audio_samples_per_llm_token: 每个 LLM token 对应多少个音频采样点——把音频时长
            换算成 token 长度的桥梁 (连续潜在 / continuous-latent 模型据此对齐文本与
            声学序列)。
        num_tokens_per_epoch:      一个 epoch 跨所有 rank 的全局 token 预算 (可选)。
        max_audio_seconds_in_batch / max_text_tokens_in_batch / max_samples_per_batch:
            动态 batch 的三道上限——按音频秒数、文本 token 数、样本条数封顶，任一触顶
            即收口当前 batch (变长序列打包，避免 OOM)。
        bucketing_pool_size:       分桶缓冲池大小——先攒这么多样本再按长度近邻分桶，
            减少同一 batch 内的 padding 浪费 (越大分桶越优但内存/打乱代价越高)。

    设计要点 / Design note:
        继承 StrictConfigBase ⇒ extra="forbid"，配置里出现未声明字段会直接报错，
        防止打错字的旋钮被静默忽略。
    """

    sources: list[DataSourceConfig]
    train_audio_sample_rate: int = Field(ge=1)
    audio_samples_per_llm_token: int = Field(ge=1)
    num_tokens_per_epoch: int | None = Field(
        default=None,
        ge=1,
        description="Global token budget across all ranks for one training epoch.",
    )
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    prefetch_factor: int = Field(
        default=2,
        ge=1,
        description="Samples prefetched by each DataLoader worker.",
    )
    max_audio_seconds_in_batch: float = Field(gt=0.0)
    max_text_tokens_in_batch: int = Field(ge=1)
    max_samples_per_batch: int | None = Field(default=None, ge=1)
    bucketing_pool_size: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def _validate_unique_source_names(self) -> "DataConfig":
        """校验数据源不重名 / Ensure source names are unique.

        ``mode="after"`` 表示在各字段已构造完成后整体校验。源名重复会让下游按名引用
        (如指定权重、统计、resume) 产生歧义，故这里直接拦截并报出所有重复名。

        Runs after all fields are built (``mode="after"``). Duplicate source names
        would make any name-based lookup downstream ambiguous, so we raise listing
        every offending name.
        """
        # 统计每个源名出现的次数 / count occurrences of each source name.
        counts: dict[str, int] = {}
        for source in self.sources:
            counts[source.name] = counts.get(source.name, 0) + 1
        # 挑出出现 >1 次的名字一并报错，便于一次性修全部重复项。
        # Collect every name seen more than once so the error lists them all.
        duplicated = [name for name, count in counts.items() if count > 1]
        if duplicated:
            raise ValueError(f"Source names must be unique: {duplicated}")
        return self


__all__ = [
    "DEFAULT_SOURCE_ADAPTER_CLASS_NAME",
    "DataConfig",
    "DataSourceConfig",
    "SourceAdapterConfig",
]
