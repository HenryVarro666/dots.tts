"""样本处理 pipeline 的抽象基类 / 接口定义。

本文件 (What this file does)
---------------------------
定义训练数据预处理流水线的统一接口 ``BaseSamplePipeline``。它规定了"把一条
原始样本(raw sample)转换成一条已处理样本(processed sample)"的**契约**:
1:1 映射(一进一出,不丢样本、不合并样本),并且**强制透传** adapter 写入的
断点续训(resume)元数据 ``_adapter_state``,使数据加载可在中断后从精确位置恢复。

在数据流里的位置 (Position in the data pipeline)
-----------------------------------------------
数据源(adapter)产出原始 dict 样本 → 本基类的子类(如
``tts_pipeline.BasicTtsPipeline`` / ``InterleaveTtsPipeline``)在 worker 进程里
逐条做重采样、静音归一化、对齐补零、tokenize、提取 speaker fbank 等转换 → 下游
collate / DataLoader 组 batch 喂给模型训练。本文件本身**不做**任何音频/文本运算,
只提供框架与不变式(invariant)校验;具体转换由子类实现的 ``process_sample`` 完成。

关键类/函数清单 (Key classes / functions)
-----------------------------------------
- ``BaseSamplePipeline``        : pipeline 抽象基类(ABC),定义接口与不变式。
  - ``process_sample``          : 抽象方法,子类必须实现的单样本转换逻辑。
  - ``_validate_input_sample``  : 校验 ``_adapter_state`` 存在(resume 不变式)。
  - ``__call__``                : 把单样本转换包装成一个流式 generator(可迭代)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator


class BaseSamplePipeline(ABC):
    """1:1 样本处理 pipeline 基类,负责保留 adapter 的断点续训元数据。

    职责 (Responsibility)
    --------------------
    把"逐条转换样本"这件事抽象成统一接口:子类只需实现 ``process_sample``
    (纯转换逻辑),由本基类负责输入/输出校验、原始字段透传与流式迭代。所谓
    **1:1** 指的是输入一条样本恰好产出一条样本(既不过滤也不展开),这是断点
    续训(resume)能精确对齐进度的前提。

    断点续训不变式 (Resume invariant)
    --------------------------------
    每条样本都必须携带 adapter 注入的 ``_adapter_state``(记录该样本在数据源里
    的位置/游标)。本基类在转换前后各校验一次,保证 ``process_sample`` 不会把这
    个续训锚点弄丢——否则中断后无法从正确位置恢复。

    对应 ML 概念 (ML concept)
    ------------------------
    这是 streaming/iterable 数据集的 per-sample transform 接口,类似 PyTorch
    ``IterableDataset`` 的 map 风格预处理,但额外强约束了 resume 元数据的透传。
    """

    @staticmethod
    def _validate_input_sample(sample: dict) -> None:
        """校验样本携带断点续训(resume)所需的 ``_adapter_state``。

        参数 (Args):
            sample: 待校验的样本 dict。
        异常 (Raises):
            RuntimeError: 缺失 ``_adapter_state`` 时抛出——该字段是续训锚点,
                丢失意味着无法从中断处恢复进度。

        说明:在 ``__call__`` 中对输入与最终输出各调用一次,作为不变式守卫
        (invariant guard),确保转换前后续训元数据始终在场。
        """
        if "_adapter_state" not in sample:
            raise RuntimeError(
                "Source sample is missing required '_adapter_state' for resume."
            )

    @abstractmethod
    def process_sample(self, sample: dict) -> dict:
        """把一条原始样本转换成一条已处理样本(子类必须实现)。

        参数 (Args):
            sample: 单条原始样本 dict(基类已对其做过浅拷贝,见 ``__call__``)。
        返回 (Returns):
            dict: 仅包含**新增/修改**字段即可——``__call__`` 会把它 ``update``
                回原始样本之上,故无需重复回填 ``_adapter_state`` 等透传字段。

        约定:实现应保持 1:1 语义(一进一出),不要在此过滤或展开样本;具体音频/
        文本转换(重采样、tokenize、提取 x-vector 用的 fbank 等)由子类落地。
        """

    def __call__(self, samples: Iterable[dict]) -> Iterator[dict]:
        """把单样本转换包装为流式 generator:逐条产出已处理样本。

        参数 (Args):
            samples: 上游(adapter)产出的原始样本可迭代对象。
        产出 (Yields):
            dict: 已处理样本——为原始样本字段叠加 ``process_sample`` 结果。

        设计要点:全程惰性(lazy)迭代,适配 streaming 数据集;对每条样本做
        输入校验 → 转换 → 合并 → 输出校验,保证 resume 元数据贯穿首尾。
        """
        for raw_sample in samples:
            self._validate_input_sample(raw_sample)
            # 传浅拷贝给子类,隔离 in-place 改动,避免污染下面用于 update 的原始样本。
            processed = self.process_sample(dict(raw_sample))
            if not isinstance(processed, dict):
                raise RuntimeError(
                    f"{self.__class__.__name__}.process_sample() must return a dict."
                )
            # 以原始样本为底(含 _adapter_state 等透传字段),叠加子类新算出的字段:
            # processed 中的同名 key 覆盖原值,未涉及的原始字段(尤其续训锚点)保留。
            item = dict(raw_sample)
            item.update(processed)
            self._validate_input_sample(item)  # 二次校验:确认合并后续训锚点仍在。
            yield item
