"""顶层应用配置 / Top-level application (run) configuration.

本文件定义整个 dots.tts 训练/评测一次运行 (run) 的"总配置入口" :class:`AppConfig`,
并提供从 YAML 文件加载它的两个等价入口 (:meth:`AppConfig.from_yaml` 与
:func:`load_config`)。它本身不含任何 TTS 算法逻辑,只是把四块子配置聚合 (aggregate)
成一个经 pydantic 校验过的强类型对象,供下游训练脚本统一读取。

This module defines :class:`AppConfig`, the single typed entry point that aggregates
every sub-config needed for one dots.tts run, plus two equivalent loaders that
deserialize it from a YAML file. It contains no model math — it is purely the
top-of-tree config object that downstream training/eval code reads from.

四块子配置 / The four aggregated sub-configs:
    - train_data : :class:`DataConfig`  — 训练数据管线 (DataLoader / 数据源 / 分桶预算)。
    - val_data   : :class:`DataConfig` | None — 可选的验证集数据管线 (无验证时为 None)。
    - loss       : :class:`LossConfig`  — 三项损失 (AR 交叉熵 / flow-matching / EOS) 权重。
    - train      : :class:`TrainConfig` — 优化器/学习率/步数/checkpoint 等训练超参。

在数据流里的位置 / Where this sits:
    configs/dots_tts.yaml
        --> load_config() / AppConfig.from_yaml()  (本文件: 读盘 + pydantic 校验)
        --> AppConfig 实例 (train_data / val_data / loss / train 四个字段)
        --> 训练脚本读取 cfg.train.*, cfg.train_data.*, cfg.loss.* 驱动整个 run

关键符号 / Key symbols:
    - DEFAULT_CONFIG_PATH : 默认 YAML 配置路径 (相对工作目录)。
    - AppConfig           : 聚合四块子配置的严格根配置类 (extra="forbid")。
    - load_config         : 函数式便捷入口,内部委托给 ``AppConfig.from_yaml``。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dots_tts.config.base import StrictConfigBase
from dots_tts.config.data import DataConfig
from dots_tts.config.train import TrainConfig
from dots_tts.models.dots_tts.config import LossConfig

DEFAULT_CONFIG_PATH = "configs/dots_tts.yaml"


class AppConfig(StrictConfigBase):
    """一次运行的根配置 / Root config for a single training/eval run.

    职责 / Responsibility:
        把 YAML 里的四块子配置 (数据/可选验证数据/损失权重/训练超参) 聚合成一个
        强类型对象。继承 :class:`StrictConfigBase` (``extra="forbid"``),因此 YAML 顶层
        出现任何拼错或多余的键都会在加载时直接报错,而不是被静默忽略 —— 让配置错误
        "尽早大声失败"。

    Aggregates the four sub-configs (data / optional val-data / loss weights / training
    hyper-params) into one validated, typed object. Because it subclasses
    :class:`StrictConfigBase` (``extra="forbid"``), any misspelled or stray top-level
    key fails loudly at load time instead of being silently dropped.

    字段 / Fields:
        train_data: 训练数据管线配置 (必填)。
        val_data:   验证数据管线配置;为 None 表示本次 run 不做验证。
        loss:       损失各项权重。
        train:      训练超参 (优化器、步数、checkpoint 等)。
    """

    train_data: DataConfig
    val_data: DataConfig | None = None
    loss: LossConfig
    train: TrainConfig

    @classmethod
    def from_yaml(cls, config_path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
        """从 YAML 文件加载并校验配置 / Load and validate config from a YAML file.

        参数 / Args:
            config_path: YAML 配置文件路径;缺省用 :data:`DEFAULT_CONFIG_PATH`。

        返回 / Returns:
            经 pydantic 校验后的 :class:`AppConfig` 实例。

        说明 / Notes:
            显式以 ``encoding="utf-8"`` 打开,避免不同平台默认编码导致中文/非 ASCII
            字段读乱码;用 ``yaml.safe_load`` 而非 ``load`` 以拒绝任意 Python 对象构造
            (安全考量)。``model_validate`` 会按字段类型递归构造各子配置并触发其校验器。
        """
        with Path(config_path).open(encoding="utf-8") as fin:
            raw_config = yaml.safe_load(fin)
        return cls.model_validate(raw_config)


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """加载配置的函数式便捷入口 / Functional shortcut to load the run config.

    与 :meth:`AppConfig.from_yaml` 完全等价,只是提供一个模块级函数风格的调用方式
    (``load_config(path)`` 而非 ``AppConfig.from_yaml(path)``),方便脚本直接 import 使用。

    Thin functional wrapper around :meth:`AppConfig.from_yaml`; behaves identically and
    exists only to offer a module-level ``load_config(path)`` call style for scripts.
    """
    return AppConfig.from_yaml(config_path)


__all__ = ["AppConfig", "DEFAULT_CONFIG_PATH", "load_config"]
