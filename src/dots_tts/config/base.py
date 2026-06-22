"""配置体系的公共基类 / Shared base classes for the config system.

本文件定义 dots.tts 全部配置对象的两个 pydantic 基类:
- ``ConfigBase``     : 宽松基类 (``extra="allow"``),允许 YAML 里出现额外字段。
- ``StrictConfigBase``: 严格基类 (``extra="forbid"``),拼错字段名会直接报错。

在数据流里的位置 / Where this sits:
    YAML (configs/dots_tts.yaml)
        --> pydantic 校验/解析 (AppConfig / DataConfig / TrainConfig / LossConfig ...)
        --> 训练与推理代码读取 config.train.* / config.train_data.* 等
所有具体配置类 (见同目录 app.py / data.py / train.py,以及 models/.../config.py)
都继承自这里的基类,从而共享统一的校验策略与序列化辅助方法。

This file does NOT describe a TTS algorithm itself; it只是配置层的地基 /
it is purely the configuration-layer foundation that every concrete config inherits.

关键类 / Key classes:
- ``ConfigBase``       : 带 ``get`` / ``to_dict`` / ``to_declared_dict`` 等辅助方法的宽松基类。
- ``StrictConfigBase`` : 仅把 ``extra`` 改为 ``forbid`` 的严格变体。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    """宽松配置基类 / Permissive base config (pydantic ``BaseModel``).

    职责 / Responsibility:
        作为所有配置对象的根基类,提供统一的 pydantic 行为 + 一组取值/序列化辅助方法。

    model_config 三项行为 / Three configured behaviors:
        - ``extra="allow"``            : 允许 YAML 里出现模型未声明的额外键(向前兼容、
          实验性字段不报错)。严格校验交给子类 :class:`StrictConfigBase`。
        - ``validate_assignment=True`` : 实例创建后再赋值也会重新跑校验(防止运行时写入非法值)。
        - ``arbitrary_types_allowed=True``: 允许字段类型为非 pydantic 原生类型(例如自定义对象),
          这样配置里可以挂载任意 Python 类型而不必为其写 validator。
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def get(self, key: str, default=None):
        """像 ``dict.get`` 一样按名取字段 / dict-style field access with a default.

        语义细节 / Subtlety:
            普通 ``getattr`` 在字段值恰好是 ``None`` 时无法区分"用户没设"和"用户显式设成 None"。
            这里用 ``model_fields_set``(pydantic 记录的"被显式赋过值的字段集合")来区分:
            若某字段值为 ``None`` 且从未被显式设置过,则视为未提供,返回 ``default``。

        参数 / Args:
            key: 字段名。default: 取不到时的回退值(默认为 ``None``)。
        返回 / Returns:
            字段值,或在未设置时返回 ``default``。
        """
        value = getattr(self, key, default)
        if value is default:
            # 字段根本不存在(getattr 已返回 default),直接回传,避免下面再访问 model_fields_set。
            return value

        fields_set = self.model_fields_set
        # 值为 None 且该字段从未被显式赋值 -> 当作"未提供",回退到 default。
        if value is None and key not in fields_set:
            return default
        return value

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict,丢弃所有值为 None 的键 / Dump to dict, dropping None values.

        ``exclude_none=True`` 让输出只包含真正有值的字段,常用于日志打印或写回 YAML。
        注意:这里会包含 ``extra="allow"`` 收进来的额外字段(对比 :meth:`to_declared_dict`)。
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def _declared_field_names(cls) -> list[str]:
        """返回"显式声明过的"字段名 / Names of fields declared on the model.

        只取在类上正式声明的字段(``model_fields``),并排除 pydantic 配置项 ``model_config``;
        因此 ``extra="allow"`` 收进来的额外键不会出现在这里 —— 这是 :meth:`to_declared_dict`
        与 :meth:`to_dict` 的关键区别(后者会带上额外字段)。
        """
        return [name for name in cls.model_fields if name != "model_config"]

    @classmethod
    def _serialize_declared_value(cls, value):
        """递归地把字段值序列化成纯 dict/list/标量 / Recursively serialize one value.

        对嵌套的 :class:`ConfigBase`(如 ``DataConfig`` 内的 ``DataSourceConfig`` 列表)继续
        调用 ``to_declared_dict``,从而整棵配置树都只保留"声明过的"字段;list/tuple/dict 逐元素递归
        (tuple 会被归一化成 list,便于写回 YAML/JSON),其余标量原样返回。
        """
        if isinstance(value, ConfigBase):
            return value.to_declared_dict()
        if isinstance(value, list):
            return [cls._serialize_declared_value(item) for item in value]
        if isinstance(value, tuple):
            # tuple 序列化为 list:YAML/JSON 没有 tuple 概念,统一成 list 更通用。
            return [cls._serialize_declared_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: cls._serialize_declared_value(item) for key, item in value.items()
            }
        return value

    def to_declared_dict(self) -> dict[str, Any]:
        """仅导出"声明字段"的 dict(跳过 None 与额外字段)/ Dump only declared fields.

        与 :meth:`to_dict` 的区别 / Contrast with ``to_dict``:
            ``to_dict`` 用 pydantic 的 ``model_dump`` 会带上 ``extra="allow"`` 吸收的额外键;
            而本方法只遍历 :meth:`_declared_field_names`,逐字段递归序列化,产出一份"干净的、
            只含模型正式 schema"的配置快照 —— 适合用来回写规范化的 YAML / 做配置对比。
            同样跳过值为 ``None`` 的字段。
        """
        data = {}
        for name in self._declared_field_names():
            value = getattr(self, name, None)
            if value is None:
                # 未设置/显式 None 的字段不写入,保持输出精简。
                continue
            data[name] = self._serialize_declared_value(value)
        return data


class StrictConfigBase(ConfigBase):
    """严格配置基类 / Strict base config (forbids unknown fields).

    与父类 :class:`ConfigBase` 唯一不同:把 ``extra`` 从 ``"allow"`` 改成 ``"forbid"``,
    即 YAML 里出现未声明的键会直接抛 ``ValidationError``。用于真正面向用户、需要"拼错即报错"
    的配置(如 ``AppConfig`` / ``DataConfig`` / ``TrainConfig``),避免配置项打错却被静默忽略。
    其余辅助方法(``get`` / ``to_dict`` / ``to_declared_dict`` 等)全部继承自父类。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


__all__ = ["ConfigBase", "StrictConfigBase"]
