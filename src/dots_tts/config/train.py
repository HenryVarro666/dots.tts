"""训练超参配置 / Training hyper-parameter config.

本文件定义 :class:`TrainConfig` —— dots.tts 训练循环(trainer)读取的那一组训练超参
(优化器、调度、checkpoint、评估、CFG 条件丢弃率等)。它是 ``configs/*.yaml`` 里
``train:`` 一节的强类型 schema：YAML 经 :class:`AppConfig.from_yaml`
(见 ``config/app.py``)解析后，``train`` 字段就被校验/反序列化成本类的一个实例。

This file declares :class:`TrainConfig`, the strongly-typed schema for the ``train:``
section of the project's YAML configs. The training loop consumes these fields to set
up the optimizer / LR schedule / checkpointing / periodic eval, and the two
classifier-free guidance (CFG) condition-drop rates used during training.

数据流位置 / Where it sits:
    ``configs/dots_tts.yaml`` --(AppConfig.from_yaml)--> ``AppConfig.train: TrainConfig``
    --> trainer 据此构建 optimizer / scheduler / 训练步数 / 存档与评估节奏。

关键类 / Key class:
    - ``TrainConfig``: 全部训练超参，继承 ``StrictConfigBase`` (extra="forbid"，
      多写未知字段会报错，避免拼写错误的配置被静默忽略)。
"""

from __future__ import annotations

from pydantic import Field

from dots_tts.config.base import StrictConfigBase


class TrainConfig(StrictConfigBase):
    """训练超参的强类型集合 / Strongly-typed bundle of training hyper-parameters.

    继承 :class:`StrictConfigBase` (``extra="forbid"``)，所以 YAML 里出现未声明的键
    会直接校验失败 —— 对训练脚本而言这是有意为之的"拼写保护"。无默认值的字段
    (如 ``learning_rate`` / ``max_train_steps``)是**必填项**，必须在配置里给出。

    Inherits :class:`StrictConfigBase` (``extra="forbid"``), so any unknown key in the
    YAML fails validation. Fields without a default are required and must be supplied
    by the config file.

    字段语义 / Field semantics:
        - ``pretrained_model_path``: 训练起点的预训练权重路径(continue/finetune 的基座)。
        - ``output_dir``: 输出根目录，存放 checkpoint / 日志 / 评估产物。
        - ``seed``: 随机种子，保证数据打散、dropout、CFG 丢弃等的可复现性。
        - ``learning_rate`` (必填): 优化器学习率(peak LR，配合 warmup 使用)。
        - ``cfg_droprate``: 训练时整条丢弃 LLM hidden(文本/自回归)条件的概率，用于
          学习 classifier-free guidance (CFG) 的"无条件"分支；推理时再用 guidance
          scale 在 cond/uncond 间外推。默认 0.0 表示此 schema 默认不丢(注意：模型侧
          ``ModelConfig`` 默认是 0.2，最终以本训练配置覆盖为准)。
        - ``xvec_drop_rate``: 训练时丢弃说话人 x-vector(音色)条件的概率，是对音色条件
          单独做的 CFG cond-drop。
        - ``weight_decay``: AdamW 权重衰减(L2 正则)。
        - ``warmup_steps``: 学习率线性 warmup 的步数(从 0 升到 peak LR)。
        - ``max_train_steps`` (必填): 训练总步数(以 optimizer step 计，决定何时停)。
        - ``gradient_accumulation_steps`` (``ge=1``): 梯度累积步数，用小显存模拟更大
          的有效 batch；有效 batch = micro-batch × 此值。
        - ``grad_clip_norm``: 梯度全局范数裁剪阈值(gradient clipping)，稳定训练。
        - ``save_interval`` (``ge=1``): 每多少 step 存一次 checkpoint。
        - ``max_checkpoints_to_keep``: 最多保留多少个 checkpoint，超出则滚动删旧的。
        - ``log_interval`` (``ge=1``): 每多少 step 打一次训练日志(loss 等)。
        - ``eval_interval`` (``None`` 或 ``ge=1``): 每多少 step 跑一次验证；``None`` 表示
          不做周期性评估。
        - ``max_eval_batches``: 单次评估最多取多少个 batch(``None`` = 跑完整个验证集)。
        - ``run_eval_on_start``: 训练开局(step 0)是否先跑一次评估，便于拿到基线指标。
    """

    pretrained_model_path: str
    output_dir: str
    seed: int = 42
    learning_rate: float
    cfg_droprate: float = 0.0  # 丢弃 LLM hidden 条件的概率 (CFG 无条件分支) / cond-drop prob for CFG
    xvec_drop_rate: float = 0.5  # 丢弃说话人 x-vector 音色条件的概率 (对音色的 CFG)
    weight_decay: float = 0.01
    warmup_steps: int = 0  # LR 线性 warmup 步数 (0 = 不预热)
    max_train_steps: int
    gradient_accumulation_steps: int = Field(default=1, ge=1)  # 梯度累积：有效 batch ×= 此值
    grad_clip_norm: float = 1.0  # 梯度范数裁剪阈值 / gradient-norm clip
    save_interval: int = Field(default=1000, ge=1)
    max_checkpoints_to_keep: int = 10  # 滚动保留的 checkpoint 上限，超出删最旧
    log_interval: int = Field(default=10, ge=1)
    eval_interval: int | None = Field(default=None, ge=1)  # None = 关闭周期性验证
    max_eval_batches: int | None = None  # None = 评估时跑完整个验证集
    run_eval_on_start: bool = False  # step 0 先评估一次以拿基线 / eval before training begins


__all__ = ["TrainConfig"]
