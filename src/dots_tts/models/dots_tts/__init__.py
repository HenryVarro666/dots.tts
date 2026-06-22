"""dots_tts 主模型包 / dots_tts model package.

中文：dots.tts 核心模型的定义所在。config(该模型的结构/超参配置)、core(核心前向逻辑：
自回归主干与 flow-matching DiT 声学头的耦合、CFG 分支、velocity field 预测、ODE 采样
等关键计算)、model(对外的模型类，封装训练 forward 与推理 generate 接口)。

EN: Defines the core dots.tts model — config (architecture/hyperparameter config),
core (the central forward logic: coupling the AR backbone with the flow-matching DiT
acoustic head, the CFG branch, velocity-field prediction and ODE sampling) and model
(the public model class wrapping the training-forward and inference-generate interfaces).
"""
