"""训练包 / Training package.

中文：模型训练所需的支撑设施。losses(损失函数：flow-matching 的 velocity 回归损失，
以及可能的 MeanFlow 蒸馏损失等)、checkpoint(模型/优化器状态的保存与加载、断点续训)、
utils(训练侧通用工具：分布式、随机种子、日志/计时等辅助)。

EN: Supporting infrastructure for model training — losses (loss functions: the
flow-matching velocity-regression loss and, e.g., MeanFlow distillation losses),
checkpoint (saving/loading model & optimizer state, resume-from-checkpoint) and utils
(training-side helpers: distributed setup, seeding, logging/timing).
"""
