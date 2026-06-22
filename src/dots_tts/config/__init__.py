"""配置包 / Configuration package.

中文：集中存放各类强类型配置定义，供训练/推理/服务读取。包含 base(配置基类与公共
字段)、data(数据流水线配置)、train(训练超参与优化器/调度配置)、app(应用/服务端配置)。
把配置与代码分离，便于复现实验、切换数据集与改超参而不动模型代码。

EN: Centralizes the strongly-typed config definitions consumed by training, inference and
serving — base (shared config classes/fields), data (data-pipeline config), train
(training hyperparameters, optimizer/scheduler) and app (application/service config).
"""
