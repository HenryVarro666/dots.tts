"""dots.tts 顶层包 / Top-level package for dots.tts.

中文：dots.tts 是一套 2B 规模的「连续潜在(continuous-latent)」自回归 TTS 系统——
不使用离散音频 token，而是用 Qwen2.5-1.5B 做自回归主干、flow-matching DiT 做声学头
(velocity field 预测)、BigVGAN AudioVAE 在连续 latent 空间编解码、CAM++ x-vector 提供
说话人/音色条件。本包是所有子模块的根命名空间，向下聚合：cli(命令行入口)、
runtime / runtime_double_streaming(推理运行时与流式)、config(配置)、data(数据流水线)、
models(模型族)、modules(网络组件:backbone/speaker/vocoder)、training(训练)、utils(工具)。

EN: dots.tts is a ~2B continuous-latent autoregressive TTS stack (no discrete audio
tokens): a Qwen2.5-1.5B autoregressive backbone, a flow-matching DiT acoustic head that
predicts the velocity field, a BigVGAN AudioVAE for encode/decode in continuous latent
space, and a CAM++ x-vector for speaker/timbre conditioning. This package is the root
namespace tying together the CLI, runtimes, config, data, models, modules, training and
utility subpackages.
"""
