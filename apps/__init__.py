"""dots.tts 应用入口包 / Application entrypoints for dots.tts.

中文：面向最终用户的应用层入口(与核心库 src/dots_tts 分离)。当前包含 gradio 子包：
一个基于 Gradio 的交互式 Web Demo，调用核心推理运行时来做文本合成与音色克隆。
本层只做「界面与服务编排」，模型逻辑仍在核心库中。

EN: User-facing application entrypoints, kept separate from the core library
(src/dots_tts). Currently contains the gradio subpackage — an interactive Gradio web
demo that calls the core inference runtime for synthesis and voice cloning. This layer
only handles UI and service orchestration; model logic stays in the core library.
"""
