"""主干网络组件 / Backbone modules.

中文：构成声学/语义建模主干的可复用网络组件。dit(flow-matching 声学头用的 Diffusion
Transformer，含 adaLN 时间/条件注入，预测 velocity field)、layers(通用层：attention、
RMSNorm、旋转位置编码等基础积木)、semantic_encoder(把文本/语义条件编码成供主干消费的
表示)。这些是「零件」，由 models 层组装成完整模型。

EN: Reusable network components forming the acoustic/semantic backbone — dit (the
Diffusion Transformer used by the flow-matching acoustic head, with adaLN time/condition
injection, predicting the velocity field), layers (generic building blocks: attention,
RMSNorm, rotary position embeddings, ...) and semantic_encoder (encodes text/semantic
conditioning into representations the backbone consumes).
"""
