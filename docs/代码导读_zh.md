# dots.tts 代码导读（中文学习版）

> 配合**全库中英双语注释**一起读：源码里每个文件都已加了模块/类/函数 docstring + 关键行内注释。本文件是"地图"，告诉你**先读什么、数据怎么流、核心概念是什么、想改从哪下手**。
> 行号为当前注释版的位置（`file:line`），后续你改代码后可能漂移——以**类/函数名**为准。
> 架构层背景见知识库 `../../05-dots.tts-技术详解.md`。

---

## 0. 一分钟总览

dots.tts = **2B 全连续潜在（continuous-latent）自回归 TTS**。和 Higgs v3 等"离散 audio token"路线最大的不同：

| | dots.tts | 典型离散 token TTS（如 Higgs v3） |
|---|---|---|
| 音频表示 | **连续 latent**（AudioVAE 编码，无 codebook） | 离散 token（codec codebook） |
| 声学建模 | **flow-matching DiT**（在 latent 空间预测 velocity field，ODE 积分） | 自回归预测离散 token |
| 主干 | Qwen2.5-1.5B 自回归，输出**条件 hidden** | 自回归 LM 直接出 token |
| 采样率 | 48 kHz | 多为 24 kHz |
| 提速 | **MeanFlow 蒸馏**（4 步 NFE） | KV cache / 投机解码等 |

**两段式生成**：①Qwen2 主干自回归地产出每个 audio patch 的**条件 hidden**；②flow-matching DiT 以该 hidden（+ 说话人 x-vector）为条件，把高斯噪声**积分**成连续 latent；③BigVGAN AudioVAE 把 latent 解码成 48 kHz 波形。

---

## 1. 目录结构与模块职责

```
src/dots_tts/
├── cli.py                     推理 CLI 入口（parse_args / main）
├── runtime.py                 推理编排 DotsTtsRuntime（加载/预处理/generate）
├── runtime_double_streaming.py  double-streaming 增量会话（逐 text token 提前出音）
├── models/dots_tts/
│   ├── config.py              模型超参 dataclass（latent_dim/patch_size/CFG/DiT…）
│   ├── model.py   ★难点       顶层 DotsTtsModel：组装 + 推理状态机（prefill/decode 循环）
│   └── core.py    ★最难       神经核心 DotsTtsCore：LLM+FM 前向、ODE 步、mask、IO 装配
├── modules/
│   ├── backbone/
│   │   ├── dit.py             flow-matching DiT（velocity 预测，adaLN 条件注入）
│   │   ├── layers.py          通用层：因果 Conv、Mlp、Attention、RoPE
│   │   └── semantic_encoder.py  VAE latent 的因果 transformer 编码器（含流式 cache）
│   ├── vocoder/
│   │   ├── bigvgan.py  ★难点  AudioVAE 连续潜在声码器 + 流式 SLSTM/窗口
│   │   └── alias_free_*.py    防混叠（anti-aliasing）激活/滤波/重采样
│   └── speaker/
│       ├── campplus.py / campplus_layers.py  CAM++ x-vector 说话人编码
│       ├── encoder.py         SpeakerXVectorFeatures：fbank+CAM++ 封装
│       └── fbank.py           mel/fbank 特征
├── data/                      数据管线：builders / collator / batchers / streaming
│   ├── pipelines/             tts_pipeline（text→audio→fbank→token）/ tokenizing / 预处理
│   └── source_adapters/       jsonl manifest / 多源(加权) adapter
├── training/                  losses / checkpoint(含 RNG resume) / utils
├── config/                    app/base/data/train 配置
└── utils/                     tokenizer(特殊 token) / text(语言检测) / audio / profiling / …
scripts/                       train_dots_tts(.py) / train_dots_tts_meanflow(.py) / 示例
apps/gradio/                   Gradio demo（app/service/languages/constants）
configs/                       dots_tts.yaml / dots_tts_meanflow.yaml（已加注释）
```

---

## 2. 推理数据流（text → 48 kHz wav）

调用链（`file:line`，以函数名为准）：

```
cli.py:main (154)
  └─ runtime.py: DotsTtsRuntime.from_pretrained (193)      # 加载权重(snapshot_download)+建模型+设备/精度
  └─ runtime.py: DotsTtsRuntime.generate (692)             # 一次性合成（流式版 generate_stream:577）
       ├─ _prepare_inputs (465)                            # 文本规整/语言标签/prompt 音频加载&重采样/生成调度 schedule
       └─ model.py: DotsTtsModel.generate_audio (2440)     # ★推理主状态机
            ├─ _prefill (1860) + _prefill_prompt_latents (1465)   # 处理 prompt：发现 patch 边界、灌入条件
            ├─ _consume_text_schedule (1748)               # 喂文本 token、推进 Qwen2 主干
            ├─ _decode_next_audio (1929)                   # ★每个 audio patch：分派 flow_matching / meanflow 求解器
            │     └─ core.py: DotsTtsCore.fm_solver_step (441)    # ★单 ODE 步 + CFG
            │           ├─ step_llm (530)                  # Qwen2 自回归一步，吐条件 hidden（KV cache）
            │           ├─ _flow_matching_step_fm (676)    # 标准：torchdiffeq.odeint 积分若干步
            │           ├─ _meanflow_step_fm (577)         # 蒸馏：极少步(默认 4)
            │           └─ dit.py: DiT.forward (257)       # 预测 velocity field v_t（adaLN 注入 t + 说话人）
            └─ _consume_audio_patch (2025)                 # patch 状态机：latent 入历史、推进
       └─ bigvgan.py: AudioVAE.decode (1016)               # 连续 latent → 48 kHz 波形（流式走 SLSTM/窗口）
  └─ soundfile.write                                       # 落盘 WAV
```

**说话人条件（zero-shot 克隆）**：prompt 音频 → `speaker/encoder.py: SpeakerXVectorFeatures`（fbank → `campplus.py: CAMPPlus`，121）→ x-vector，作为 DiT 的条件，`--speaker-scale` 控制其强度。

**关键参数对应**（`cli.py` 注释里逐个有）：`--num-steps`=ODE 步数（越多越稳越慢）、`--guidance-scale`=CFG 强度、`--speaker-scale`=音色强度、`--ode-method`=solver（euler/midpoint…）、`--template-name`=tts/instruction_tts/text_to_audio/tts_interleave。

---

## 3. 训练数据流 + MeanFlow 蒸馏

**普通训练**（`scripts/train_dots_tts.py: DotsTtsTrainingRun` 124，`main` 967）：

```
数据源 → data/source_adapters/* → data/pipelines/tts_pipeline（text→audio波形→fbank→token，模板拼装）
       → data/collator.py: PadCollator (39)（按长度 pad，出 input_ids/labels/audio/fbank）
       → core.py: DotsTtsCore.forward (220)
            ├─ 自回归 LLM 前向（把 audio span token 替换成 patch embedding）
            ├─ IOHelper (994) 装配 DiT 的输入（span/latent 交错、加噪）
            ├─ DiT 预测 velocity → flow-matching 目标 u_t = x1−(1−σ)·x0
            └─ 三套监督：LLM CE loss + flow-matching loss + EOS loss
       → training/losses.py: LossTerm (52) / collapse_loss_terms (164)（逐元素+mask→标量、分布式聚合）
       → AdamW + cosine warmup（accelerate DDP / 梯度累积）
```

**MeanFlow 蒸馏**（`scripts/train_dots_tts_meanflow.py`：`MeanFlowSettings` 80、`MeanFlowDotsTtsModel` 187、`DotsTtsMeanFlowTrainingRun` 901、`main` 1170）：
- 思想：teacher（多步 flow-matching）生成 ODE 轨迹，student 学会**用极少步（NFE=4）**一跳到位 → 推理提速。
- `core.py` 里 `_meanflow_step_fm` 与 DiT 的 duration/mean 模式即为蒸馏后的推理路径；产物对应 HF 权重 `dots.tts-mf`。

---

## 4. 核心 ML 概念入门（中英对照）

| 概念 | 一句话 | 代码位置 |
|---|---|---|
| **Continuous-latent AudioVAE** | 用冻结 VAE 把音频压成**连续** latent（不是离散 token），声学建模在 latent 空间做 | `bigvgan.py: AudioVAE` (1016) |
| **Flow matching** | 学一个 velocity field v_t，把噪声 x0 沿直线"流"到数据 x1；目标 u_t=x1−(1−σ)x0 | `core.py: FlowMatchingHelper` (809) |
| **ODE solver** | 推理时用 euler/midpoint 等数值积分 v_t，从噪声解出 latent | `core.py: _flow_matching_step_fm` (676)（torchdiffeq） |
| **DiT + adaLN** | Diffusion Transformer 预测 velocity；用 adaptive LayerNorm 把"时间 t + 说话人"调制进每个 block | `dit.py: DiT/DiTBlock` (257/155) |
| **Qwen2 自回归主干** | 文本+音频统一成序列自回归，吐每个 audio patch 的**条件 hidden** | `core.py: step_llm` (530) |
| **CAM++ x-vector** | 从 prompt 音频抽说话人 embedding，实现 zero-shot 音色克隆 | `campplus.py: CAMPPlus` (121) |
| **Classifier-free guidance (CFG)** | 训练随机丢条件、推理 batch 翻倍取 `vt = vt_c + scale·(vt_c − vt_u)` 外推，增强可控性 | `core.py: fm_solver_step` (441) |
| **MeanFlow 蒸馏** | 学生用极少步逼近教师 ODE 轨迹，换推理速度 | `scripts/train_dots_tts_meanflow.py` |
| **Causal / block-diagonal mask** | 文本因果、每个 audio patch 内块对角、patch 对文本部分可见——决定信息流向 | `core.py: CausalHelper` (865) |
| **Double-streaming** | 逐 text token 喂入、边收文本边出音，低延迟 | `runtime_double_streaming.py` |
| **SCA（self-corrective alignment）** | `soar` 权重用的后训练对齐，提升克隆/稳健性 | 见 `05` 文档 §训练 |

---

## 5. 推荐阅读顺序（新手向）

1. **跑通心智模型**：`cli.py` → `runtime.py`（`from_pretrained` / `_prepare_inputs` / `generate`）——看清"参数怎么进、流程怎么走"。
2. **配置即词典**：`models/dots_tts/config.py`——所有超参语义集中在这。
3. **推理主状态机**：`model.py` 的 `generate_audio` → `_prefill` → `_decode_next_audio` → `_consume_audio_patch`（先看 docstring，再追细节）。
4. **神经核心**：`core.py` 的 `forward`（训练）与 `fm_solver_step`（推理）；再啃 `FlowMatchingHelper` / `CausalHelper` / `IOHelper` 三个 helper（最硬，注释最密）。
5. **声学头与声码器**：`dit.py`（velocity 预测）→ `bigvgan.py`（latent→波形 + 流式 SLSTM）。
6. **说话人条件**：`speaker/encoder.py` + `campplus.py`。
7. **数据与训练**：`data/pipelines/tts_pipeline.py` + `collator.py` → `training/losses.py` → `scripts/train_dots_tts.py`。
8. **进阶**：`train_dots_tts_meanflow.py`（蒸馏）、`runtime_double_streaming.py`（流式）。

---

## 6. 改进切入点（想改哪看哪）

| 想做的改进 | 从哪下手 |
|---|---|
| 调采样质量/速度折中 | `--num-steps` / `--guidance-scale` / `--ode-method`（`cli.py`），求解器在 `core.py:_flow_matching_step_fm/_meanflow_step_fm` |
| 换/调 backbone（如换更大 Qwen 或别的 LM） | `core.py: DotsTtsCore.__init__`（构建 LLM 处）+ `config.py` |
| 改 DiT 容量/条件注入方式 | `dit.py: DiTBlock`（adaLN）、`config.py:_DiTConfig` |
| 换声码器 / 提采样率 | `modules/vocoder/bigvgan.py: AudioVAE`，注意 latent_dim/hop 对齐 |
| 增强音色克隆 | `speaker/encoder.py` + `campplus.py`，及 `--speaker-scale`、CFG 的 `xvec_drop_rate` |
| 加语言 / 改文本前端 | `utils/text.py`（检测/标签）、`data/pipelines/tts_pipeline.py`（模板） |
| 更激进蒸馏（更少步） | `scripts/train_dots_tts_meanflow.py` + `MeanFlowSettings`(80) |
| 改训练损失/权重 | `training/losses.py`、`config.py:LossConfig`、`configs/*.yaml` |
| 低延迟流式 | `runtime_double_streaming.py` + `bigvgan.py` 的流式状态 |

---

## 7. 交叉引用
- 架构/benchmark/与 Higgs 对比：`../../05-dots.tts-技术详解.md`
- "这个仓库有没有核心代码"的逐文件核对：`../../参考代码/核心代码分析.md`
- 本地调用封装脚本：`../../scripts/light-models/run_dots_tts.py`

> 注释与本导读基于注释版核对（2026‑06）。源码版权归 dots.tts Team / RedNote，Apache‑2.0；本导读为学习用途整理。
