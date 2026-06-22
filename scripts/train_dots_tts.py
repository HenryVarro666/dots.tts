#!/usr/bin/env python3

"""dots.tts 训练入口 / Accelerate training entrypoint for dots.tts.

本文件做什么 / What this file does
==================================
这是 **训练主驱动**：用 HuggingFace ``accelerate`` 把 dots.tts (Qwen2.5-1.5B 自回归主干
+ flow-matching DiT 声学头 + BigVGAN AudioVAE + CAM++ x-vector) 端到端微调起来。它负责
分布式 (DDP)/梯度累积 (gradient accumulation)、模型与优化器加载、AdamW + cosine warmup
学习率调度、训练前的契约校验 (采样率 / audio token 契约 / 模型规模)、训练主循环、周期性
validation 与 checkpoint，以及可选的 debug batch 打印。

它在数据流里的位置 / Where it sits in the pipeline
--------------------------------------------------
``dots_tts.data.builders`` 产出 (流式) DataLoader → 本文件每步取一个 **梯度累积窗口** 的
micro-batch → ``DotsTtsModel.prepare_training_batch`` 在 CPU 侧补 loss mask 元数据 →
``model(batch)`` 前向得到三类损失项 (LM logits 的 cross-entropy、flow-matching velocity
field 的回归损失、EOS 判别损失) → 在 ``losses`` 里按全局归一化因子聚合成单个标量 →
``accelerator.backward`` 反传、AdamW 更新 → 周期性 validation / checkpoint。
注意：dots.tts 走 **continuous-latent** 路线，没有离散 audio token；这里的 "audio token"
指的是被切成 patch 的连续 latent 帧 (AudioVAE 输出)。

关键类/函数清单 / Key classes & functions
------------------------------------------
- ``_PreparedTrainingStep`` / ``_AccumulatedTrainingStep`` / ``_CompletedTrainingStep``:
  把一个 optimizer step 的三个阶段 (准备 micro-batch、前向反向累积、收尾约简) 的状态显式
  打包成只读 dataclass，便于在三段式流水线之间传递。
- ``DotsTtsTrainingRun``: 训练运行对象。``__init__`` 做加载与前置校验，``run`` 是主循环，
  ``_run_training_step`` / ``_prepare_training_step`` / ``_accumulate_training_step`` /
  ``_finalize_completed_training_step`` 组成三段式训练步，``_run_validation`` 做评估，
  ``_save_checkpoint`` / ``_resume_if_available`` 管理断点续训。
- ``parse_args`` / ``main``: CLI 入口。
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dots_tts.config import app as app_config
from dots_tts.data import builders as data_module
from dots_tts.models.dots_tts import model as dots_tts_model
from dots_tts.training import checkpoint as train_checkpoint
from dots_tts.training import losses as loss_ops
from dots_tts.training import utils as train_utils
from dots_tts.utils import util as util_module

# 连续多少个 epoch 都凑不出 "所有 rank 同时有 batch" 的窗口就报错 (防止 DDP 各 rank 数据
# 不同步导致 collective 永久卡死)。/ Max consecutive empty epochs before aborting.
_EMPTY_EPOCH_TOLERANCE = 32
# --debug 时最多打印几条训练 batch 的明细 (只在主进程)。/ Cap on debug batch dumps.
_DEBUG_BATCH_LIMIT = 3
# 训练最初几步无条件打印梯度调试信息 (此后只在 log_interval 命中时打)。/ Always dump grad
# debug for the first few steps regardless of log_interval.
_DEBUG_GRAD_EARLY_STEP_LIMIT = 3


# region Training Step State
@dataclass(slots=True)
class _PreparedTrainingStep:
    """一个梯度累积窗口的 "已准备" 状态 / Stage-1 output: one accumulation window.

    在动模型状态之前先把整窗口攒齐，并算好 **全局** 归一化分母，这样窗口内每个 micro-batch
    的损失能用同一套分母 (跨所有 rank、跨整个累积窗口) 做归一化，等价于真正的大 batch。

    字段 / Fields:
    - ``micro_batches``: 本窗口的 ``grad_accumulation_steps`` 个已准备 batch (含 loss mask)。
    - ``consumed_counts``: 跨 rank 求和后的 [总 token, audio token, text token] 计数，用于进度统计。
    - ``global_denominators``: 各损失项的全局归一化分母 (有效元素数)，跨 rank+跨窗口求和后的值。
    """

    micro_batches: list[dict]
    consumed_counts: list[int]
    global_denominators: dict[str, float]


@dataclass(slots=True)
class _AccumulatedTrainingStep:
    """前向/反向累积后的 "已累积" 状态 / Stage-2 output: forward+backward done.

    字段 / Fields:
    - ``loss_totals`` / ``loss_denominators``: 各损失项 rank-local 的分子(加权和) 与分母(计数)，
      留到收尾阶段再跨 rank 约简成可读的平均损失。
    - ``source_loss_totals`` / ``source_loss_denominators``: 同上但按数据 source 分组 (多数据源
      混训时分别看每个 source 的损失)。
    - ``completed_optimizer_step``: 本步是否真的走了一次 optimizer.step (累积未满 / AMP 跳步时为 False)。
    - ``grad_norm``: 裁剪前的梯度范数 (仅在 sync_gradients 那个 micro-batch 上算出)，否则 None。
    """

    loss_totals: dict[str, float]
    loss_denominators: dict[str, float]
    source_loss_totals: dict[str, dict[str, float]]
    source_loss_denominators: dict[str, dict[str, float]]
    completed_optimizer_step: bool
    grad_norm: torch.Tensor | None


@dataclass(slots=True)
class _CompletedTrainingStep:
    """收尾约简后的 "已完成" 状态 / Stage-3 output: ready for logging.

    字段 / Fields:
    - ``reduced_metrics``: 跨 rank 约简后的标量损失 (可直接打印/记 TensorBoard)。
    - ``learning_rate``: 本步使用的学习率 (从 optimizer 的 param_group 读出)。
    - ``grad_norm_value``: 梯度范数的 host float (无梯度同步步时为 ``nan``)。
    """

    reduced_metrics: dict[str, float]
    learning_rate: float
    grad_norm_value: float

# endregion Training Step State

class DotsTtsTrainingRun:
    """一次完整训练运行 / A single end-to-end training run.

    职责 / Responsibility:
    持有所有训练状态 (accelerator、model、optimizer、scheduler、data loader、progress)，
    在 ``__init__`` 里完成加载 + 前置契约校验，在 ``run`` 里跑主循环直到 ``max_train_steps``。

    设计要点 / Design notes:
    - 三段式训练步 (prepare → accumulate → finalize)：把 "取数据/算归一化"、"前向反向"、
      "约简 metrics + 触发副作用" 分开，使梯度累积窗口的归一化逻辑清晰、副作用 (log/eval/save)
      只在真正完成一次 optimizer step 后触发。
    - 所有跨 rank 聚合都走 **tensor reduction**，刻意避免 object collective——后者会让 NCCL 在
      GPU 上 materialize pickled payload，逼近显存上限时容易 OOM (validation 注释里有详述)。
    """

    # region Lifecycle
    def __init__(self, cfg: app_config.AppConfig, *, debug_enabled: bool = False):
        """加载模型/数据并做前置校验 / Build accelerator, model, optim, data; validate contracts.

        参数 / Args:
        - ``cfg``: 解析后的 ``AppConfig`` (train / train_data / val_data / loss 等子配置)。
        - ``debug_enabled``: 是否打印训练/梯度调试信息 (对应 CLI ``--debug``)。

        关键步骤 / Key steps:
        构造 ``Accelerator`` (DDP + 梯度累积 + tensorboard) → 设随机种子 → 从预训练权重加载
        ``DotsTtsModel`` → 建 AdamW + cosine warmup → ``accelerator.prepare`` 包装 → 校验
        采样率 / audio token 契约与配置一致 → 建训练/验证 DataLoader → 尝试断点续训。
        """
        self.cfg = cfg
        self.progress = train_utils.TrainProgress()
        self.max_train_steps = int(cfg.train.max_train_steps)
        self.grad_accumulation_steps = int(cfg.train.gradient_accumulation_steps)
        self.last_validation_step: int | None = None
        self.consecutive_empty_epochs = 0
        self.saved_latest_checkpoint = False
        self._last_log_step = 0
        self._last_log_time = 0.0
        self._debug_enabled = bool(debug_enabled)
        self._debug_batch_count = 0
        self._debug_audio_sample_rate = int(self.cfg.train_data.train_audio_sample_rate)

        project_config = ProjectConfiguration(
            project_dir=self.cfg.train.output_dir,
            total_limit=self.cfg.train.max_checkpoints_to_keep,
        )
        # find_unused_parameters=False: 所有参数都参与反传，关掉 DDP 的未用参数扫描可省开销。
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.grad_accumulation_steps,
            log_with="tensorboard",
            project_config=project_config,
            # 调度器由我们手动 step (见 _accumulate_training_step)，不让 accelerate 自动随
            # optimizer 推进——这样在 AMP 跳步 (optimizer_step_was_skipped) 时学习率不误进。
            step_scheduler_with_optimizer=False,
        )

        util_module.seed_everything(self.cfg.train.seed)

        model = dots_tts_model.DotsTtsModel.from_pretrained(
            self.cfg.train.pretrained_model_path
        )
        # model.set_cfg_droprate(
        #     cfg_droprate=self.cfg.train.cfg_droprate,
        #     xvec_drop_rate=self.cfg.train.xvec_drop_rate,
        # )
        # 只优化 requires_grad 的参数 (冻结部分如 vocoder / x-vector 编码器不会进 optimizer)。
        optimizer = AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.train.warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        # prepare 在 DDP/混精下包装 model/optim/scheduler；之后 self.model 可能是 DDP wrapper。
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )
        # unwrap 出原始 DotsTtsModel：调用它的非 forward 方法 (prepare_training_batch、读 config /
        # hop_size 等) 需要绕过 DDP wrapper。
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        expected_sample_rate = int(self.unwrapped_model.config.vocoder.sample_rate)
        # 每个 "LLM token"(即一个 audio patch) 覆盖的波形采样点数 = hop_size(每 latent 帧的采样点)
        # × patch_size(每 patch 含多少 latent 帧)。数据侧切 patch 必须与模型契约严格一致。
        expected_audio_samples_per_llm_token = (
            int(self.unwrapped_model.hop_size) * int(self.unwrapped_model.config.patch_size)
        )
        # 前置契约校验 / Pre-flight contract checks: 数据配置的采样率与 audio-token 粒度必须与
        # 预训练模型完全对齐，否则 latent 切 patch 会错位，损失无意义——宁可早报错也不静默训坏。
        if int(self.cfg.train_data.train_audio_sample_rate) != expected_sample_rate:
            raise ValueError(
                f"train_data.train_audio_sample_rate={int(self.cfg.train_data.train_audio_sample_rate)} "
                f"does not match the pretrained model sample rate {expected_sample_rate}."
            )
        if (
            int(self.cfg.train_data.audio_samples_per_llm_token)
            != expected_audio_samples_per_llm_token
        ):
            raise ValueError(
                "train_data.audio_samples_per_llm_token="
                f"{int(self.cfg.train_data.audio_samples_per_llm_token)} "
                "does not match the pretrained model audio token contract "
                f"{expected_audio_samples_per_llm_token}."
            )
        if self.cfg.val_data is not None:
            if int(self.cfg.val_data.train_audio_sample_rate) != expected_sample_rate:
                raise ValueError(
                    f"val_data.train_audio_sample_rate={int(self.cfg.val_data.train_audio_sample_rate)} "
                    f"does not match the pretrained model sample rate {expected_sample_rate}."
                )
            if (
                int(self.cfg.val_data.audio_samples_per_llm_token)
                != expected_audio_samples_per_llm_token
            ):
                raise ValueError(
                    "val_data.audio_samples_per_llm_token="
                    f"{int(self.cfg.val_data.audio_samples_per_llm_token)} "
                    "does not match the pretrained model audio token contract "
                    f"{expected_audio_samples_per_llm_token}."
                )

        # 仅主进程打印模型规模与分布式类型 (避免多 rank 重复刷屏)。/ Main-process-only banner.
        if self.accelerator.is_main_process:
            total_params = sum(param.numel() for param in self.unwrapped_model.parameters())
            trainable_params = sum(
                param.numel()
                for param in self.unwrapped_model.parameters()
                if param.requires_grad
            )
            self.accelerator.print(f"Total parameters: {total_params:,}")
            self.accelerator.print(f"Trainable parameters: {trainable_params:,}")
            self.accelerator.print(
                f"Distributed type: {self.accelerator.distributed_type}"
            )

        tokenizer = self.unwrapped_model.tokenizer
        self.tokenizer = tokenizer
        train_dataset = data_module.build_training_dataset(
            self.cfg.train_data,
            tokenizer=tokenizer,
            seed=int(self.cfg.train.seed),
            accelerator=self.accelerator,
        )
        self.train_loader = data_module.build_training_dataloader(
            train_dataset,
            self.cfg.train_data,
            tokenizer=tokenizer,
        )

        self.val_loader = None
        if (
            self.cfg.train.eval_interval is not None
            or self.cfg.train.run_eval_on_start
        ):
            if self.cfg.val_data is None:
                raise ValueError(
                    "Validation requires val_data when eval_interval or "
                    "run_eval_on_start is enabled."
                )
            # 验证集深拷贝一份配置并清掉 num_tokens_per_epoch：评估要走完整一遍数据 (不按 token
            # 预算截断 epoch)，否则每次 eval 看到的样本数不固定、指标不可比。
            validation_data_cfg = self.cfg.val_data.model_copy(deep=True)
            validation_data_cfg.num_tokens_per_epoch = None
            val_dataset = data_module.build_validation_dataset(
                validation_data_cfg,
                tokenizer=tokenizer,
                seed=int(self.cfg.train.seed),
                accelerator=self.accelerator,
            )
            self.val_loader = data_module.build_validation_dataloader(
                val_dataset,
                validation_data_cfg,
                tokenizer=tokenizer,
            )

        self._resume_if_available()
        # 续训后用恢复出的 epoch 给 loader 播种，保证 shuffle/分片与中断前一致。
        self.train_loader.set_epoch(self.progress.epoch)

    def run(self) -> int:
        """训练主循环 / Main loop: optional eval-on-start → step until done → final eval+save.

        返回 / Returns: 进程退出码 (0 表示正常结束)，供 ``SystemExit`` 使用。
        ``finally`` 块保证无论成功还是异常都关掉数据流并结束 tracker (释放文件句柄/子进程)。
        """
        self.accelerator.init_trackers("dots_tts")
        self._write_run_config()
        self.optimizer.zero_grad(set_to_none=True)

        try:
            if self.cfg.train.run_eval_on_start:
                self._run_validation()
                self.last_validation_step = self.progress.global_step

            self._last_log_step = self.progress.global_step
            self._last_log_time = time.perf_counter()

            while self.progress.global_step < self.max_train_steps:
                self._run_training_step()

            # 收尾 eval：仅当开启了周期 eval、且最后一步还没在循环内评估过时补一次。
            if (
                self.cfg.train.eval_interval is not None
                and self.val_loader is not None
                and self.progress.global_step > 0
                and self.last_validation_step != self.progress.global_step
            ):
                self._run_validation()

            # 收尾 save：若最后一步恰好没落在 save_interval 上，补存一次最新 checkpoint。
            if not self.saved_latest_checkpoint:
                self._save_checkpoint(float(self.optimizer.param_groups[0]["lr"]))
            return 0
        finally:
            try:
                self._close_data_streams()
            finally:
                self.accelerator.end_training()

    def _write_run_config(self) -> None:
        """把生效配置落盘 / Dump the resolved config to ``output_dir/config.yml`` (main only)."""
        if not bool(getattr(self.accelerator, "is_main_process", True)):
            return
        output_dir = Path(self.cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.yml"
        with config_path.open("w", encoding="utf-8") as fout:
            yaml.safe_dump(
                self.cfg.to_dict(),
                fout,
                sort_keys=False,
                allow_unicode=True,
            )

    def _close_data_streams(self) -> None:
        """关闭流式 loader / Close streaming loaders to release worker processes & file handles."""
        for loader_name in ("train_loader", "val_loader"):
            loader = getattr(self, loader_name, None)
            # 鸭子类型探测 close()：不是所有 loader 都有 (例如 None 或普通 DataLoader)。
            close = getattr(loader, "close", None)
            if callable(close):
                close()
            setattr(self, loader_name, None)

    def _resume_if_available(self) -> None:
        """有最新 checkpoint 就续训 / Resume from the latest checkpoint if one exists.

        无 checkpoint (``FileNotFoundError``) 时静默从头训。续训会恢复 model/optim/scheduler/
        progress 与 **已提交的** 数据进度；内存里的 prefetch/batching 状态在重启时重建，故只能
        续到 "已提交样本" 的粒度 (见末尾打印)。
        """
        try:
            resume_dir = train_checkpoint.resolve_latest_train_checkpoint(
                self.cfg.train.output_dir
            )
        except FileNotFoundError:
            return

        resume_state = train_checkpoint.load_train_checkpoint(
            self.accelerator,
            self.model,
            self.optimizer,
            self.progress,
            resume_dir,
            self.scheduler,
        )
        # 若续训时改了 max_train_steps，cosine 调度的形状会与存档不一致，仅警告不报错。
        saved_max_train_steps = int(resume_state["scheduler_state"]["max_train_steps"])
        if saved_max_train_steps != self.max_train_steps:
            self.accelerator.print(
                "Warning: resumed scheduler was saved with "
                f"max_train_steps={saved_max_train_steps}, but current run uses "
                f"{self.max_train_steps}."
            )

        self.train_loader.load_state_dict(resume_state["data_state"])
        self.accelerator.print(
            "Resumed training from "
            f"{resume_dir} at step {self.progress.global_step}. "
            "Restored committed data state. "
            "In-memory prefetch and batching state is rebuilt on restart, so only "
            "committed sample progress is resumed."
        )
    # endregion Lifecycle

    # region Training Step Pipeline
    def _run_training_step(self) -> None:
        """跑一个完整 optimizer step (三段式) / One full optimizer step, in three stages.

        Stage1 ``_prepare_training_step`` 攒一个梯度累积窗口 + 全局归一化分母；
        Stage2 ``_accumulate_training_step`` 前向反向并累积统计；
        Stage3 仅在真正完成 optimizer step 后约简 metrics 并触发 log/eval/save 副作用。
        整步用 try/except 包住，OOM 时调 ``abort_on_out_of_memory`` 打印诊断后再抛。
        """
        try:
            self.model.train()
            # Stage 1: collect one synchronized accumulation window and its
            # normalization factors before touching model state.
            prepared_step = self._prepare_training_step()

            # Stage 2: run forward/backward over the prepared micro-batches and
            # accumulate overall/source statistics for the completed optimizer step.
            accumulated_step = self._accumulate_training_step(prepared_step)

            # Stage 3: advance counters, reduce metrics, then trigger side effects
            # (logging, validation, checkpointing) only after a real optimizer step.
            self._apply_consumed_counts(prepared_step.consumed_counts)
            # 累积窗口未满 / AMP 跳步时没有真正更新参数：进度计数已记 (token 确实被消费了)，
            # 但不推进 global_step、不触发任何副作用，直接返回等下个累积步。
            if not accumulated_step.completed_optimizer_step:
                return
            completed_step = self._finalize_completed_training_step(accumulated_step)
            if train_utils.should_log_training_step(
                self.progress.global_step,
                int(self.cfg.train.log_interval),
            ):
                reduced_by_source = train_utils.reduce_source_metrics(
                    accumulated_step.source_loss_totals,
                    accumulated_step.source_loss_denominators,
                    device=self.accelerator.device,
                    loss_config=self.cfg.loss,
                )
                current_time = time.perf_counter()
                report = train_utils.build_train_step_report(
                    completed_step.reduced_metrics,
                    learning_rate=completed_step.learning_rate,
                    grad_norm=completed_step.grad_norm_value,
                    current_time=current_time,
                    last_log_step=self._last_log_step,
                    last_log_time=self._last_log_time,
                    progress=self.progress,
                    max_train_steps=self.max_train_steps,
                    reduced_by_source=reduced_by_source,
                )
                self.accelerator.log(
                    report.log_values,
                    step=self.progress.global_step,
                )
                self.accelerator.print(report.console_line)
                self._last_log_step = self.progress.global_step
                self._last_log_time = current_time

            # 周期性 validation：命中 eval_interval 就评估，并记下避免收尾时重复评。
            if (
                self.cfg.train.eval_interval is not None
                and self.progress.global_step % self.cfg.train.eval_interval == 0
            ):
                self._run_validation()
                self.last_validation_step = self.progress.global_step

            # 周期性 checkpoint：命中 save_interval 就存，置位避免收尾时重复存。
            if self.progress.global_step % self.cfg.train.save_interval == 0:
                self._save_checkpoint(completed_step.learning_rate)
                self.saved_latest_checkpoint = True
        except BaseException as exc:
            train_utils.abort_on_out_of_memory(
                exc,
                stage="train",
                batch=None,
                progress=self.progress,
                device=self.accelerator.device,
                process_index=int(getattr(self.accelerator, "process_index", 0)),
                num_processes=int(getattr(self.accelerator, "num_processes", 1)),
            )
            raise

    def _prepare_training_step(self) -> _PreparedTrainingStep:
        """Stage 1: 攒一个累积窗口 + 算全局归一化分母 / Build one accumulation window.

        关键技巧 / Key trick:
        peek/commit 两段式取数据——先 ``peek_batch`` 看本 rank 有没有 batch，再用
        ``any_rank_true`` 跨 rank 求 "或"：只要 **任一** rank 没数据就让所有 rank 一起翻 epoch。
        这样保证所有 rank 步调一致 (否则下游的 collective 会因 rank 数据数不齐而死锁)。确认全员
        都有 batch 后才 ``commit_batch`` (真正消费)。分母用 loss mask 折叠出各损失项的有效元素数。
        """
        micro_batches: list[dict] = []
        local_denominators: dict[str, float] = {}

        while len(micro_batches) < self.grad_accumulation_steps:
            # peek 不消费：先探一眼，等确认全 rank 同步后再决定 commit 还是翻 epoch。
            batch, has_batch = self.train_loader.peek_batch()
            # 跨 rank 求 "任一为真"：只要有 rank 取不到 batch，全体一起翻 epoch 重来。
            if train_utils.any_rank_true(not has_batch, device=self.accelerator.device):
                self._advance_epoch_after_empty_batch(has_local_batch=has_batch)
                continue

            self.consecutive_empty_epochs = 0
            self.train_loader.commit_batch()  # 全员都有 batch，正式消费这一个。
            # CPU 侧补 loss mask 元数据 (LM / flow-matching / EOS 各自的有效位置)。
            prepared_batch = self.unwrapped_model.prepare_training_batch(batch)
            self._maybe_debug_training_batch(prepared_batch)
            # 累积本 rank 的分母 (各损失项有效元素数)；首个 batch 用其 key 初始化字典骨架。
            batch_denominators = loss_ops.to_host_named_scalars(
                loss_ops.collapse_loss_masks(prepared_batch["loss_masks"])
            )
            if not local_denominators:
                local_denominators = {name: 0.0 for name in batch_denominators}
            loss_ops.accumulate_named_scalars_(local_denominators, batch_denominators)
            micro_batches.append(prepared_batch)

        # 把本窗口消费的 [总 token, audio token, text token] 计数跨 rank 求和，用于进度统计。
        consumed_counts = train_utils.sum_integer_counters_across_ranks(
            [
                sum(int(batch["input_ids_lengths"].sum().item()) for batch in micro_batches),
                sum(int(batch["num_audio_tokens"].sum().item()) for batch in micro_batches),
                sum(int(batch["num_text_tokens"].sum().item()) for batch in micro_batches),
            ],
            device=self.accelerator.device,
        )
        # 全局分母 = 各 rank、整窗口的有效元素数之和，使损失归一化等价于一个超大 batch 的平均。
        global_denominators = loss_ops.sum_named_scalars_across_ranks(
            local_denominators,
            device=self.accelerator.device,
        )
        return _PreparedTrainingStep(
            micro_batches=micro_batches,
            consumed_counts=consumed_counts,
            global_denominators=global_denominators,
        )

    def _advance_epoch_after_empty_batch(self, *, has_local_batch: bool) -> None:
        """翻 epoch / Advance epoch when ranks went out of sync on data availability.

        若本 rank 本来 peek 到了 batch (但别的 rank 没有)，要 ``discard_batch`` 丢掉这次 peek，
        保证下个 epoch 不漏不重；连续翻太多次仍凑不齐就报错 (数据分片/过滤把某些 rank 饿死了)。
        """
        if has_local_batch:
            self.train_loader.discard_batch()  # 本 rank 有数据但全局不同步，丢弃这次 peek。
        self.progress.epoch += 1
        self.train_loader.set_epoch(self.progress.epoch)
        self.consecutive_empty_epochs += 1
        if self.consecutive_empty_epochs > _EMPTY_EPOCH_TOLERANCE:
            raise RuntimeError(
                "Unable to obtain a synchronized training batch across ranks. "
                "Check shard assignment, dataset size, and filtering constraints."
            )

    def _accumulate_training_step(
        self,
        prepared_step: _PreparedTrainingStep,
    ) -> _AccumulatedTrainingStep:
        """Stage 2: 对窗口内每个 micro-batch 跑前向/反向并累积统计 / Forward+backward.

        关键点 / Key points:
        - ``accelerator.accumulate`` 上下文在累积窗口内自动只在最后一个 micro-batch 做梯度同步
          (DDP no_sync 优化)，故 ``sync_gradients`` 仅在窗口末为真——此时才裁剪/step/调度/清零。
        - ``compute_gradient_loss`` 用 **全局分母** 归一化，并把 ddp_world_size 与累积步数一并传入，
          抵消 DDP 的 all-reduce 平均 + 累积求和，让等效梯度等于一个大 batch 的真实平均梯度。
        - 损失统计 (totals/denominators) 与 source 分组统计在 host 上累积，留到收尾阶段跨 rank 约简。
        """
        accumulated_loss_totals: dict[str, float] = {}
        accumulated_loss_denominators: dict[str, float] = {}
        accumulated_source_loss_totals: dict[str, dict[str, float]] = {}
        accumulated_source_loss_denominators: dict[str, dict[str, float]] = {}
        completed_optimizer_step = False
        grad_norm = None

        for batch in prepared_step.micro_batches:
            # 此处才把 batch 张量搬到 GPU (准备阶段保持在 CPU 以压低峰值显存)。
            batch = train_utils.move_to_device(batch, self.accelerator.device)
            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():  # AMP 混精前向。
                    # forward 返回三类损失项：LM logits CE、flow-matching velocity 回归、EOS。
                    loss_terms = self.model(batch)
                    loss = loss_ops.compute_gradient_loss(
                        loss_terms,
                        global_normalizers=prepared_step.global_denominators,
                        loss_config=self.cfg.loss,
                        ddp_world_size=int(self.accelerator.num_processes),
                        gradient_accumulation_steps=self.grad_accumulation_steps,
                    )

                # 把损失项折叠成 (分子=加权和, 分母=计数) 并搬到 host：这是 **统计用** 的数值，
                # 与上面用于反传的 loss 标量分开 (统计求未归一化的平均，反传求带梯度的归一化损失)。
                batch_loss_totals, batch_loss_denominators = (
                    loss_ops.collapse_loss_terms(loss_terms)
                )
                batch_loss_totals = loss_ops.to_host_named_scalars(batch_loss_totals)
                batch_loss_denominators = loss_ops.to_host_named_scalars(
                    batch_loss_denominators
                )
                if not accumulated_loss_totals:  # 首个 micro-batch 用其 key 建字典骨架。
                    accumulated_loss_totals = {name: 0.0 for name in batch_loss_totals}
                    accumulated_loss_denominators = {
                        name: 0.0 for name in batch_loss_denominators
                    }
                loss_ops.accumulate_named_scalars_(
                    accumulated_loss_totals,
                    batch_loss_totals,
                )
                loss_ops.accumulate_named_scalars_(
                    accumulated_loss_denominators,
                    batch_loss_denominators,
                )

                # 同样的统计但按数据 source 分组 (多源混训时能分别监控每个 source 的损失)。
                batch_source_totals, batch_source_denominators = (
                    loss_ops.collapse_loss_terms_by_source(
                        loss_terms,
                        source_names=batch["source_names"],
                    )
                )
                loss_ops.accumulate_grouped_named_scalars_(
                    accumulated_source_loss_totals,
                    batch_source_totals,
                )
                loss_ops.accumulate_grouped_named_scalars_(
                    accumulated_source_loss_denominators,
                    batch_source_denominators,
                )

                self.accelerator.backward(loss)  # 累积梯度 (窗口末才真正 all-reduce 同步)。
                # sync_gradients 仅在累积窗口最后一个 micro-batch 为真——此时梯度已就绪，做裁剪+更新。
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.train.grad_clip_norm,
                    )
                    self._maybe_print_gradient_debug(grad_norm)
                    self.optimizer.step()
                    # AMP 在梯度溢出时会跳过这次 step；跳过则不推进 scheduler，避免学习率轨迹错位。
                    completed_optimizer_step = (
                        not self.accelerator.optimizer_step_was_skipped
                    )
                    if completed_optimizer_step:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
            batch.clear()  # 主动清空 dict 释放本 micro-batch 的 GPU 张量引用，压低峰值显存。

        return _AccumulatedTrainingStep(
            loss_totals=accumulated_loss_totals,
            loss_denominators=accumulated_loss_denominators,
            source_loss_totals=accumulated_source_loss_totals,
            source_loss_denominators=accumulated_source_loss_denominators,
            completed_optimizer_step=completed_optimizer_step,
            grad_norm=grad_norm,
        )

    def _apply_consumed_counts(self, consumed_counts: list[int]) -> None:
        """累加进度计数 / Bump token counters by [total, audio, text] from this window."""
        self.progress.total_tokens += consumed_counts[0]
        self.progress.audio_tokens += consumed_counts[1]
        self.progress.text_tokens += consumed_counts[2]

    def _finalize_completed_training_step(
        self,
        accumulated_step: _AccumulatedTrainingStep,
    ) -> _CompletedTrainingStep:
        """Stage 3: 推进 step 并把统计跨 rank 约简成可读 metrics / Reduce to scalar metrics.

        先做完整性断言 (没有任何统计就说明数据/mask 配置坏了)，再 ``global_step += 1``，
        然后把分子/分母分别跨 rank 求和、相除得到平均损失，并读出当前 lr 与梯度范数。
        """
        if not accumulated_step.loss_totals or not accumulated_step.loss_denominators:
            raise RuntimeError("Training step produced no accumulated loss totals.")
        # 所有分母都为 0 → 整步没有任何有效监督信号 (mask 全空)，属于配置错误，直接报错。
        if all(
            float(value) == 0.0 for value in accumulated_step.loss_denominators.values()
        ):
            raise RuntimeError("Accumulated training step produced no loss statistics.")

        self.progress.global_step += 1
        self.saved_latest_checkpoint = False  # 参数已变，最新 checkpoint 失效，需重新保存。

        reduced_totals = loss_ops.sum_named_scalars_across_ranks(
            accumulated_step.loss_totals,
            device=self.accelerator.device,
        )
        reduced_denominators = loss_ops.sum_named_scalars_across_ranks(
            accumulated_step.loss_denominators,
            device=self.accelerator.device,
        )
        reduced_metrics = loss_ops.reduce_loss_statistics(
            reduced_totals,
            reduced_denominators,
            loss_config=self.cfg.loss,
        )
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        # grad_norm 可能为 None (理论上完成 step 时应非 None)；用 nan 占位以免日志里出现假零。
        grad_norm_value = (
            math.nan
            if accumulated_step.grad_norm is None
            else float(accumulated_step.grad_norm.detach().float().item())
        )
        return _CompletedTrainingStep(
            reduced_metrics=reduced_metrics,
            learning_rate=learning_rate,
            grad_norm_value=grad_norm_value,
        )
    # endregion Training Step Pipeline

    # region Validation
    def _run_validation(self) -> None:
        """跑一遍验证集并记指标 / Run validation and log reduced metrics.

        复用与训练完全相同的 batch 准备与损失聚合路径 (保证 train/val 指标可比)，但在
        ``no_grad`` + ``eval`` 下跑、不反传。聚合刻意只用 **tensor reduction** 而非 object
        collective：验证逼近训练显存上限，object collective 会让 NCCL 在 GPU 上 materialize
        pickled payload 而 OOM。评估完按需把模型切回 train 模式。
        """
        try:
            if self.val_loader is None:
                raise ValueError(
                    "Validation requested, but validation loader was not initialized."
                )
            self.val_loader.set_epoch(0)  # 验证固定从 epoch 0 起，保证每次评估遍历顺序一致。

            was_training = bool(self.model.training)  # 记下原状态，评估后好还原。
            self.model.eval()

            overall_loss_totals = None
            overall_loss_denominators = None
            source_loss_totals: dict[str, dict[str, float]] = {}
            source_loss_denominators: dict[str, dict[str, float]] = {}
            processed_batches = 0

            # Collect rank-local partial sums using the same batch preparation and
            # loss aggregation path as training.
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    # 可选地只评前 max_eval_batches 个 batch (大验证集时控时长)。
                    if (
                        self.cfg.train.max_eval_batches is not None
                        and batch_idx >= self.cfg.train.max_eval_batches
                    ):
                        break

                    batch = self.unwrapped_model.prepare_training_batch(batch)
                    batch = train_utils.move_to_device(batch, self.accelerator.device)

                    with self.accelerator.autocast():
                        loss_terms = self.model(batch)

                    batch_loss_totals, batch_loss_denominators = (
                        loss_ops.collapse_loss_terms(loss_terms)
                    )
                    batch_loss_totals = loss_ops.to_host_named_scalars(batch_loss_totals)
                    batch_loss_denominators = loss_ops.to_host_named_scalars(
                        batch_loss_denominators
                    )
                    if overall_loss_totals is None:
                        overall_loss_totals = {name: 0.0 for name in batch_loss_totals}
                        overall_loss_denominators = {
                            name: 0.0 for name in batch_loss_denominators
                        }
                    loss_ops.accumulate_named_scalars_(
                        overall_loss_totals,
                        batch_loss_totals,
                    )
                    loss_ops.accumulate_named_scalars_(
                        overall_loss_denominators,
                        batch_loss_denominators,
                    )

                    batch_source_totals, batch_source_denominators = (
                        loss_ops.collapse_loss_terms_by_source(
                            loss_terms,
                            source_names=batch["source_names"],
                        )
                    )
                    loss_ops.accumulate_grouped_named_scalars_(
                        source_loss_totals,
                        batch_source_totals,
                    )
                    loss_ops.accumulate_grouped_named_scalars_(
                        source_loss_denominators,
                        batch_source_denominators,
                    )
                    processed_batches += 1

            # Merge rank-local partial sums with tensor reductions only. Validation
            # runs close to the training memory ceiling, so object collectives are
            # not acceptable here because NCCL materializes pickled payloads on GPU.
            processed_batches = train_utils.sum_integer_counters_across_ranks(
                [processed_batches],
                device=self.accelerator.device,
            )[0]
            overall_loss_totals = loss_ops.sum_named_scalars_across_ranks(
                overall_loss_totals or {},
                device=self.accelerator.device,
            )
            overall_loss_denominators = loss_ops.sum_named_scalars_across_ranks(
                overall_loss_denominators or {},
                device=self.accelerator.device,
            )
            source_loss_totals = loss_ops.sum_grouped_named_scalars_across_ranks(
                source_loss_totals,
                device=self.accelerator.device,
            )
            source_loss_denominators = (
                loss_ops.sum_grouped_named_scalars_across_ranks(
                    source_loss_denominators,
                    device=self.accelerator.device,
                )
            )

            if processed_batches <= 0:
                raise RuntimeError(
                    "Validation produced no batches. Check validation data configuration."
                )
            if not overall_loss_totals or not overall_loss_denominators:
                raise RuntimeError("Validation produced no aggregate loss totals.")

            reduced_metrics = loss_ops.reduce_loss_statistics(
                overall_loss_totals,
                overall_loss_denominators,
                loss_config=self.cfg.loss,
            )
            reduced_by_source = loss_ops.reduce_loss_statistics_by_source(
                source_loss_totals,
                source_loss_denominators,
                loss_config=self.cfg.loss,
            )

            if was_training:
                self.model.train()  # 评估前若处于 train 模式，还原回去继续训练。

            self.accelerator.log(
                train_utils.build_validation_log_dict(
                    reduced_metrics,
                    reduced_by_source=reduced_by_source,
                ),
                step=self.progress.global_step,
            )
            self.accelerator.print(
                train_utils.format_validation_line(
                    reduced_metrics,
                    global_step=self.progress.global_step,
                    reduced_by_source=reduced_by_source,
                )
            )
        except BaseException as exc:
            train_utils.abort_on_out_of_memory(
                exc,
                stage="validation",
                batch=None,
                progress=self.progress,
                device=self.accelerator.device,
                process_index=int(getattr(self.accelerator, "process_index", 0)),
                num_processes=int(getattr(self.accelerator, "num_processes", 1)),
            )
            raise
    # endregion Validation

    # region Checkpointing
    def _save_checkpoint(self, learning_rate: float) -> None:
        """保存断点 / Save a resumable checkpoint (model/optim/progress/data + scheduler meta).

        随 scheduler 一起存的元信息 (type/base_lr/warmup_steps/max_train_steps 等) 让续训时能
        校验调度形状是否一致 (见 ``_resume_if_available`` 里的 max_train_steps 警告)。
        ``train_loader.state_dict()`` 仅记录 **已提交** 的数据进度。
        """
        train_checkpoint.save_train_checkpoint(
            self.accelerator,
            self.model,
            self.optimizer,
            self.progress,
            self.cfg.train.output_dir,
            self.cfg.train.max_checkpoints_to_keep,
            self.train_loader.state_dict(),
            {
                "type": "transformers_cosine_with_warmup",
                "global_step": int(self.progress.global_step),
                "base_lr": float(self.cfg.train.learning_rate),
                "current_lr": float(learning_rate),
                "warmup_steps": int(self.cfg.train.warmup_steps),
                "max_train_steps": int(self.max_train_steps),
                "state_dict": self.scheduler.state_dict(),
            },
        )
    # endregion Checkpointing

    # region Debug Logging
    def _maybe_debug_training_batch(self, batch: dict[str, object]) -> None:
        """可选打印前几条训练 batch 的明细 / Dump first few train batches when --debug.

        三道门控：必须开了 ``--debug``、必须是主进程、且累计打印数未超 ``_DEBUG_BATCH_LIMIT``。
        用于人工核对数据契约 (token 解码、波形采样率等)，正常训练不触发。
        """
        if not bool(getattr(self, "_debug_enabled", False)):
            return
        if not bool(getattr(self.accelerator, "is_main_process", True)):
            return
        if self._debug_batch_count >= _DEBUG_BATCH_LIMIT:  # 只看头几条，看够就闭嘴。
            return

        batch_index = self._debug_batch_count
        self._debug_batch_count += 1
        for line in train_utils.build_data_debug_lines(
            batch,
            batch_index=batch_index,
            tokenizer=self.tokenizer,
            sample_rate=self._debug_audio_sample_rate,
        ):
            self.accelerator.print(line)

    def _maybe_print_gradient_debug(self, grad_norm: torch.Tensor | None) -> None:
        """可选打印逐层梯度诊断 / Optionally dump per-layer grad stats.

        触发条件由 ``should_print_gradient_debug`` 决定：开了 debug、主进程、且处于 "最初几步"
        或命中 log_interval。用 ``global_step + 1`` 是因为本步的 ``global_step`` 还没在
        ``_finalize_completed_training_step`` 里自增 (此刻在累积阶段)，+1 即对应本步将成为的步号。
        """
        if grad_norm is None:  # 非梯度同步步没有 grad_norm，直接跳过。
            return
        if not train_utils.should_print_gradient_debug(
            debug_enabled=bool(getattr(self, "_debug_enabled", False)),
            is_main_process=bool(getattr(self.accelerator, "is_main_process", True)),
            next_global_step=self.progress.global_step + 1,
            log_interval=int(self.cfg.train.log_interval),
            early_step_limit=_DEBUG_GRAD_EARLY_STEP_LIMIT,
        ):
            return
        for line in train_utils.build_gradient_debug_lines(
            self.unwrapped_model,
            global_step=self.progress.global_step + 1,
            grad_norm=float(grad_norm.detach().float().item()),
            grad_clip_norm=float(self.cfg.train.grad_clip_norm),
        ):
            self.accelerator.print(line)
    # endregion Debug Logging


# region CLI
def parse_args(argv=None):
    """解析命令行 / Parse CLI: ``--config`` (路径) 与 ``--debug`` (开关)。"""
    parser = argparse.ArgumentParser(
        description="Accelerate training entrypoint for dots.tts."
    )
    parser.add_argument("--config", default=app_config.DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print training debug information.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """入口 / Entry: load config, build the run, execute, return its exit code."""
    args = parse_args(argv)
    return DotsTtsTrainingRun(
        app_config.load_config(args.config),
        debug_enabled=args.debug,
    ).run()


if __name__ == "__main__":
    # 以 run() 的返回码作为进程退出码 (0=成功)，便于 accelerate launch / shell 判断成败。
    raise SystemExit(main())
# endregion CLI
