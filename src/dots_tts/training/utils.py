"""Shared helpers for the dots_tts training entrypoints.

本文件：dots.tts 训练入口共享的"工具集"，集中放那些与模型/损失无关、却被
训练主循环反复用到的零碎逻辑。它不参与前向/反向计算，只负责把训练跑顺、跑得
可观测、出错时能优雅退出。

在数据流里的位置 / Where this sits in the pipeline：
训练主循环（trainer）在每个 step 里会用到这里的几类能力——
  1. 分布式聚合 distributed reduction：把各 rank（GPU 进程）上的标量计数器、
     OOM 标志位通过 ``torch.distributed.all_reduce`` 汇总成全局值；
  2. 设备搬运 device placement：把一个 batch（嵌套的 tensor/dict/list/dataclass）
     递归搬到 GPU；
  3. 失败处理 failure handling：检测到 CUDA out-of-memory 时打印诊断信息并在
     多进程下强制退出（避免 NCCL 死锁）；
  4. 调试输出 debug logging：把一个 batch / 梯度状态打印成可读的诊断行，
     便于排查数据问题和梯度爆炸/NaN；
  5. 步进报告 step reporting：把 loss/lr/grad_norm/吞吐/ETA 等指标既组织成
     给 experiment tracker（如 wandb）的扁平 dict，又格式化成一行 console log。

关键类 / 函数清单 Key classes & functions：
  - TrainProgress / TrainStepReport：训练进度计数器 与 单步报告的数据容器。
  - any_rank_true / sum_integer_counters_across_ranks：跨 rank 的布尔归约 与 整数求和。
  - move_to_device：递归把 batch 搬到目标 device。
  - abort_on_out_of_memory 及其 ``_*`` 私有助手：OOM 诊断与退出。
  - build_data_debug_lines / build_gradient_debug_lines：数据/梯度调试行构造。
  - reduce_source_metrics / build_train_step_report：按数据来源聚合指标、组装单步报告。
  - flatten_config / format_scalar / build_train_log_dict / format_train_line
    / build_validation_log_dict / format_validation_line：配置展平与各类日志格式化。
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
import torch.distributed as dist

from dots_tts.training import losses as loss_ops

# ---------------------------------------------------------------------------
# Training State
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainProgress:
    """Minimal progress counters that must survive checkpoint save/load.

    训练进度计数器：只保留"必须随 checkpoint 一起存档/恢复"的最小状态，
    这样断点续训（resume）后 step 数、已消费 token 数都能接续，日志和 ETA
    不会错乱。``slots=True`` 关掉 ``__dict__`` 以省内存、防止误加字段。

    字段含义 / fields：
      - global_step：已完成的优化步数（optimizer step，非 micro-batch）。
      - epoch：当前 epoch 序号。
      - total_tokens / audio_tokens / text_tokens：累计消费的总 token / 音频
        latent token / 文本 token 数（吞吐与 ETA 的统计口径）。
    """

    global_step: int = 0
    epoch: int = 0
    total_tokens: int = 0
    audio_tokens: int = 0
    text_tokens: int = 0


@dataclass(slots=True)
class TrainStepReport:
    """One training step's two output forms / 单步报告的两种产物容器。

    把一次 step 的指标同时打包成两份：``log_values`` 是给 experiment
    tracker（wandb 等）的扁平 ``{metric_name: float}`` 字典；``console_line``
    是给终端打印的单行人类可读字符串。二者由 :func:`build_train_step_report`
    一起生成，避免调用方两处各算一遍。
    """

    log_values: dict[str, float]
    console_line: str


# ---------------------------------------------------------------------------
# Distributed Helpers
# ---------------------------------------------------------------------------


def any_rank_true(flag: bool, *, device: torch.device) -> bool:
    """Return ``True`` if any distributed rank reports ``flag=True``.

    跨 rank 的"逻辑或"：只要任意一个 GPU 进程的 ``flag`` 为真就返回 True。
    典型用途是让所有 rank 对"某个 rank 出错/数据耗尽，大家一起停"达成一致——
    必须所有 rank 都执行同一个 all_reduce，否则会卡死（collective 不匹配）。

    实现：把布尔转成 0/1 张量，用 ``ReduceOp.MAX`` 归约（MAX 即布尔 OR）。
    单机非分布式时跳过 all_reduce，直接返回本地值。
    """
    packed = torch.tensor(int(flag), device=device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)  # MAX over {0,1} == 逻辑 OR
    return bool(packed.item())


def sum_integer_counters_across_ranks(
    values: list[int],
    *,
    device: torch.device,
) -> list[int]:
    """All-reduce integer counters and return their cross-rank sums.

    跨 rank 求和：把一组整数计数器（如本 rank 本 step 消费的 token 数）通过
    ``ReduceOp.SUM`` 汇总成全局总量，返回与输入等长的列表。用 int64 打包以
    避免大计数溢出。非分布式时原样返回本地计数。
    """
    packed = torch.tensor(values, device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return [int(value) for value in packed.tolist()]


def move_to_device(value, device):
    """Recursively move nested tensors/dataclasses onto ``device``.

    把一个任意嵌套的 batch（tensor / dict / list / tuple / dataclass 的任意
    组合）整体搬到目标 ``device``（通常是当前 rank 的 GPU）。collate 出来的
    batch 结构往往是嵌套的，单纯 ``batch.to(device)`` 覆盖不到，故在此递归处理。

    设计要点：
      - tensor 用 ``non_blocking=True`` 异步拷贝，配合 pinned-memory 可与计算
        重叠、减少搬运等待。
      - dataclass 分支重建同类型实例（``type(value)(**...)``），逐字段递归搬运；
        ``not isinstance(value, type)`` 排除"类对象本身"误入此分支。
      - 非张量、非容器的标量/字符串等原样返回。
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)  # 异步 H2D 拷贝，可与计算重叠
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    # dataclass（且不是类本身）：按字段递归搬运后重建同类型实例
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: move_to_device(getattr(value, field.name), device)
                for field in fields(value)
            }
        )
    return value  # 标量/字符串/None 等不可搬运对象原样返回


# ---------------------------------------------------------------------------
# Failure Handling
# ---------------------------------------------------------------------------


def abort_on_out_of_memory(
    exc: BaseException,
    *,
    stage: str,
    batch: dict[str, object] | None,
    progress: TrainProgress,
    device: torch.device,
    process_index: int,
    num_processes: int,
) -> None:
    """On a CUDA OOM exception, print diagnostics and hard-exit if distributed.

    专门处理训练中的 CUDA out-of-memory：先判定异常是否真的是 OOM（非 OOM
    直接返回，让上层正常处理）；是 OOM 则把出错阶段、进度、batch 形状摘要、
    当前显存占用打到 stderr，连同完整 traceback，方便事后定位是哪个 batch
    把显存撑爆。

    关键设计——多进程下用 ``os._exit(1)`` 而非 raise/sys.exit：
    OOM 往往只发生在部分 rank 上，若让该 rank 正常抛栈退出，其余 rank 仍会
    阻塞在下一次 collective（all_reduce 等）上形成 NCCL 死锁/挂起。直接
    ``os._exit`` 立即杀掉进程，触发整个分布式作业由 launcher 一起失败退出，
    避免长时间挂死。单进程（num_processes<=1）则不强退，让异常照常向上传播。

    参数：``stage`` 出错阶段标签；``batch`` 当前 batch（可为 None）；
    ``progress`` 进度计数器；``device`` 当前设备；``process_index/num_processes``
    即 rank 与总进程数。
    """
    if not _is_out_of_memory_error(exc):
        return  # 非 OOM 异常：交回上层处理，本函数不插手

    message = (
        "Fatal out-of-memory during training. "
        f"stage={stage}, "
        f"epoch={progress.epoch}, "
        f"global_step={progress.global_step}, "
        f"rank={process_index}/{num_processes}. "
        f"{_build_batch_memory_summary(batch)}. "
        f"{_build_cuda_memory_summary(device)}."
    )
    print(message, file=sys.stderr, flush=True)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    sys.stderr.flush()

    if num_processes > 1:
        os._exit(1)  # 多进程：立即硬退出，避免其余 rank 卡在 collective 上死锁


def _is_out_of_memory_error(exc: BaseException) -> bool:
    """Detect a CUDA OOM exception across torch versions / 判定是否为显存 OOM。

    两条判据：① 新版 torch 有专门的 ``torch.OutOfMemoryError`` 类型，直接 isinstance；
    ② 老版本把 OOM 包成普通 ``RuntimeError``，只能靠错误信息里的 "out of memory"
    子串（小写匹配）来识别。``getattr(torch, ..., None)`` 兼容缺失该类型的旧版。
    """
    oom_error_type = getattr(torch, "OutOfMemoryError", None)
    if oom_error_type is not None and isinstance(exc, oom_error_type):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    return "out of memory" in str(exc).lower()  # 旧版本只能靠字符串匹配


def _build_batch_memory_summary(batch: dict[str, object] | None) -> str:
    """Summarise the OOM batch's shapes/lengths for the crash log.

    把触发 OOM 的 batch 概括成一行：序列张量形状、最长 input_ids / audio /
    text token 数。这些"最大长度"通常就是显存峰值的元凶（最长样本决定了
    attention 与 KV cache 的内存），故单列出来便于定位。逐字段用 isinstance
    保护，缺字段就略过，任何字段都没有时返回 "batch=unavailable"。
    """
    if not isinstance(batch, dict):
        return "batch=unavailable"

    fields = []
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        fields.append(f"input_ids_shape={tuple(input_ids.shape)}")
    sample = batch.get("sample")
    if isinstance(sample, torch.Tensor):
        fields.append(f"sample_shape={tuple(sample.shape)}")
    input_ids_lengths = batch.get("input_ids_lengths")
    if isinstance(input_ids_lengths, torch.Tensor) and input_ids_lengths.numel() > 0:
        fields.append(
            f"max_input_ids_length={int(input_ids_lengths.max().detach().item())}"
        )
    num_audio_tokens = batch.get("num_audio_tokens")
    if isinstance(num_audio_tokens, torch.Tensor) and num_audio_tokens.numel() > 0:
        fields.append(f"max_audio_tokens={int(num_audio_tokens.max().detach().item())}")
    num_text_tokens = batch.get("num_text_tokens")
    if isinstance(num_text_tokens, torch.Tensor) and num_text_tokens.numel() > 0:
        fields.append(f"max_text_tokens={int(num_text_tokens.max().detach().item())}")
    return ", ".join(fields) if fields else "batch=unavailable"


def _build_cuda_memory_summary(device: torch.device) -> str:
    """Summarise current/peak CUDA memory (GiB) for the crash log.

    报告当前与峰值显存：``allocated`` 是 PyTorch 实际占用的张量内存，
    ``reserved`` 是 caching allocator 向驱动预留的缓存池（通常更大），
    ``max_*`` 是本进程历史峰值。所有数值除 ``1024**3`` 换算成 GiB。
    非 CUDA 设备或 CUDA 不可用时返回占位串。
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return "device_memory=unavailable"
    allocated = torch.cuda.memory_allocated(device) / (1024**3)  # 转 GiB
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return (
        f"device={device}, "
        f"allocated_gb={allocated:.2f}, "
        f"reserved_gb={reserved:.2f}, "
        f"max_allocated_gb={max_allocated:.2f}, "
        f"max_reserved_gb={max_reserved:.2f}"
    )


# ---------------------------------------------------------------------------
# Debug Helpers
# ---------------------------------------------------------------------------


def build_data_debug_lines(
    batch: dict[str, object],
    *,
    batch_index: int,
    tokenizer: Any,
    sample_rate: int,
) -> list[str]:
    """Build human-readable debug lines describing one data batch.

    把一个 batch 渲染成若干 ``[debug:data]`` 行，用于训练早期/抽样检查数据
    管线是否正确：形状、各样本的 token 长度分布、音频时长、可选的 fbank 与
    loss mask 密度，并对前 3 个样本解码出文本，肉眼核对 text↔audio 对齐与
    特殊 token。纯诊断输出，不参与训练。

    输入约定：``batch`` 至少含 input_ids / input_ids_lengths / sample /
    sample_lengths / num_audio_tokens / num_text_tokens 六个张量字段
    （缺失或非张量会抛 TypeError）；``sample`` 是波形 (B, 1, T_wav) 这类原始音频，
    ``sample_lengths`` 为有效采样点数，配合 ``sample_rate`` 换算成秒。
    返回每行一个字符串的列表，由调用方逐行打印。
    """
    input_ids = batch["input_ids"]
    input_ids_lengths = batch["input_ids_lengths"]
    sample = batch["sample"]
    sample_lengths = batch["sample_lengths"]
    num_audio_tokens = batch["num_audio_tokens"]
    num_text_tokens = batch["num_text_tokens"]

    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        input_ids_lengths, torch.Tensor
    ):
        raise TypeError("Debug batch requires tensor input_ids and input_ids_lengths.")
    if not isinstance(sample, torch.Tensor) or not isinstance(
        sample_lengths, torch.Tensor
    ):
        raise TypeError("Debug batch requires tensor sample and sample_lengths.")
    if not isinstance(num_audio_tokens, torch.Tensor) or not isinstance(
        num_text_tokens, torch.Tensor
    ):
        raise TypeError(
            "Debug batch requires tensor num_audio_tokens and num_text_tokens."
        )

    source_names = batch.get("source_names")
    # 第一行：批级元信息 + 各数据来源(source)的样本计数（dict(Counter(...))）
    debug_lines = [
        (
            "[debug:data] "
            f"batch_index={batch_index} "
            f"batch_size={int(input_ids.size(0))} "
            f"input_ids_shape={tuple(input_ids.shape)} "
            f"sample_shape={tuple(sample.shape)} "
            f"sample_rate={sample_rate} "
            f"sources={dict(Counter(source_names or []))}"
        ),
        (
            "[debug:data] "
            f"input_tokens(min/mean/max)={_format_tensor_triplet(input_ids_lengths)} "
            f"text_tokens(min/mean/max)={_format_tensor_triplet(num_text_tokens)} "
            f"audio_tokens(min/mean/max)={_format_tensor_triplet(num_audio_tokens)} "
            f"audio_seconds(min/mean/max)={_format_audio_seconds_triplet(sample_lengths, sample_rate)}"
        ),
    ]

    fbank = batch.get("fbank")
    fbank_lengths = batch.get("fbank_lengths")
    if isinstance(fbank, torch.Tensor):
        debug_lines.append(
            "[debug:data] "
            f"fbank_shape={tuple(fbank.shape)} "
            f"fbank_frames(min/mean/max)={_format_tensor_triplet(fbank_lengths)}"
        )

    loss_masks = batch.get("loss_masks")
    if isinstance(loss_masks, dict):
        debug_lines.append(
            "[debug:data] "
            "loss_masks="
            + ", ".join(
                f"{name}:{_format_mask_density(mask)}"
                for name, mask in sorted(loss_masks.items())
            )
        )

    fids = batch.get("fids") or []
    # 只抽样前 3 个样本逐条详述，避免大 batch 刷屏；fids 为各样本文件 id
    sample_count = min(int(input_ids.size(0)), 3)
    for sample_idx in range(sample_count):
        input_length = int(input_ids_lengths[sample_idx].item())
        audio_length = int(sample_lengths[sample_idx].item())
        fbank_shape = "unavailable"
        if isinstance(fbank, torch.Tensor) and isinstance(fbank_lengths, torch.Tensor):
            fbank_shape = (
                f"({int(fbank_lengths[sample_idx].item())}, {int(fbank.size(-1))})"
            )
        debug_lines.append(
            "[debug:data] "
            f"sample_index={sample_idx} "
            f"fid={str(fids[sample_idx]) if sample_idx < len(fids) else f'sample_{sample_idx:02d}'} "
            f"source_name={source_names[sample_idx] if source_names else None} "
            f"input_ids_shape=({input_length},) "
            f"sample_shape=(1, {audio_length}) "
            f"fbank_shape={fbank_shape} "
            f"num_text_tokens={int(num_text_tokens[sample_idx].item())} "
            f"num_audio_tokens={int(num_audio_tokens[sample_idx].item())} "
            f"audio_seconds={audio_length / float(sample_rate):.2f} "
            # 解码该样本有效区间 [:input_length] 的 input_ids 回文本，
            # 保留特殊 token（skip_special_tokens=False）以便核对 prompt 结构
            "text="
            f"{tokenizer.decode(input_ids[sample_idx, :input_length].detach().cpu().tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)!r}"
        )
    return debug_lines


def should_print_gradient_debug(
    *,
    debug_enabled: bool,
    is_main_process: bool,
    next_global_step: int,
    log_interval: int,
    early_step_limit: int,
) -> bool:
    """Decide whether to dump gradient diagnostics on this step.

    判定本 step 是否打印梯度调试：必须开了 debug 且是主进程（只在 rank0 打，
    避免重复刷屏），且满足"训练早期前 ``early_step_limit`` 步逐步打"或"之后每
    ``log_interval`` 步打一次"——早期密集观察（最易发现初始化/数据问题）、后期
    稀疏采样。这里用的是 ``next_global_step``（即将完成的那一步）。
    """
    return bool(
        debug_enabled
        and is_main_process  # 仅主进程打印，防止各 rank 重复输出
        and (
            next_global_step <= early_step_limit  # 训练早期：逐步密集打印
            or next_global_step % log_interval == 0  # 之后：按间隔稀疏打印
        )
    )


def build_gradient_debug_lines(
    model: torch.nn.Module,
    *,
    global_step: int,
    grad_norm: float,
    grad_clip_norm: float,
) -> list[str]:
    """Scan per-parameter gradients and build gradient-health debug lines.

    遍历模型所有可训练参数的 ``.grad``，统计梯度健康度，渲染成 ``[debug:grad]``
    行。要排查的典型问题：梯度爆炸/消失、出现 NaN/Inf（混合精度训练常见）、
    某些参数始终没收到梯度（连不进计算图）。

    必须在 ``backward()`` 之后、``optimizer.step()`` 之前调用，此时 ``.grad``
    已填好；通常也在 gradient clipping 之前，故 ``grad_norm`` 标注为
    pre_clip。每个参数算 L2 norm、最大/平均绝对值，并全局汇总：
      - params_with_grad / params_without_grad：有/无梯度的参数计数；
      - nonfinite_param_count + 最多 8 个非有限参数名样本；
      - max_abs_grad / mean_abs_grad：全局最大、按元素加权平均的绝对梯度；
      - top_param_norms：梯度 L2 范数最大的前 6 个参数（最可能是爆炸源头）。
    ``grad_clip_norm`` 用于算 clip_ratio = grad_norm / clip 阈值，>1 表示被裁剪。
    """
    top_param_candidates: list[tuple[str, float, float, float]] = []
    nonfinite_grad_params: list[str] = []
    nonfinite_param_count = 0
    params_with_grad = 0
    params_without_grad = 0
    abs_sum = 0.0
    abs_count = 0
    max_abs_grad = 0.0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue  # 冻结参数（如不训练的 backbone 层）不计入梯度统计
        grad = parameter.grad
        if grad is None:
            params_without_grad += 1  # requires_grad 却拿不到梯度：未参与本次反传
            continue

        grad_tensor = grad.detach().float()  # 脱离计算图并升到 fp32，统计更稳
        params_with_grad += 1
        # 检测 NaN/Inf：混合精度下溢/上溢或损失数值问题的早期信号
        if not bool(torch.isfinite(grad_tensor).all().item()):
            nonfinite_param_count += 1
            if len(nonfinite_grad_params) < 8:  # 只留前 8 个样本名，避免日志过长
                nonfinite_grad_params.append(name)
        grad_abs = grad_tensor.abs()
        param_norm = float(torch.linalg.vector_norm(grad_tensor).item())  # 该参数梯度 L2 范数
        param_max_abs = float(grad_abs.max().item())
        param_mean_abs = float(grad_abs.mean().item())
        max_abs_grad = max(max_abs_grad, param_max_abs)
        # 累积"绝对值之和"与"元素总数"，最后整体平均（按元素加权，非按参数）
        abs_sum += float(grad_abs.sum().item())
        abs_count += int(grad_abs.numel())
        top_param_candidates.append((name, param_norm, param_max_abs, param_mean_abs))

    mean_abs_grad = math.nan if abs_count == 0 else abs_sum / float(abs_count)
    # 按 L2 范数（元组第 1 项）降序取前 6，定位最可能"爆炸"的参数
    top_param_norms = sorted(
        top_param_candidates,
        key=lambda item: item[1],
        reverse=True,
    )[:6]

    debug_lines = [
        (
            "[debug:grad] "
            f"step={global_step} "
            f"pre_clip_grad_norm={format_scalar(grad_norm)} "
            f"clip_ratio={format_scalar(_safe_grad_clip_ratio(grad_norm, grad_clip_norm))} "
            f"params_with_grad={params_with_grad} "
            f"params_without_grad={params_without_grad} "
            f"nonfinite_param_count={nonfinite_param_count} "
            f"max_abs_grad={format_scalar(max_abs_grad)} "
            f"mean_abs_grad={format_scalar(mean_abs_grad)}"
        )
    ]
    if top_param_norms:
        debug_lines.append(
            "[debug:grad] top_params="
            + ", ".join(
                (
                    f"{name}:{param_norm:.4f}"
                    f"(max={param_max_abs:.4e},mean={param_mean_abs:.4e})"
                )
                for name, param_norm, param_max_abs, param_mean_abs in top_param_norms
            )
        )
    if nonfinite_grad_params:
        debug_lines.append(
            "[debug:grad] nonfinite_params=" + ", ".join(nonfinite_grad_params)
        )
    return debug_lines


def _format_tensor_triplet(values: object) -> str:
    """Render a tensor as ``min/mean/max`` (ints for min/max) or ``n/a``.

    把一维长度/计数张量概括成 "最小/均值/最大" 三元组：min、max 取整显示
    （它们本是整数长度），mean 保留两位小数。空张量或非张量返回 "n/a"。
    先 ``.cpu().to(float32)`` 以安全做统计。
    """
    if not isinstance(values, torch.Tensor) or values.numel() == 0:
        return "n/a"
    flattened = values.detach().cpu().to(torch.float32)
    return (
        f"{int(flattened.min().item())}/"
        f"{flattened.mean().item():.2f}/"
        f"{int(flattened.max().item())}"
    )


def _format_audio_seconds_triplet(values: object, sample_rate: int) -> str:
    """Render sample-count lengths as ``min/mean/max`` seconds or ``n/a``.

    与 :func:`_format_tensor_triplet` 同形，但先把"采样点数"除以
    ``sample_rate`` 换算成秒，三个量都保留两位小数，用于直观看清 batch 内
    音频时长分布（找出过长样本）。
    """
    if not isinstance(values, torch.Tensor) or values.numel() == 0:
        return "n/a"
    seconds = values.detach().cpu().to(torch.float32) / float(sample_rate)  # 采样点数→秒
    return (
        f"{seconds.min().item():.2f}/"
        f"{seconds.mean().item():.2f}/"
        f"{seconds.max().item():.2f}"
    )


def _format_mask_density(mask: object) -> str:
    """Render a loss mask's density as ``active/total`` or ``n/a``.

    loss mask 密度：用 ``> 0`` 计数有效（参与 loss）的位置，输出 "有效数/总数"。
    一眼看出该 mask 实际监督了多少 token——若有效数为 0，说明该项 loss 这批
    没生效，往往是数据/mask 构造出了问题。
    """
    if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
        return "n/a"
    return f"{int(mask.detach().gt(0).sum().item())}/{int(mask.numel())}"


def _safe_grad_clip_ratio(grad_norm: float, grad_clip_norm: float) -> float:
    """Compute ``grad_norm / clip_threshold``, returning NaN if grad is non-finite.

    裁剪比 = 总梯度范数 / 裁剪阈值；>1 表示本步触发了 gradient clipping。
    若 ``grad_norm`` 已是 NaN/Inf（梯度炸了），直接返回 nan 而不做除法，
    避免传播无意义数值。
    """
    if not math.isfinite(grad_norm):
        return math.nan
    return grad_norm / float(grad_clip_norm)


# ---------------------------------------------------------------------------
# Step Reporting
# ---------------------------------------------------------------------------


def should_log_training_step(global_step: int, log_interval: int) -> bool:
    """Return True every ``log_interval`` steps / 每隔 log_interval 步记一次日志。"""
    return global_step % log_interval == 0


def reduce_source_metrics(
    source_loss_totals: dict[str, dict[str, float]],
    source_loss_denominators: dict[str, dict[str, float]],
    *,
    device: torch.device,
    loss_config: Any,
) -> dict[str, dict[str, float]]:
    """Reduce per-source loss accumulators across ranks into final metrics.

    按数据来源(source)聚合损失：训练时各 rank 分别累积了
    ``source -> metric -> 分子(loss 加权和)`` 与对应 ``分母(归一化项)``。这里先
    把两套嵌套累加器各自跨 rank ``all_reduce(SUM)`` 汇总成全局量，再交给
    :func:`losses.reduce_loss_statistics_by_source` 逐 source 做"分子/分母"
    归一，得到每个来源的最终标量指标。

    这样按来源拆分，便于观察多数据集混训时各源各自的 loss（哪个数据集学得好/
    差）。返回 ``{source_name: {metric: value}}``。
    """
    reduced_source_totals = loss_ops.sum_grouped_named_scalars_across_ranks(
        source_loss_totals,
        device=device,
    )
    reduced_source_denominators = loss_ops.sum_grouped_named_scalars_across_ranks(
        source_loss_denominators,
        device=device,
    )
    return loss_ops.reduce_loss_statistics_by_source(
        reduced_source_totals,
        reduced_source_denominators,
        loss_config=loss_config,
    )


def build_train_step_report(
    metrics: dict[str, Any],
    *,
    learning_rate: float,
    grad_norm: float,
    current_time: float,
    last_log_step: int,
    last_log_time: float,
    progress: TrainProgress,
    max_train_steps: int,
    reduced_by_source: dict[str, dict[str, float]],
) -> TrainStepReport:
    """Assemble one step's :class:`TrainStepReport` (tracker dict + console line).

    汇总一次 step 的报告：先根据"自上次日志以来经过的步数/墙钟时间"算出
    吞吐 ``steps_per_second`` 与剩余时间 ``eta_seconds``，再把指标分别交给
    :func:`build_train_log_dict`（给 tracker 的扁平 dict）与
    :func:`format_train_line`（console 单行）打包进 :class:`TrainStepReport`。

    数值稳健性：步数<=0 或耗时<=0 时吞吐取 nan（避免除零 / 首次记录无基准）；
    吞吐非有限或<=0 时 ETA 也取 nan。``last_log_step``/``last_log_time`` 是上次
    记日志时的步数与时间戳。
    """
    logged_steps = progress.global_step - last_log_step  # 自上次日志以来完成的步数
    elapsed = current_time - last_log_time  # 自上次日志以来的墙钟秒数
    steps_per_second = (
        math.nan
        if logged_steps <= 0 or elapsed <= 0.0  # 无有效区间则吞吐不可定义
        else float(logged_steps) / elapsed
    )
    eta_seconds = (
        math.nan
        if not math.isfinite(steps_per_second) or steps_per_second <= 0.0
        # 剩余步数 / 当前吞吐 = 预计剩余秒数
        else float(max_train_steps - progress.global_step) / steps_per_second
    )
    return TrainStepReport(
        log_values=build_train_log_dict(
            metrics,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            steps_per_second=steps_per_second,
            eta_seconds=eta_seconds,
            progress=progress,
            reduced_by_source=reduced_by_source,
        ),
        console_line=format_train_line(
            metrics,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            steps_per_second=steps_per_second,
            eta_seconds=eta_seconds,
            progress=progress,
            max_train_steps=max_train_steps,
            reduced_by_source=reduced_by_source,
        ),
    )


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------


def flatten_config(values, parent_key="", sep="/"):
    """Flatten a nested config dict into ``path/to/key -> value`` pairs.

    把嵌套配置字典递归展平成单层："a"->{"b":1} 变成 "a/b"->1。便于把整套
    超参一次性记录进 experiment tracker（多数 tracker 只接受扁平 dict）。
    list/tuple 与 None 转成字符串以保证值可序列化；其余标量原样保留。
    ``parent_key``/``sep`` 仅供递归拼接路径前缀，外部调用通常用默认值。
    """
    items = []
    for key, value in values.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key, sep).items())
        elif isinstance(value, (list, tuple)):
            items.append((new_key, str(value)))
        elif value is None:
            items.append((new_key, "None"))
        else:
            items.append((new_key, value))
    return dict(items)


def format_scalar(value: float) -> str:
    """Format a scalar for concise console logging.

    标量的紧凑显示：非有限(NaN/Inf)统一显示 "nan"；整数去掉多余小数（如 3.0→"3"）；
    其余保留 4 位小数。让 console 日志整齐易读。
    """
    if not math.isfinite(value):
        return "nan"
    if float(value).is_integer():
        return str(int(value))  # 整数值省略小数位
    return f"{value:.4f}"


def _format_eta(eta_seconds: float) -> str:
    """Render ETA seconds as ``HH:MM:SS`` or ``n/a``.

    把"预计剩余秒数"格式化为 HH:MM:SS。非有限或负数（尚无吞吐估计/已超步数）
    显示 "n/a"。各段补零到两位。
    """
    if not math.isfinite(eta_seconds) or eta_seconds < 0.0:
        return "n/a"
    total_seconds = int(round(eta_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_train_log_dict(
    metrics: dict[str, Any],
    *,
    learning_rate: float,
    grad_norm: float,
    steps_per_second: float,
    eta_seconds: float,
    progress: TrainProgress,
    reduced_by_source: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Build the flat metric dict sent to experiment trackers.

    组装发往 tracker 的扁平训练指标字典：固定项（epoch、lr、grad_norm、吞吐、
    ETA、累计消费的总/音频/文本 token）+ 本步各项 ``metrics`` + 各数据来源的
    分项指标。键统一加 ``train/`` 前缀，按来源再加一层 ``train/<source>/<name>``，
    所有值强制转 float 以满足 tracker 的数值要求。
    """
    log_dict = {
        "train/epoch": float(progress.epoch),
        "train/learning_rate": learning_rate,
        "train/grad_norm": grad_norm,
        "train/steps_per_second": steps_per_second,
        "train/eta_seconds": eta_seconds,
        "train/consumed_tokens": float(progress.total_tokens),
        "train/consumed_audio_tokens": float(progress.audio_tokens),
        "train/consumed_text_tokens": float(progress.text_tokens),
    }
    for name, value in metrics.items():
        log_dict[f"train/{name}"] = float(value)
    for source_name, source_metrics in reduced_by_source.items():
        log_dict.update(
            {
                f"train/{source_name}/{name}": float(value)
                for name, value in source_metrics.items()
            }
        )
    return log_dict


def format_train_line(
    metrics: dict[str, Any],
    *,
    learning_rate: float,
    grad_norm: float,
    steps_per_second: float,
    eta_seconds: float,
    progress: TrainProgress,
    max_train_steps: int,
    reduced_by_source: dict[str, dict[str, Any]],
) -> str:
    """Build a single human-readable console line for one training step.

    把一步训练拼成一行 ``" | "`` 分隔的 console 日志：先固定字段（迭代进度、
    epoch、消费 token、lr、吞吐、ETA、grad_norm），再按字母序列出除 "loss"
    外的各指标，最后单独把 "loss" 放到末尾（最关注的量压轴），随后同样规则
    追加每个数据来源的分项指标。
    """
    parts = [
        f"iteration {progress.global_step}/{max_train_steps}",
        f"epoch: {progress.epoch}",
        f"consumed_tokens: {progress.total_tokens}",
        f"consumed_audio_tokens: {progress.audio_tokens}",
        f"consumed_text_tokens: {progress.text_tokens}",
        f"learning_rate: {learning_rate:.2e}",
        f"steps_per_second: {format_scalar(steps_per_second)}",
        f"job_eta: {_format_eta(eta_seconds)}",
        f"grad_norm: {format_scalar(grad_norm)}",
    ]
    # 除 loss 外的指标按名排序输出，保证每步字段顺序稳定、便于扫读对比
    for name in sorted(name for name in metrics if name != "loss"):
        parts.append(f"{name}: {format_scalar(float(metrics[name]))}")
    if "loss" in metrics:
        parts.append(f"loss: {format_scalar(float(metrics['loss']))}")  # loss 压轴
    for source_name, source_metrics in reduced_by_source.items():
        for name in sorted(name for name in source_metrics if name != "loss"):
            parts.append(
                f"{source_name}_{name}: {format_scalar(float(source_metrics[name]))}"
            )
        if "loss" in source_metrics:
            parts.append(
                f"{source_name}_loss: {format_scalar(float(source_metrics['loss']))}"
            )
    return " | ".join(parts)


def build_validation_log_dict(
    metrics: dict[str, Any],
    *,
    reduced_by_source: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Build the flat validation metric dict sent to experiment trackers.

    验证集版本的扁平指标字典：与 :func:`build_train_log_dict` 同构，但键前缀
    是 ``val/``（含按来源的 ``val/<source>/<name>``），且不带 lr/吞吐等训练态
    字段——验证只汇报损失/指标本身。
    """
    log_dict = {f"val/{name}": float(value) for name, value in metrics.items()}
    for source_name, source_metrics in reduced_by_source.items():
        log_dict.update(
            {
                f"val/{source_name}/{name}": float(value)
                for name, value in source_metrics.items()
            }
        )
    return log_dict


def format_validation_line(
    metrics: dict[str, Any],
    *,
    global_step: int,
    reduced_by_source: dict[str, dict[str, Any]],
) -> str:
    """Build the console summary line printed after a validation pass.

    验证结束后的 console 汇总行：以 "validation at iteration <step>" 开头，
    其后与 :func:`format_train_line` 同样规则——非 loss 指标按名排序、loss 压轴，
    再追加各来源分项，``" | "`` 拼接成一行。
    """
    parts = [f"validation at iteration {global_step}"]
    for name in sorted(name for name in metrics if name != "loss"):
        parts.append(f"{name}: {format_scalar(float(metrics[name]))}")
    if "loss" in metrics:
        parts.append(f"loss: {format_scalar(float(metrics['loss']))}")
    for source_name, source_metrics in reduced_by_source.items():
        for name in sorted(name for name in source_metrics if name != "loss"):
            parts.append(
                f"{source_name}_{name}: {format_scalar(float(source_metrics[name]))}"
            )
        if "loss" in source_metrics:
            parts.append(
                f"{source_name}_loss: {format_scalar(float(source_metrics['loss']))}"
            )
    return " | ".join(parts)


__all__ = [
    "TrainProgress",
    "TrainStepReport",
    "abort_on_out_of_memory",
    "any_rank_true",
    "build_data_debug_lines",
    "build_gradient_debug_lines",
    "build_train_step_report",
    "build_train_log_dict",
    "build_validation_log_dict",
    "flatten_config",
    "format_scalar",
    "format_train_line",
    "format_validation_line",
    "move_to_device",
    "reduce_source_metrics",
    "should_log_training_step",
    "should_print_gradient_debug",
    "sum_integer_counters_across_ranks",
]
