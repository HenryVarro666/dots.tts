"""Checkpoint helpers for distributed dots_tts training.

This module persists not only model/optimizer/scheduler state, but also
rank-local RNG state and data-loader progress so resumed training can continue
from the same point with minimal drift.

中文说明 (本文件做什么)
======================
分布式训练的 checkpoint 读写工具。除了常规的 model / optimizer / scheduler
权重之外, 它额外持久化两类"每个 rank 各不相同"的状态:
  1. RNG state —— Python random / NumPy / PyTorch(含每张 GPU 的 cuda RNG),
     用于 **确定性 resume**(deterministic resume): 续训时各 rank 恢复到与中断
     前完全一致的随机数序列, 避免 dropout / 数据增强 / 采样发生漂移(drift)。
  2. data-loader 进度 —— 每个 rank 读到自己数据分片(shard)的哪个位置。

在训练数据流中的位置
--------------------
这是**训练侧(training)**的基础设施, 与推理(inference)无关。训练循环每隔
若干步调用 ``save_train_checkpoint`` 落盘; 重启时先用
``resolve_latest_train_checkpoint`` 找到最新目录, 再用 ``load_train_checkpoint``
把全部状态灌回内存, 从中断处接着训练 dots.tts 的 Qwen2.5 AR 主干 +
flow-matching DiT 声学头。

关键函数清单
------------
- ``resolve_latest_train_checkpoint`` : 解析 ``latest`` symlink / 最大编号目录。
- ``save_train_checkpoint`` / ``load_train_checkpoint`` : 全量保存 / 恢复入口。
- ``_rng_state`` / ``_restore_rng_state`` : 捕获 / 还原跨库 RNG 状态。
- ``_pack_rank_payload`` / ``_extract_rank_payload`` : 把 rank-local 状态在主进程
  汇总成 ``{world_size, per_rank}``, 续训时再按当前 rank 取回。
- ``_replace_latest_symlink`` / ``_cleanup_old_checkpoints`` : 原子更新 symlink、
  清理旧 ckpt。
- ``_checkpoint_dir`` / ``_checkpoint_entries`` : 目录命名约定与枚举。
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def _checkpoint_dir(log_dir: str, step: int) -> Path:
    """Return the canonical directory name for a training step checkpoint.

    返回某个训练步对应的标准 checkpoint 目录名。step 用 ``08d`` 零填充
    (例如 ``checkpoint-00012000``), 这样字典序排序天然等于数值序, 便于
    ``glob`` + 文本排序找到最新 ckpt。
    """
    return Path(log_dir) / f"checkpoint-{step:08d}"


def _checkpoint_entries(log_dir: str) -> list[tuple[int, Path]]:
    """List valid ``checkpoint-*`` directories sorted by step number.

    枚举 ``log_dir`` 下所有合法的 ``checkpoint-<step>`` 目录, 返回按 step 升序
    排好的 ``(step, path)`` 列表; 列表末尾即最新 ckpt。``save``/``cleanup``/
    ``resolve_latest`` 都复用它作为单一事实来源(single source of truth)。
    """
    entries = []
    for path in Path(log_dir).glob("checkpoint-*"):
        if not path.is_dir():
            continue
        # 只接受后缀为纯数字的目录, 借此过滤掉 ``checkpoint-*.tmp`` 等半成品 /
        # 非法目录, 避免把它们当成有效 ckpt 参与排序或被清理逻辑误删。
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            entries.append((int(suffix), path))
    return sorted(entries)


def resolve_latest_train_checkpoint(log_dir: str) -> Path:
    """Resolve the checkpoint directory that should be used for resume.

    Preference order:
    1. ``<log_dir>/latest`` symlink, if present.
    2. The numerically largest ``checkpoint-*`` directory.

    决定续训(resume)应当加载哪个 checkpoint 目录。优先用 ``latest`` symlink
    (由 ``_replace_latest_symlink`` 维护), 它是"上一次成功保存"的权威指针;
    若 symlink 缺失则回退到编号最大的目录。找不到任何 ckpt 抛
    ``FileNotFoundError``。返回的是 ``resolve(strict=True)`` 后的真实路径
    (解开 symlink、要求路径必须存在)。
    """
    latest_path = Path(log_dir) / "latest"
    # ``is_symlink()`` 与 ``exists()`` 并列判断: 若 symlink 指向已被删除的目标,
    # ``exists()`` 会返回 False(它会跟随 symlink), 这里仍需识别到 symlink 本身。
    if latest_path.exists() or latest_path.is_symlink():
        return latest_path.resolve(strict=True)

    entries = _checkpoint_entries(log_dir)
    if not entries:
        raise FileNotFoundError(
            f"No checkpoint found under {log_dir!s}; expected latest or checkpoint-*."
        )
    return entries[-1][1].resolve(strict=True)


def _rng_state() -> dict:
    """Capture Python/NumPy/PyTorch RNG state for deterministic resume.

    捕获本进程(本 rank)的三套随机数发生器状态, 用于确定性 resume:
    Python ``random``、NumPy、以及 PyTorch(CPU RNG; 若有 GPU 还含每张卡的
    cuda RNG)。返回的 dict 之后会经 ``_pack_rank_payload`` 跨 rank 汇总并
    ``torch.save`` 落盘。

    为什么 NumPy 要拆成字段:``np.random.get_state()`` 返回一个混杂 tuple
    (字符串 + ``uint32`` ndarray + 若干标量), 直接序列化跨版本不稳。这里显式
    把它拆成 ``bit_generator / keys / pos / has_gauss / cached_gaussian`` 五项纯
    Python/list 标量, 既便于 JSON 友好的存储, 也避免 dtype 漂移; 还原时由
    ``_restore_rng_state`` 逐字段重组。
    """
    numpy_state = np.random.get_state()
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].tolist(),  # uint32 ndarray -> list, 便于序列化
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),  # 是否有缓存的高斯样本(Box-Muller 余项)
            "cached_gaussian": float(numpy_state[4]),
        },
    }
    if torch.cuda.is_available():
        # get_rng_state_all(): 收集 *所有* 可见 GPU 的 RNG, 返回 state 列表,
        # 与 set_rng_state_all 一一对应, 保证多卡续训各卡 RNG 都精确复原。
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore RNG state previously produced by :func:`_rng_state`.

    ``_rng_state`` 的逆操作: 把已拆解的 RNG 字段重新装回 PyTorch / Python /
    NumPy。注意 NumPy 这一步必须把 ``keys`` 还原成 ``uint32`` ndarray 并复原成
    与 ``get_state()`` 同构的 5 元 tuple, 否则 ``set_state`` 会报错。多卡时按
    保存顺序逐卡复原 cuda RNG。
    """
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),  # list -> uint32 ndarray
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _replace_latest_symlink(log_dir: str, save_dir: Path) -> None:
    """Atomically refresh the ``latest`` symlink to point at ``save_dir``.

    原子地把 ``latest`` symlink 重新指向最新 ckpt 目录。技巧: 先建一个临时
    symlink ``latest.tmp``, 再用 ``rename`` 覆盖 ``latest`` —— ``rename`` 在
    POSIX 上是原子操作, 任何并发读取者要么看到旧 ``latest`` 要么看到新
    ``latest``, 绝不会撞见"指针不存在"的中间态。

    symlink 用相对名(``save_dir.name``)而非绝对路径, 这样整个 ``log_dir`` 被
    拷贝 / 移动到别处后, ``latest`` 依然能正确解析到同目录下的 ckpt。
    """
    log_path = Path(log_dir)
    link_path = log_path / "latest"
    tmp_link_path = log_path / "latest.tmp"

    # 清掉可能残留的上次中断遗留的临时 symlink, 再新建指向本次 save_dir。
    if tmp_link_path.exists() or tmp_link_path.is_symlink():
        tmp_link_path.unlink()
    tmp_link_path.symlink_to(save_dir.name)  # 相对指向: 仅目录名, 保证可迁移

    if link_path.exists() or link_path.is_symlink():
        # 兼容历史遗留: 若 ``latest`` 曾被写成真实目录而非 symlink, 需用
        # rmtree 整目录删除; 否则按普通 symlink/文件 unlink。
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    tmp_link_path.rename(link_path)  # 原子替换: latest.tmp -> latest


def _cleanup_old_checkpoints(log_dir: str, keep_max: int) -> None:
    """Delete older checkpoints while keeping the newest ``keep_max`` ones.

    保留最新 ``keep_max`` 个 ckpt, 删除更旧的, 控制磁盘占用。``keep_max <= 0``
    视为"无上限/不清理"直接返回。entries 已按 step 升序, 切片 ``[:-keep_max]``
    精确取出"除末尾 keep_max 个外"的旧目录。``ignore_errors=True`` 让单个目录
    删除失败(例如被占用)不会中断训练主流程。
    """
    if keep_max <= 0:
        return
    for _, path in _checkpoint_entries(log_dir)[:-keep_max]:
        shutil.rmtree(path, ignore_errors=True)


def _pack_rank_payload(accelerator, payload: dict, *, payload_name: str) -> dict | None:
    """Collect rank-local payloads onto the main process for checkpointing.

    Some training state is intentionally local to each rank, for example RNG
    state or data-loader shard progress. We therefore gather a per-rank payload
    and store it in the checkpoint as ``{world_size, per_rank}``.

    把"每个 rank 各不相同"的状态(RNG state、data-loader 分片进度)用
    ``all_gather_object`` 汇总到主进程, 打包成 ``{world_size, per_rank}`` 一并存盘
    —— 只有主进程真正写文件, 所以非主进程做完 gather 后返回 ``None``。

    参数:
        accelerator : HuggingFace Accelerate 的 Accelerator, 提供 ``process_index``
            (本 rank 号)、``num_processes``(world size)、``is_main_process``。
        payload     : 本 rank 要保存的状态 dict。
        payload_name: 仅用于报错信息, 标明出错的是哪类 payload。

    返回:
        主进程 -> ``{"world_size": N, "per_rank": {"0": ..., "1": ...}}``;
        非主进程 -> ``None``。per_rank 的 key 用字符串化的 rank 号, 因为之后会
        ``torch.save`` 并可能经 JSON 友好的容器流转, 字符串 key 更稳。
    """
    local_payload = {
        "rank": int(accelerator.process_index),
        "payload": payload,
    }
    if dist.is_available() and dist.is_initialized():
        # 多进程: 预分配 world_size 大小的占位列表, all_gather_object 会把每个
        # rank 的 local_payload 原地填进对应槽位(按 rank 排序)。
        gathered: list[dict | None] = [None] * int(accelerator.num_processes)
        dist.all_gather_object(gathered, local_payload)
    else:
        # 单进程 / 未初始化分布式: 无需通信, gathered 就是本 rank 自身。
        gathered = [local_payload]

    if not accelerator.is_main_process:
        return None

    per_rank = {}
    for item in gathered:
        # gather 槽位若仍是 None(某 rank 未成功上报), 说明汇总失败, 直接报错,
        # 避免存出一个缺 rank 的、续训时无法对齐的残缺 checkpoint。
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Failed to gather rank-scoped {payload_name} for checkpointing."
            )
        per_rank[str(int(item["rank"]))] = item["payload"]
    return {
        "world_size": len(gathered),
        "per_rank": per_rank,
    }


def _extract_rank_payload(
    accelerator, payload: dict | None, *, payload_name: str
) -> dict:
    """Recover the payload for the current rank from a packed checkpoint blob.

    ``_pack_rank_payload`` 的逆操作: 从 ``{world_size, per_rank}`` 包中取出
    **当前 rank** 那一份。``payload is None`` 时返回空 dict(例如该类状态本就没存)。

    强校验 world_size 必须与当前进程数一致: rank-local 的 RNG / 数据进度是按
    "几卡训"严格绑定的, 若续训时改了卡数, per_rank 映射无法一一对应, 强行恢复
    会破坏确定性, 因此宁可直接报错。缺失本 rank 的条目同理报错。
    """
    if payload is None:
        return {}

    expected_world_size = int(accelerator.num_processes)
    if int(payload["world_size"]) != expected_world_size:
        raise RuntimeError(
            f"Checkpoint {payload_name} payload does not match the current world."
        )

    local_rank = str(int(accelerator.process_index))  # 与 pack 端一致用字符串 key
    if local_rank not in payload["per_rank"]:
        raise RuntimeError(f"Checkpoint {payload_name} is missing rank {local_rank}.")
    return payload["per_rank"][local_rank]


def save_train_checkpoint(
    accelerator,
    model,
    optimizer,
    progress,
    log_dir: str,
    keep_max: int,
    data_state: dict,
    scheduler_state: dict,
) -> None:
    """Save a full resumable training checkpoint.

    Stored artifacts include:
    - model weights in ``save_pretrained`` format
    - optimizer / scheduler / scaler state
    - training progress counters
    - rank-local RNG state
    - rank-local data pipeline state

    保存一份"可完整 resume"的训练 checkpoint。落盘内容:
    ``model/``(HF save_pretrained 权重)、``optimizer.pt``、``scheduler.pt``、
    ``scaler.pt``(AMP GradScaler, 无则存空 dict)、``rng_state.pt``、
    ``data_state.pt``、``trainer_state.json``(进度计数器)。

    参数:
        accelerator    : Accelerate 句柄, 负责进程同步与 unwrap_model。
        model          : 被 Accelerate 包装过的模型(DDP/FSDP 等)。
        optimizer      : 优化器。
        progress       : dataclass 形式的进度对象, 各字段是 int 计数器
                         (如 global_step), 通过 ``dataclasses.fields`` 遍历存取。
        log_dir        : checkpoint 根目录。
        keep_max       : 最多保留几个 ckpt, 多余的旧目录会被清理。
        data_state     : 本 rank 的数据管线进度(rank-local, 需跨 rank 打包)。
        scheduler_state: 调度器状态(由调用方组织, 含 ``state_dict`` 等)。

    设计要点(确定性 + 原子性):
        1. 先 ``wait_for_everyone`` 对齐所有 rank, 保证抓取的 RNG / 数据进度同属
           同一训练步。
        2. RNG / 数据进度是 rank-local 的, 必须 **在所有进程上** 调用打包函数
           (内含集合通信 all_gather), 故这两步在主进程判断 *之前* 执行。
        3. 真正写文件只在主进程, 且先写入 ``*.tmp`` 临时目录再 rename, 杜绝中断
           留下"看起来有效实则半成品"的 ckpt。
    """
    accelerator.wait_for_everyone()  # 屏障: 确保各 rank 处于同一步再抓状态
    # 注意: 下面两次 _pack_rank_payload 含 all_gather, 必须由所有 rank 共同执行,
    # 因此放在 is_main_process 分支外面; 非主进程会拿到 None。
    packed_data_state = _pack_rank_payload(
        accelerator,
        data_state,
        payload_name="data_state",
    )
    packed_rng_state = _pack_rank_payload(
        accelerator,
        _rng_state(),
        payload_name="rng_state",
    )

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)  # 剥掉 DDP/FSDP 包装再存
        save_dir = _checkpoint_dir(log_dir, progress.global_step)
        tmp_dir = save_dir.with_name(f"{save_dir.name}.tmp")  # 同名加 .tmp 后缀
        model_dir = tmp_dir / "model"
        scaler = getattr(accelerator, "scaler", None)  # 可能没启用 AMP, 故用 getattr

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)  # 清掉上次中断残留的 .tmp, 从干净状态开始
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Write into a temporary directory first so interrupted saves never
            # leave behind a half-written checkpoint that looks valid.
            unwrapped_model.save_pretrained(model_dir)

            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            torch.save(scheduler_state, tmp_dir / "scheduler.pt")
            # 未启用 AMP 时 scaler 为 None, 存空 dict 占位, 让加载侧逻辑统一。
            torch.save(
                {} if scaler is None else scaler.state_dict(),
                tmp_dir / "scaler.pt",
            )
            torch.save(packed_rng_state, tmp_dir / "rng_state.pt")
            torch.save(packed_data_state, tmp_dir / "data_state.pt")
            # 进度计数器存为人类可读 JSON: 遍历 dataclass 字段, 全部转成 int。
            (tmp_dir / "trainer_state.json").write_text(
                json.dumps(
                    {
                        field.name: int(getattr(progress, field.name))
                        for field in fields(progress)
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # 全部文件写完后才把 .tmp 整体 rename 成正式目录, 这一步是"提交点":
            # 之前任何中断都只污染 .tmp, 不会产生半成品 ckpt。
            if save_dir.exists():
                shutil.rmtree(save_dir)
            tmp_dir.rename(save_dir)
            _replace_latest_symlink(log_dir, save_dir)  # 原子更新 latest 指针
            _cleanup_old_checkpoints(log_dir, keep_max)  # 删旧 ckpt 控磁盘
            accelerator.print(f"Checkpoint saved: {save_dir}")
        except Exception:
            # 任意写入步骤失败: 清掉残缺的 .tmp 再向上抛, 保证目录始终干净。
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    accelerator.wait_for_everyone()  # 屏障: 非主进程在此等主进程写完再继续训练


def load_train_checkpoint(
    accelerator,
    model,
    optimizer,
    progress,
    checkpoint_dir: str | Path,
    scheduler,
) -> dict:
    """Restore a checkpoint previously written by :func:`save_train_checkpoint`.

    Returns auxiliary state that the caller usually needs to resume the input
    pipeline and scheduler bookkeeping.

    把 ``save_train_checkpoint`` 写的目录原地恢复到内存: 加载模型权重、
    optimizer / scheduler / scaler 状态, 还原本 rank 的 RNG, 并把进度计数器
    灌回 ``progress`` 对象(就地 setattr)。所有 ``torch.load`` 都用
    ``map_location="cpu"``, 先落到 CPU 再由后续训练逻辑迁回 GPU, 避免反序列化时
    直接绑死某张卡导致跨设备 / 跨卡数 resume 失败。

    返回:
        一个 dict, 提供调用方接着续训所需的辅助状态:
        ``checkpoint_dir`` 实际加载的目录、``data_state`` 本 rank 的数据管线进度
        (用于让 data-loader 跳回原分片位置)、``scheduler_state`` 调度器原始 payload
        (调用方可能还要读其中的额外字段)。
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_dir = checkpoint_dir / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint model directory not found: {model_dir!s}")

    accelerator.wait_for_everyone()  # 屏障: 各 rank 同步进入加载阶段

    unwrapped_model = accelerator.unwrap_model(model)  # 往未包装的底层模型灌权重
    unwrapped_model.load_pretrained_weights(model_dir)

    optimizer.load_state_dict(
        torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu")
    )

    scheduler_payload = torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu")
    scheduler.load_state_dict(scheduler_payload["state_dict"])  # 只取标准 state_dict

    scaler = getattr(accelerator, "scaler", None)
    scaler_state = torch.load(checkpoint_dir / "scaler.pt", map_location="cpu")
    # 仅当本次确实启用了 AMP(scaler 非 None)且 ckpt 存了非空 scaler 状态才恢复;
    # 与 save 端"无 AMP 存空 dict"的约定对称, 允许 AMP 开关在两次运行间变化。
    if scaler is not None and scaler_state:
        scaler.load_state_dict(scaler_state)

    rng_state_payload = torch.load(checkpoint_dir / "rng_state.pt", map_location="cpu")
    # 从打包的全 rank RNG 中取出本 rank 那份再还原, 实现确定性 resume。
    _restore_rng_state(
        _extract_rank_payload(
            accelerator,
            rng_state_payload,
            payload_name="rng_state",
        )
    )
    data_state_payload = torch.load(
        checkpoint_dir / "data_state.pt", map_location="cpu"
    )

    trainer_state = json.loads(
        (checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")
    )
    # 就地把每个进度计数器写回 progress dataclass(原对象被原地改, 调用方持有同一引用)。
    for field in fields(progress):
        setattr(progress, field.name, int(trainer_state[field.name]))

    accelerator.wait_for_everyone()  # 屏障: 全部 rank 加载完再一起恢复训练
    return {
        "checkpoint_dir": checkpoint_dir,
        "data_state": _extract_rank_payload(
            accelerator,
            data_state_payload,
            payload_name="data_state",
        ),
        "scheduler_state": scheduler_payload,
    }
