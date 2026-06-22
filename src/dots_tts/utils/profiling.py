"""逐模块计时 / RTF profiling 工具 —— Per-stage timing & real-time-factor profiling.

本文件提供两套互补的性能埋点(profiling)工具，覆盖 dots.tts 的两条数据流：

1) `DataProfiler` —— 面向**训练时的数据流水线(data pipeline)**。
   - 数据加载/特征抽取通常跑在多个子进程(worker)里，所以它用一个跨进程的
     `multiprocessing.Queue` 把每段 `ProfileEvent`(阶段名、耗时、计数、pid)汇报回主进程，
     由主进程聚合。`measure()` 只用 `time.perf_counter()` 计墙钟时间(wall-clock)，
     不做 CUDA 同步——数据 worker 一般是纯 CPU 工作(load/resample/tokenize/fbank)。
   - `enabled` 由是否传入 queue 决定；未启用时 `measure()` 是零开销的空壳。

2) `InferenceProfiler` —— 面向**推理时的逐模块计时**，单进程、按固定的
   `INFERENCE_STAGE_NAMES` 分桶累加。它在每次 `measure()` 前后做
   `torch.cuda.synchronize()`，否则 GPU kernel 异步执行会让计时严重失真。
   `summary(duration_seconds=...)` 把每个阶段耗时除以音频时长得到 **RTF(real-time
   factor)** ——RTF<1 表示比实时还快，是 TTS 推理的核心性能指标。

为了让调用方(model.py 里散布的 `with measure_inference("LLM"): ...`)不必层层传递
profiler 句柄，这里用 `ContextVar` (`_CURRENT_INFERENCE_PROFILER`) 把"当前活跃的
InferenceProfiler"挂在上下文里，配合 `inference_profiling`/`activate_inference_profiler`
两个上下文管理器进出作用域；`measure_inference()` 是真正给业务代码用的轻量门面。

关键符号清单：
  - `INFERENCE_STAGE_NAMES` / `normalize_inference_stage_name`：合法推理阶段名 + 归一化。
  - `DataProfiler` / `ProfileEvent` / `ensure_data_profiler`：跨进程数据流水线埋点。
  - `InferenceProfiler` / `InferenceStageStat`：推理逐阶段计时与累加状态。
  - `inference_profiling` / `activate_inference_profiler` / `measure_inference`：
    基于 ContextVar 的"激活 + 取用"上下文管理器三件套。
  - `log_inference_profile`：把 summary 结果格式化打到 loguru 日志(含 RTF)。
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Iterator

import torch
from loguru import logger

# 推理流水线里被单独计时的 7 个模块。对应 dots.tts 的核心组件：
#   FM           —— flow-matching DiT 声学头(velocity field 预测 + ODE 采样)
#   latent_encoder / latent_decoder —— BigVGAN AudioVAE 在连续 latent 空间的编/解码
#   patch_encoder —— 把 latent 切成 patch 喂给 LLM 的前置编码
#   LLM          —— Qwen2.5-1.5B 自回归主干(含 KV cache 的逐步生成)
#   speaker_encoder —— CAM++ 抽取 x-vector(说话人/音色条件)
#   vocoder      —— 由 latent 还原 24/48kHz 波形的声码器
# 这是一个有序 tuple，model.py 用字符串名(如 "LLM")来标记 `with measure_inference(...)`。
INFERENCE_STAGE_NAMES = (
    "FM",
    "latent_encoder",
    "patch_encoder",
    "LLM",
    "latent_decoder",
    "speaker_encoder",
    "vocoder",
)

# 小写名 -> 规范名 的查表，让阶段名匹配大小写不敏感(调用方写 "llm"/"LLM" 都行)。
_INFERENCE_STAGE_NAME_MAP = {
    name.lower(): name for name in INFERENCE_STAGE_NAMES
}
# 当前线程/任务上下文中"活跃的"InferenceProfiler。用 ContextVar 而非全局变量，
# 是为了让 measure_inference() 在深层调用栈里能就近取到 profiler，且并发安全
# (asyncio 任务 / 线程间互不串扰)。默认 None 表示未开启 profiling。
_CURRENT_INFERENCE_PROFILER: ContextVar[InferenceProfiler | None] = ContextVar(
    "current_inference_profiler",
    default=None,
)


def normalize_inference_stage_name(name: str) -> str:
    """把任意大小写/带空格的阶段名归一化为 INFERENCE_STAGE_NAMES 里的规范名。

    Args:
        name: 调用方传入的阶段名，如 " llm "、"FM"。
    Returns:
        规范化后的阶段名(如 "LLM")。
    Raises:
        ValueError: 名字不在合法阶段集合里——故意 fail-fast，避免把耗时
            悄悄记到一个拼错的、永远查不到的桶里。
    """
    canonical = _INFERENCE_STAGE_NAME_MAP.get(name.strip().lower())
    if canonical is None:
        raise ValueError(
            f"Unsupported inference stage '{name}'. "
            f"Expected one of: {', '.join(INFERENCE_STAGE_NAMES)}."
        )
    return canonical


@dataclass(slots=True)
class InferenceStageStat:
    """单个推理阶段的累加器：总耗时(秒)与被计时的次数(count)。

    `slots=True` 省内存并防止误加字段；非 frozen 因为要在 measure() 里就地累加。
    """

    seconds: float = 0.0
    count: int = 0


@dataclass(frozen=True, slots=True)
class ProfileEvent:
    """数据流水线 worker 经 Queue 汇报回主进程的一条计时事件(不可变值对象)。

    Attributes:
        stage:   阶段名(如 "worker.load_audio")。
        seconds: 本次该阶段耗时(秒)。
        count:   本次处理的样本/单位数,用于算单位耗时。
        pid:     产生事件的进程号,便于按 worker 归并/排查。
    `frozen=True` 让它能安全跨进程传递且语义清晰(只读快照)。
    """

    stage: str
    seconds: float
    count: int
    pid: int


class DataProfiler:
    """训练数据流水线的跨进程埋点器(每个 worker 持一份,经 Queue 上报)。

    Design:
        数据加载跑在 multiprocessing worker 里,无法直接累加到主进程的对象,
        因此把每段耗时打包成 `ProfileEvent` 投进共享 `Queue`,由主进程统一聚合。
        若未传 queue(`None`),整个 measure() 退化为零开销的直通,方便默认关闭。
    """

    def __init__(self, queue: Queue | None = None):
        self._queue = queue
        self._pid = os.getpid()  # 记录本 worker 的 pid,事件里带上以便区分来源

    @property
    def enabled(self) -> bool:
        """是否真正在采样:有 queue 才上报,否则 measure() 是空操作。"""
        return self._queue is not None

    @contextmanager
    def measure(self, stage: str, *, count: int = 1) -> Iterator[None]:
        """上下文管理器:测量 `with` 体的墙钟耗时并把事件投递到 Queue。

        Args:
            stage: 阶段名(自由字符串,如 "worker.load_audio")。
            count: 本段处理的单位数(样本数),用于后续算单位耗时。
        未启用时(无 queue)直接 yield,不引入任何同步/IO 开销。
        """
        if self._queue is None:
            # 未启用:直接放行,避免在热路径上付出 perf_counter/Queue 代价。
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            # 放进 finally:即便 with 体抛异常,也把已发生的耗时如实上报。
            self._queue.put(
                ProfileEvent(
                    stage=stage,
                    seconds=time.perf_counter() - start,
                    count=int(count),
                    pid=self._pid,
                )
            )

    def child(self) -> DataProfiler:
        """派生一个共享同一 Queue 的子 profiler(用于 fork/spawn 出新 worker)。

        新实例会在自己的进程里重新取 `os.getpid()`,从而事件 pid 反映真实子进程。
        """
        return DataProfiler(self._queue)


def ensure_data_profiler(profiler: DataProfiler | None) -> DataProfiler:
    """归一化:None 时返回一个未启用的 DataProfiler,免去调用方到处判空。"""
    return DataProfiler() if profiler is None else profiler


class InferenceProfiler:
    """推理逐模块计时器:单进程,按 INFERENCE_STAGE_NAMES 分桶累加并算 RTF。

    Design:
        预先为每个合法阶段建一个 `InferenceStageStat`,measure() 只做就地累加,
        从而同一阶段被多次调用(如 LLM 逐步解码)会自然汇总。CUDA 设备上,计时
        前后都 `synchronize` 以消除 GPU 异步执行带来的失真(见 `_sync`)。
    """

    def __init__(self, device: torch.device):
        self._device = device
        # 预建全部阶段的累加器:避免 measure() 里判 key 是否存在,且 summary() 顺序稳定。
        self._stats = {
            stage: InferenceStageStat() for stage in INFERENCE_STAGE_NAMES
        }

    def _sync(self) -> None:
        # 只有 CUDA 才需要同步:GPU kernel 是异步下发的,不 synchronize 测到的只是
        # "下发耗时"而非"真正算完"的耗时。CPU 路径无此问题,跳过以省开销。
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @contextmanager
    def measure(self, stage: str, *, count: int = 1) -> Iterator[None]:
        """测量某推理阶段的耗时并累加到对应桶(CUDA 上做前后同步)。

        Args:
            stage: 阶段名,会先经 normalize 校验并转成规范名。
            count: 本次处理的单位数(如 token 数),累加进 stat.count。
        计时窗口被两次 `_sync()` 夹住:进入前同步保证起点干净,退出时(finally)
        再同步保证 GPU 真正算完才记终点。
        """
        stage = normalize_inference_stage_name(stage)
        self._sync()  # 起点同步:确保此前的 GPU 工作已落地,不计入本段
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()  # 终点同步:等本段 GPU kernel 真正执行完再读时钟
            stat = self._stats[stage]
            stat.seconds += time.perf_counter() - start
            stat.count += int(count)

    def summary(
        self,
        *,
        duration_seconds: float | None = None,
    ) -> dict[str, dict[str, float | int]]:
        """导出各阶段的 {seconds, count(, rtf)} 汇总字典。

        Args:
            duration_seconds: 合成出的音频时长(秒)。给出时为每个阶段附加
                RTF = 阶段耗时 / 音频时长(<1 即快于实时)。

        Returns:
            形如 {stage: {"seconds": .., "count": .., "rtf": ..}} 的嵌套字典;
            按 INFERENCE_STAGE_NAMES 顺序遍历,保证输出阶段顺序稳定。
        """
        summary: dict[str, dict[str, float | int]] = {}
        for stage in INFERENCE_STAGE_NAMES:
            stat = self._stats[stage]
            payload: dict[str, float | int] = {
                "seconds": stat.seconds,
                "count": stat.count,
            }
            if duration_seconds is not None:
                # 音频时长为 0 时 RTF 无意义,用 +inf 显式标记而非除零崩溃。
                payload["rtf"] = (
                    stat.seconds / duration_seconds
                    if duration_seconds > 0
                    else float("inf")
                )
            summary[stage] = payload
        return summary


@contextmanager
def inference_profiling(
    *,
    enabled: bool,
    device: torch.device,
) -> Iterator[InferenceProfiler | None]:
    """顶层入口:按 enabled 决定是否新建 profiler,并把它激活到当前上下文。

    runtime.py 用 `with inference_profiling(enabled=..., device=...) as p:` 包住一次
    完整推理;enabled=False 时产出 None(下游 measure_inference 自动空转,零开销)。
    yield 出的 profiler 可在退出后调 summary() 取结果。
    """
    profiler = InferenceProfiler(device) if enabled else None
    with activate_inference_profiler(profiler):
        yield profiler


@contextmanager
def activate_inference_profiler(
    profiler: InferenceProfiler | None,
) -> Iterator[InferenceProfiler | None]:
    """把给定 profiler 设为当前 ContextVar,退出时精确还原(支持嵌套)。

    用 `set()` 返回的 `Token` 在 finally 里 `reset()`,而不是粗暴地置 None——
    这样即使外层已有一个活跃 profiler,内层退出后也能恢复成外层的那个。
    profiler 为 None 时不动 ContextVar,直接放行(相当于"不开启 profiling")。
    """
    if profiler is None:
        yield None
        return
    token: Token[InferenceProfiler | None] = _CURRENT_INFERENCE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _CURRENT_INFERENCE_PROFILER.reset(token)  # 还原到 set 之前的值(可能是另一个 profiler)


@contextmanager
def measure_inference(stage: str, *, count: int = 1) -> Iterator[None]:
    """给业务代码用的轻量门面:就近取当前 profiler,代为计时某阶段。

    model.py 里大量出现 `with measure_inference("LLM"): ...`。它从 ContextVar 取
    活跃 profiler:没有(未开启 profiling)时直接 yield 空转,有则委托给
    `profiler.measure()`。好处是埋点代码与"是否启用 profiling"完全解耦。
    """
    profiler = _CURRENT_INFERENCE_PROFILER.get()
    if profiler is None:
        # 未激活任何 profiler:零开销直通,埋点对生产路径无影响。
        yield
        return
    with profiler.measure(stage, count=count):
        yield


def log_inference_profile(
    *,
    request_id: str,
    profiling: dict[str, dict[str, float | int]],
    duration_seconds: float,
) -> None:
    """把 `summary()` 的结果逐阶段打到 loguru 日志(含 seconds/count/rtf)。

    Args:
        request_id: 关联日志与具体请求,便于排查。
        profiling:  `InferenceProfiler.summary(duration_seconds=...)` 的输出,
                    因此各阶段应已带 "rtf" 字段。
        duration_seconds: 合成音频时长,仅在"无任何阶段被计时"时打一条概览日志。
    """
    # 只打真正跑过的阶段(count>0):一次请求未必触发全部 7 个模块。
    active_stages = [
        stage
        for stage in INFERENCE_STAGE_NAMES
        if int(profiling[stage]["count"]) > 0
    ]
    if not active_stages:
        logger.info(
            "Inference profiling summary: request_id={} no_profiled_stages duration_seconds={:.3f}",
            request_id,
            duration_seconds,
        )
        return
    for stage in active_stages:
        stats = profiling[stage]
        logger.info(
            "Inference profiling: request_id={} stage={} seconds={:.4f} count={} rtf={:.4f}",
            request_id,
            stage,
            float(stats["seconds"]),
            int(stats["count"]),
            float(stats["rtf"]),
        )


__all__ = [
    "DataProfiler",
    "ProfileEvent",
    "INFERENCE_STAGE_NAMES",
    "activate_inference_profiler",
    "ensure_data_profiler",
    "InferenceProfiler",
    "InferenceStageStat",
    "inference_profiling",
    "log_inference_profile",
    "measure_inference",
    "normalize_inference_stage_name",
]
