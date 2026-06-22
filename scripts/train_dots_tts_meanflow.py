#!/usr/bin/env python3

"""MeanFlow 蒸馏训练入口 / MeanFlow distillation training entrypoint for dots.tts.

本文件做什么 (What this file does)
================================
把一个已经预训练好的 flow-matching 声学头 (DiT velocity field predictor) **蒸馏**成一个
MeanFlow 学生模型，使它能用极少的 NFE (number of function evaluations，即去噪步数) 逼近
teacher 用多步 ODE solver 才能跑出的同一条轨迹。常规 flow-matching 推理需要几十步迭代解
ODE；MeanFlow 学生学习的是「在区间 [t, t+Δt] 上的**平均** velocity」，于是单步(甚至少数几
步)就能从噪声跳到数据端，大幅降低推理延迟。

蒸馏的核心思想 (Distillation idea)
---------------------------------
- **Teacher**: 冻结的原始 flow-matching DiT。给定路径上的样本 x_t、时刻 t 和区间长度 Δt，
  teacher 用 euler/midpoint/rk4 等 ODE solver 从 t 积分到 t+Δt，得到终点 z；区间上的
  **平均速度** = (z - x_t) / Δt，这就是要让学生匹配的回归目标 (regression target)。
- **Student**: 在 teacher 的 DiT 权重基础上额外加一个 ``duration_embedder``，把区间长度 Δt
  也编码进条件 c。学生直接预测平均速度，无需迭代，因此推理 NFE 可以压到 1~几步。
- **Anchor**: 以 ``anchor_prob`` 的概率令 Δt=0 (退化到瞬时速度)，此时目标可以直接用 teacher
  的瞬时 velocity，或用闭式公式 u_t = x1-(1-σ)x0 (``anchor_target``)。anchor 样本把学生在
  Δt→0 极限处「锚」回真正的 flow-matching velocity field，防止平均速度训练发散。
- **CFG 蒸馏 (CFG distillation)**: 推理时常用 classifier-free guidance (CFG) 把条件/无条件
  两次预测组合放大引导强度，相当于双倍 NFE。这里把 CFG 直接「烤进」teacher 目标里 (fused
  模式)：teacher 目标 = cond + scale·(cond - uncond)，学生一次前向即可复现 CFG 效果；
  natural 模式则保留训练期 condition dropout，让学生自己学到条件/无条件两支。

在数据流里的位置 (Position in the pipeline)
------------------------------------------
预训练 (train_dots_tts.py) ──► 本脚本 MeanFlow 蒸馏 ──► 少步推理 (inference 时 NFE 大减)。
复用预训练好的 LLM 自回归主干 (Qwen2.5)、patch_encoder、xvec 条件等，只重训/微调 DiT 声学头。

关键类与函数 (Key classes / functions)
--------------------------------------
- ``MeanFlowSettings``        : 冻结的蒸馏超参 dataclass (solver、步数、CFG 模式、anchor 等)。
- ``enable_meanflow_student`` : 把预训练 DiT 就地升级成带 duration_embedder 的 MeanFlow DiT。
- ``MeanFlowDotsTtsModel``    : 包住 student+teacher 的训练模块，实现蒸馏前向与目标计算。
    - ``compute_teacher_meanflow_target`` : teacher 多步 ODE rollout，算平均速度目标。
    - ``meanflow_forward`` / ``meanflow_fm_segment`` : 学生一次前向 + 时间/anchor 采样。
- ``DotsTtsMeanFlowTrainingRun`` : 继承常规训练循环，换上 MeanFlow 模型与 checkpoint 元数据。
- ``parse_args`` / ``main``   : CLI 入口。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from einops import rearrange
from torch.optim import AdamW
from train_dots_tts import DotsTtsTrainingRun
from transformers import get_cosine_schedule_with_warmup

from dots_tts.config import app as app_config
from dots_tts.data import builders as data_module
from dots_tts.models.dots_tts import model as dots_tts_model
from dots_tts.models.dots_tts.config import MeanFlowConfig
from dots_tts.models.dots_tts.core import DotsTtsForwardOutput
from dots_tts.modules.backbone.dit import DiT
from dots_tts.training import checkpoint as train_checkpoint
from dots_tts.training import utils as train_utils
from dots_tts.utils import util as util_module

# teacher 跑 ODE 时可选的求解器 / allowed ODE solvers for the teacher rollout
_ALLOWED_TEACHER_SOLVERS = ("euler", "midpoint", "rk4")
# CFG 蒸馏模式：fused=把 CFG 烤进目标；natural=保留 condition dropout 让学生自学两支
_ALLOWED_CFG_DISTILL_MODES = ("natural", "fused")
# anchor (Δt=0) 样本的目标来源：闭式公式 u_t，或 teacher 的瞬时 velocity
_ALLOWED_ANCHOR_TARGETS = ("formula", "teacher")


@dataclass(frozen=True, slots=True)
class MeanFlowSettings:
    """MeanFlow 蒸馏的全部超参 / immutable hyper-parameters for MeanFlow distillation.

    用 ``frozen=True`` 冻结，保证一次 run 内蒸馏配置不可变 (会写进 checkpoint 元数据，便于复现)。
    各字段含义见下方 inline 注释；``__post_init__`` 做取值校验，``to_dict`` 序列化进 config.yml。
    """

    teacher_model_path: str | None  # 冻结 teacher 路径；None 则复用 student 的预训练权重
    teacher_steps: int = 8  # teacher ODE rollout 的步数 (NFE)；步数越多目标越精确、越慢
    teacher_solver: str = "euler"  # 求解 ODE 用的数值积分器
    cfg_distill_mode: str = "fused"  # 见 _ALLOWED_CFG_DISTILL_MODES
    distill_cfg_scale: float = 1.2  # fused 模式下烤进目标的 CFG 引导强度 scale
    anchor_prob: float = 0.5  # 每个样本被设为 anchor (Δt=0) 的概率
    anchor_target: str = "formula"  # anchor 样本目标用闭式公式还是 teacher 瞬时速度
    time_sampling_mean: float = -0.4  # 时间采样 logit-normal 的均值 (sigmoid 前)
    time_sampling_std: float = 1.0  # 时间采样 logit-normal 的标准差 (sigmoid 前)
    train_all_parameters: bool = False  # True=全参微调；False=只训 DiT 声学头

    def __post_init__(self) -> None:
        # 取值校验：非法 solver / 模式 / 概率会在构造期直接报错，避免训练跑到一半才发现
        if int(self.teacher_steps) <= 0:
            raise ValueError("teacher_steps must be positive.")
        if self.teacher_solver not in _ALLOWED_TEACHER_SOLVERS:
            raise ValueError(
                f"teacher_solver must be one of {_ALLOWED_TEACHER_SOLVERS}, "
                f"got {self.teacher_solver!r}."
            )
        if self.cfg_distill_mode not in _ALLOWED_CFG_DISTILL_MODES:
            raise ValueError(
                "cfg_distill_mode must be one of "
                f"{_ALLOWED_CFG_DISTILL_MODES}, got {self.cfg_distill_mode!r}."
            )
        if self.anchor_target not in _ALLOWED_ANCHOR_TARGETS:
            raise ValueError(
                f"anchor_target must be one of {_ALLOWED_ANCHOR_TARGETS}, "
                f"got {self.anchor_target!r}."
            )
        if not 0.0 <= float(self.anchor_prob) <= 1.0:
            raise ValueError("anchor_prob must be in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        # 序列化成纯标量 dict，写入 config.yml 与 checkpoint 元数据，保证蒸馏配置可复现
        return {
            "teacher_model_path": self.teacher_model_path,
            "teacher_steps": int(self.teacher_steps),
            "teacher_solver": self.teacher_solver,
            "cfg_distill_mode": self.cfg_distill_mode,
            "distill_cfg_scale": float(self.distill_cfg_scale),
            "anchor_prob": float(self.anchor_prob),
            "anchor_target": self.anchor_target,
            "time_sampling_mean": float(self.time_sampling_mean),
            "time_sampling_std": float(self.time_sampling_std),
            "train_all_parameters": bool(self.train_all_parameters),
        }


def enable_meanflow_student(model: dots_tts_model.DotsTtsModel) -> None:
    """把预训练的 flow-matching student 就地升级成 MeanFlow 模式 / upgrade student DiT in-place.

    做三件事：(1) 打开 MeanFlowConfig 并把 core 切到 "meanflow" 模式；(2) 新建一个带
    ``duration_embedder`` 的 DiT，从旧 DiT 拷贝全部权重 (旧 DiT 没有 duration_embedder，所以
    那部分作为 missing_keys 被容忍)；(3) 把 duration_embedder 的输出层**零初始化**。

    为什么零初始化 duration 输出 (Why zero-init the duration head)
    ------------------------------------------------------------
    零初始化让 duration_embedder 一开始对条件 c 贡献为 0 —— 也就是说升级后的 MeanFlow DiT
    在训练第 0 步与原 flow-matching DiT **数值完全等价**，相当于从 teacher 的轨迹平滑起步，
    再逐渐学会利用 Δt 信息。这是 residual/adaLN 类条件注入常用的稳妥 warm-start 技巧。

    幂等 (idempotent)：若 DiT 已带 duration_embedder，直接返回，不重复初始化。
    """
    meanflow_config = MeanFlowConfig(enabled=True, use_duration_embedding=True)
    model.config.meanflow = meanflow_config
    model.core.meanflow_config = meanflow_config
    model.core.mode = "meanflow"  # core 前向据此走 MeanFlow 分支 (注入 duration 条件)

    old_dit = model.core.velocity_field_predictor
    if getattr(old_dit, "duration_embedder", None) is not None:
        return  # 已是 MeanFlow DiT，避免二次升级把权重再随机化

    new_dit = DiT(
        in_dim=model.core.fm_hidden_size,
        out_dim=model.core.latent_dim,
        transformer_config=model.core.config.DiT,
        mode="meanflow",
    )
    # strict=False：旧 DiT 缺少 duration_emb.* 权重，允许部分加载
    missing_keys, unexpected_keys = new_dit.load_state_dict(
        old_dit.state_dict(),
        strict=False,
    )
    # 唯一允许缺失的就是新增的 duration_embedder.*；其余缺失/多余都视为加载出错
    missing_keys = [
        key for key in missing_keys if not key.startswith("duration_embedder.")
    ]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to initialize MeanFlow DiT from the pretrained flow-matching "
            f"DiT: missing={missing_keys[:5]} unexpected={unexpected_keys[:5]}"
        )
    # 零初始化 duration 嵌入的输出层 → 初始时 duration 对条件 c 无影响 (见 docstring)
    duration_output = new_dit.duration_embedder.mlp[-1]
    nn.init.zeros_(duration_output.weight)
    nn.init.zeros_(duration_output.bias)
    model.core.velocity_field_predictor = new_dit


class MeanFlowDotsTtsModel(nn.Module):
    """蒸馏训练用的包装模块 / training wrapper holding both student and teacher.

    职责 (Responsibilities)
    -----------------------
    - 持有可训练的 ``student`` (会被 optimizer 更新) 和冻结的 ``teacher``。
    - 实现蒸馏前向：采时间/anchor、构造 DiT 输入、跑 teacher rollout 得目标、学生一次前向出预测。
    - 把大部分接口 (tokenizer、save/load、CFG droprate 等) 透传给 student，使外层训练循环
      (继承自普通 DotsTtsTrainingRun) 几乎无需改动。

    设计要点 (Design notes)
    -----------------------
    teacher **不**注册成子模块，而是放进普通 dict ``_teacher_holder`` 里。这样 teacher 的参数
    不会出现在 ``self.parameters()`` 中，于是：(1) 不会被 optimizer 当作可训练参数；(2) 不会被
    accelerate / DDP 包裹或 all-reduce 梯度。代价是 ``.to()/.cuda()/.train()`` 必须手动把
    teacher 一起搬设备并强制 eval，见下面重写的那几个方法。
    """

    def __init__(
        self,
        student: dots_tts_model.DotsTtsModel,
        settings: MeanFlowSettings,
    ):
        super().__init__()
        self.student = student
        self.settings = settings
        # 用 dict 而非 attribute 持有 teacher：绕过 nn.Module 的子模块注册 (见类 docstring)
        self._teacher_holder: dict[str, dots_tts_model.DotsTtsModel] = {}

    @property
    def config(self):
        return self.student.config

    @property
    def tokenizer(self):
        return self.student.tokenizer

    @property
    def teacher(self) -> dots_tts_model.DotsTtsModel:
        # 取冻结 teacher；未 set_teacher 就用会直接报错而不是静默出 None
        teacher = self._teacher_holder.get("model")
        if teacher is None:
            raise RuntimeError("MeanFlow teacher model has not been initialized.")
        return teacher

    def set_teacher(self, teacher: dots_tts_model.DotsTtsModel) -> None:
        """注入冻结 teacher：关掉所有梯度并切 eval / freeze and register the teacher."""
        for param in teacher.parameters():
            param.requires_grad_(False)  # teacher 永不更新，关梯度省显存与算力
        teacher.eval()  # 关 dropout/BN 等训练态随机性，目标必须确定性
        self._teacher_holder["model"] = teacher

    def to(self, *args, **kwargs):
        # 重写 .to()：父类只搬 student；teacher 在 dict 里需手动同步搬设备/精度并保持 eval
        super().to(*args, **kwargs)
        teacher = self._teacher_holder.get("model")
        if teacher is not None:
            self._teacher_holder["model"] = teacher.to(*args, **kwargs)
            self._teacher_holder["model"].eval()
        return self

    def cuda(self, device=None):
        # 同 .to()：手动把 teacher 一并搬到 GPU
        super().cuda(device)
        teacher = self._teacher_holder.get("model")
        if teacher is not None:
            self._teacher_holder["model"] = teacher.cuda(device).eval()
        return self

    def train(self, mode: bool = True):
        # 切训练态时强制 teacher 留在 eval：哪怕外层 model.train()，teacher 也不进训练态
        super().train(mode)
        teacher = self._teacher_holder.get("model")
        if teacher is not None:
            teacher.eval()
        return self

    def prepare_training_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.student.prepare_training_batch(data)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        return self.student.save_pretrained(save_directory)

    def load_pretrained_weights(
        self, pretrained_model_name_or_path: str | Path
    ) -> None:
        self.student.load_pretrained_weights(pretrained_model_name_or_path)

    def set_cfg_droprate(
        self,
        cfg_droprate: float | None = None,
        xvec_drop_rate: float | None = None,
    ) -> None:
        self.student.set_cfg_droprate(
            cfg_droprate=cfg_droprate,
            xvec_drop_rate=xvec_drop_rate,
        )

    @torch.no_grad()
    def compute_teacher_meanflow_target(
        self,
        *,
        xt: torch.Tensor,
        t: torch.Tensor,
        delta_t: torch.Tensor,
        prefix_data: dict[str, Any],
        g_cond: torch.Tensor | None,
        cfg_distill: bool,
        uncond_prefix_data: dict[str, Any] | None,
        uncond_g_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        """用 teacher 跑多步 ODE 得到 MeanFlow 回归目标 / teacher ODE rollout → mean velocity.

        蒸馏目标的核心计算 (The heart of distillation)
        ----------------------------------------------
        给定起点 x_t、起始时刻 t、区间长度 Δt，用 teacher 的 flow-matching velocity field 从 t
        积分到 t+Δt 得到终点 z；区间上的**平均速度** = (z - x_t) / Δt 即学生要回归的目标。
        学生只需一次前向预测这个平均速度，就能等效 teacher 的多步 ODE，从而 NFE 大降。

        anchor 特例 (Δt=0)
        ------------------
        当 Δt=0 (anchor 样本) 时平均速度无定义，改用 t 处的**瞬时** velocity ``v_init``
        (teacher 在起点的单次预测) 作为目标，把学生锚回真正的 velocity field。

        CFG 蒸馏 (fused 模式)
        ---------------------
        若 ``cfg_distill``，每次评估 velocity 都额外用 uncond 前缀算一遍 ``pred_u``，组合成
        ``pred + scale·(pred - pred_u)``，即把 CFG 引导烤进轨迹，学生一次前向复现引导效果。

        Args:
            xt: (B, T, D) 路径上的样本点 (各样本同一 padding 长度 T)。
            t:  (B,) 起始时刻。
            delta_t: (B,) 区间长度 Δt；==0 标记 anchor。
            prefix_data: cond 前缀的 DiT 输入 (含 fm_seq、mask、pos_ids 等)。
            g_cond: (B, ...) 说话人 x-vector 条件，可为 None。
            cfg_distill: 是否做 fused CFG 蒸馏。
            uncond_prefix_data / uncond_g_cond: CFG 的无条件分支输入。

        Returns:
            (sum_patches, latent_patch_size, D) 的目标张量，已按 patch 切好、转回 xt.dtype，
            供与学生预测逐元素回归。
        """
        teacher_core = self.teacher.core
        teacher_dit = teacher_core.velocity_field_predictor
        io_helper = teacher_core.io_helper
        noisy_proj = teacher_core.coordinate_proj
        n_steps = int(self.settings.teacher_steps)
        solver = self.settings.teacher_solver
        cfg_scale = float(self.settings.distill_cfg_scale)

        if solver not in _ALLOWED_TEACHER_SOLVERS:
            raise ValueError(f"Unsupported teacher solver: {solver!r}.")

        device = xt.device
        batch_size = xt.size(0)
        latent_lens = prefix_data["latent_lens"]
        latent_patch_size = int(prefix_data["latent_patch_size"])
        anchor_mask = delta_t.float() == 0  # Δt==0 的样本走 anchor 分支，不做 rollout

        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        # 强制关 autocast 全程用 fp32：ODE 积分对数值精度敏感，半精度会让目标累积误差
        with torch.autocast(device_type=autocast_device, enabled=False):
            z = xt.float()  # 积分状态，从起点 x_t 出发
            cur_t = t.float()  # 当前时刻，随每步前进
            safe_dt = delta_t.float().clamp(min=1e-6)  # 防 anchor 的 Δt=0 在下面做除数
            step_dt = safe_dt / n_steps  # 每个积分子步推进的时间 (B,)

            def evaluate(z_in: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
                # 在时刻 t_val 评估 teacher 的 velocity field，返回展平成 ((sum_len), D)。
                # 先把当前状态 z_in 写回 fm_seq 的 noise 区段 (前缀条件保持不变),再过 DiT。
                fm_seq = io_helper.replace_noise_latents_in_fm_seq(
                    prefix_data,
                    z_in.to(xt.dtype),
                    noisy_proj,
                ).float()
                vt = teacher_dit(
                    x=fm_seq,
                    timesteps=t_val,
                    pos_ids=prefix_data["fm_pos_ids"],
                    mask=prefix_data["fm_seq_mask"],
                    attn_mask=prefix_data["fm_attn_mask"],
                    g_cond=None if g_cond is None else g_cond.float(),
                )
                pred = io_helper.get_dit_outputs(
                    pred_v=vt,
                    fm_prefix_lengths=prefix_data["fm_prefix_lengths"],
                    fm_gen_lengths=prefix_data["fm_gen_lengths"],
                    fm_gen_patch_size=prefix_data["fm_gen_patch_size"],
                    latent_patch_size=prefix_data["latent_patch_size"],
                )

                if cfg_distill:
                    # fused CFG：再算一次无条件预测 pred_u，组合放大引导后烤进目标
                    if uncond_prefix_data is None:
                        raise RuntimeError(
                            "CFG distillation requires an uncond prefix."
                        )
                    fm_seq_u = io_helper.replace_noise_latents_in_fm_seq(
                        uncond_prefix_data,
                        z_in.to(xt.dtype),
                        noisy_proj,
                    ).float()
                    vt_u = teacher_dit(
                        x=fm_seq_u,
                        timesteps=t_val,
                        pos_ids=uncond_prefix_data["fm_pos_ids"],
                        mask=uncond_prefix_data["fm_seq_mask"],
                        attn_mask=uncond_prefix_data["fm_attn_mask"],
                        g_cond=None if uncond_g_cond is None else uncond_g_cond.float(),
                    )
                    pred_u = io_helper.get_dit_outputs(
                        pred_v=vt_u,
                        fm_prefix_lengths=uncond_prefix_data["fm_prefix_lengths"],
                        fm_gen_lengths=uncond_prefix_data["fm_gen_lengths"],
                        fm_gen_patch_size=uncond_prefix_data["fm_gen_patch_size"],
                        latent_patch_size=uncond_prefix_data["latent_patch_size"],
                    )
                    # CFG 公式：guided = cond + scale·(cond - uncond)
                    pred = pred + cfg_scale * (pred - pred_u)
                return rearrange(pred, "n p d -> (n p) d")  # 展平成 ((sum_len), D)

            v_init_flat = evaluate(z, cur_t)  # 起点 t 处的瞬时速度，anchor 目标会复用它

            def apply_velocity(
                z_cur: torch.Tensor,
                v_flat: torch.Tensor,
                *,
                dt_factor: float,
            ) -> torch.Tensor:
                # ODE 单步更新：z ← z + v·(step_dt·dt_factor)。dt_factor 给 midpoint/rk4
                # 的半步/试探步用 (如 0.5)。v_flat 是展平的 ((sum_len),D)，按各样本有效
                # 长度 length 切片写回，逐样本累加 offset 对齐到对应区段。
                new_z = z_cur.clone()
                offset = 0
                for batch_idx in range(batch_size):
                    length = int(latent_lens[batch_idx].item())
                    if length <= 0:
                        continue
                    # anchor 样本 (Δt=0) 不积分，状态保持不动 (其目标另用瞬时速度)
                    if not bool(anchor_mask[batch_idx].item()):
                        new_z[batch_idx, :length, :] = z_cur[
                            batch_idx, :length, :
                        ] + v_flat[offset : offset + length, :] * (
                            step_dt[batch_idx] * float(dt_factor)
                        )
                    offset += length  # 即便 anchor 也要前移 offset，保持与 v_flat 对齐
                return new_z

            # 三种 ODE solver：精度递增、每步 NFE 递增 (euler 1 次 / midpoint 2 次 / rk4 4 次)。
            # 第 0 步统一复用已算好的 v_init_flat，省一次 teacher 前向。
            if solver == "euler":
                # 一阶显式欧拉：z ← z + v(z,t)·dt
                v_flat = v_init_flat
                for step in range(n_steps):
                    if step > 0:
                        v_flat = evaluate(z, cur_t)
                    z = apply_velocity(z, v_flat, dt_factor=1.0)
                    cur_t = cur_t + step_dt
            elif solver == "midpoint":
                # 二阶中点法：用半步处的斜率 k2 做整步更新
                for step in range(n_steps):
                    k1 = v_init_flat if step == 0 else evaluate(z, cur_t)
                    z_mid = apply_velocity(z, k1, dt_factor=0.5)  # 试探到中点
                    k2 = evaluate(z_mid, cur_t + 0.5 * step_dt)  # 中点斜率
                    z = apply_velocity(z, k2, dt_factor=1.0)
                    cur_t = cur_t + step_dt
            else:
                # 经典四阶 Runge-Kutta (rk4)：四个斜率 k1..k4 加权平均，精度最高
                for step in range(n_steps):
                    k1 = v_init_flat if step == 0 else evaluate(z, cur_t)
                    z1 = apply_velocity(z, k1, dt_factor=0.5)
                    k2 = evaluate(z1, cur_t + 0.5 * step_dt)
                    z2 = apply_velocity(z, k2, dt_factor=0.5)
                    k3 = evaluate(z2, cur_t + 0.5 * step_dt)
                    z3 = apply_velocity(z, k3, dt_factor=1.0)
                    k4 = evaluate(z3, cur_t + step_dt)
                    z = apply_velocity(
                        z,
                        (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,  # RK4 加权斜率
                        dt_factor=1.0,
                    )
                    cur_t = cur_t + step_dt

            # rollout 终点 z 减起点 x_t 再除 Δt = 区间平均速度 (MeanFlow 的回归目标)
            mean_velocity = (z - xt.float()) / safe_dt.view(-1, 1, 1)
            # 逐样本拼目标：anchor 用瞬时速度 v_init，其余用区间平均速度 mean_velocity，
            # 都按 latent_patch_size 切回 (n_patch, patch, D) 后沿 batch 维 cat 起来。
            target_chunks = []
            offset = 0
            for batch_idx in range(batch_size):
                length = int(latent_lens[batch_idx].item())
                if length <= 0:
                    continue
                if bool(anchor_mask[batch_idx].item()):
                    # anchor：取 v_init_flat 中对应区段 (展平索引用 offset 对齐)
                    target_b = v_init_flat[offset : offset + length, :]
                else:
                    target_b = mean_velocity[batch_idx, :length, :]
                target_chunks.append(
                    rearrange(target_b, "(n p) d -> n p d", p=latent_patch_size)
                )
                offset += length
            if not target_chunks:
                raise RuntimeError("Teacher rollout produced no MeanFlow target.")
            return torch.cat(target_chunks, dim=0).to(xt.dtype)

    def forward(self, data: dict[str, Any]):
        """训练前向：组装输入 → MeanFlow 前向 → 复用 student 的 loss / training step entry.

        loss 项仍由 student._compute_loss_terms 计算 (文本 logits CE + DiT 回归 + eos)，
        只是 DiT 那部分的 (pred, target) 来自 MeanFlow 蒸馏而非普通 flow-matching。
        """
        loss_masks = data["loss_masks"]
        processed = self.student.prepare_training_inputs(data)
        # span mask 标记序列里哪些位置是音频 patch (输入端/输出端),DiT 段构造要用
        processed["input_span_mask"] = data["input_span_mask"]
        processed["output_span_mask"] = data["output_span_mask"]
        outputs = self.meanflow_forward(processed)
        return self.student._compute_loss_terms(
            outputs,
            labels=processed["labels"],
            loss_masks=loss_masks,
        )

    def meanflow_forward(self, data: dict[str, Any]) -> DotsTtsForwardOutput:
        """LLM 主干前向 + 准备 DiT 段，再分流到 MeanFlow 蒸馏 / LLM pass then MeanFlow head.

        流程：把音频 latent 编码成 patch embedding 填进 LLM 输入序列对应 span 位置 → 跑
        Qwen2.5 自回归主干拿 logits 与最后层 hidden → 若本 batch 有输出音频 patch，则走
        ``meanflow_fm_segment`` 做蒸馏，否则走 ``dummy_fm_forward`` 占位 (保证 DiT 参数
        始终参与 autograd 图，DDP 才不会因 unused params 报错)。

        Returns:
            DotsTtsForwardOutput(llm_logits, pred, target, eos_out)，交给 loss 计算。
        """
        core = self.student.core
        input_ids: torch.Tensor = data["input_ids"]
        input_ids_lengths: torch.Tensor = data["input_ids_lengths"]
        input_span_mask: torch.Tensor = data["input_span_mask"]
        output_span_mask: torch.Tensor = data["output_span_mask"]
        batch_size = input_ids.size(0)
        device = input_ids.device

        latents: torch.Tensor | None = data.get("latents")
        latents_sampled: torch.Tensor | None = data.get("latents_sampled")
        latent_lengths: torch.Tensor | None = data.get("latent_lengths")
        has_latents = latents is not None or latents_sampled is not None

        if has_latents:
            # 没有现成采样就从 AudioVAE 后验里采一份 latent；再编成 patch embedding 喂 LLM
            if latents_sampled is None:
                latents_sampled = core.io_helper.sample_from_latent(latents)
            patch_embeddings = core.patch_encoder(
                latents_sampled, x_lens=latent_lengths
            )
            valid_patch_counts = latent_lengths // core.latent_patch_size
            latents_sampled = core.io_helper.normalize(latents_sampled)  # 归一化到 DiT 工作域
        else:
            latents_sampled = None
            patch_embeddings = None
            valid_patch_counts = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )

        input_span_counts = input_span_mask.sum(dim=1)
        if input_span_counts.sum() > 0 and patch_embeddings is None:
            raise RuntimeError(
                "Found audio span tokens but no latents provided to compute patch embeddings."
            )

        # 文本 token 走词嵌入；音频 span 位置则用 patch_embeddings 覆写,实现文本+音频混合序列
        inputs_embeds = core.llm.get_input_embeddings()(input_ids)
        if patch_embeddings is not None:
            inputs_embeds = inputs_embeds.clone()  # clone 后再原地覆写，不污染嵌入表
            patch_embeddings = patch_embeddings.to(inputs_embeds.dtype)
            for batch_idx in range(batch_size):
                span_num = int(input_span_counts[batch_idx].item())
                if span_num == 0:
                    continue
                expected = int(valid_patch_counts[batch_idx].item())
                if expected != span_num:
                    raise RuntimeError(
                        f"Mismatch between span tokens ({span_num}) and latent patches "
                        f"({expected}) for sample {batch_idx}."
                    )
                # span_mask 的非零位即音频 patch 占位 token，逐位填入对应 patch embedding
                indices = input_span_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
                inputs_embeds[batch_idx, indices, :] = patch_embeddings[
                    batch_idx,
                    :span_num,
                    :,
                ]

        # LLM 段用普通因果 mask (区别于 DiT 段的块对角 chunk mask)
        _llm_attn_mask, llm_seq_mask, _ = core.causal_helper.create_causal_mask_and_pos(
            seq_lens=input_ids_lengths,
            max_len=input_ids.size(1),
        )
        llm_outputs = core.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=llm_seq_mask.long(),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        llm_logits = llm_outputs.logits
        llm_hidden = llm_outputs.hidden_states[-1]  # 最后一层 hidden，作为 DiT 段的条件来源
        eos = core.eos_proj(llm_hidden.detach())  # detach：eos 头的梯度不回传 LLM 主干

        total_patches = int(output_span_mask.sum().item())
        if total_patches > 0 and latents_sampled is None:
            raise RuntimeError("MeanFlow training requested but latents are missing.")

        if total_patches > 0:
            pred, target = self.meanflow_fm_segment(
                data,
                llm_hidden=llm_hidden,
                inputs_embeds=inputs_embeds,
                output_span_mask=output_span_mask,
                latents_sampled=latents_sampled,
                latent_lengths=latent_lengths,
            )
        else:
            pred, target = self.dummy_fm_forward(core, llm_hidden, device)

        return DotsTtsForwardOutput(
            llm_logits=llm_logits,
            pred=pred,
            target=target,
            eos_out=eos,
        )

    def meanflow_fm_segment(
        self,
        data: dict[str, Any],
        *,
        llm_hidden: torch.Tensor,
        inputs_embeds: torch.Tensor,
        output_span_mask: torch.Tensor,
        latents_sampled: torch.Tensor,
        latent_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MeanFlow 蒸馏的核心一步：采时间/anchor → teacher 目标 → 学生预测 / one distill step.

        关键步骤 (Key steps)
        --------------------
        1. **时间采样**：对每个样本采两个 logit-normal 时刻 t1,t2，取 ``t=min(t1,t2)``、
           ``Δt=|t1-t2|``。logit-normal (高斯过 sigmoid) 偏向中间时刻，是 flow-matching 常用
           的时间分布；用两点之差天然给出区间长度 Δt。
        2. **anchor 注入**：以 anchor_prob 把部分样本的 Δt 置 0，让它们退化为瞬时 velocity 监督。
        3. **构造 x_t**：x_t = sample_x_t(z0, x1, t)，z0 为噪声、x1 为数据端 latent。
        4. **teacher 目标**：teacher 在同一 (x_t,t,Δt) 上 rollout 得平均速度 (见
           ``compute_teacher_meanflow_target``)；anchor 样本可换成闭式 formula 目标。
        5. **学生预测**：student DiT 一次前向，额外吃 ``duration=Δt`` 条件，输出 pred。
        返回 (pred, target) 供回归 loss。

        CFG 模式差异 (CFG mode)
        -----------------------
        - fused：关掉训练期 condition dropout (cfg_mask=0),把 CFG 烤进 teacher 目标 (需 uncond 前缀)。
        - natural：按 droprate 随机丢条件，让学生自己学到条件/无条件两支,推理时再外部做 CFG。
        """
        core = self.student.core
        teacher_core = self.teacher.core
        settings = self.settings
        batch_size = latents_sampled.size(0)
        device = latents_sampled.device
        latent_dtype = latents_sampled.dtype
        # 采两个时刻：高斯 → 仿射 (mean/std) → sigmoid = logit-normal 分布于 (0,1)
        first_t = torch.randn(batch_size, device=device, dtype=latent_dtype)
        second_t = torch.randn(batch_size, device=device, dtype=latent_dtype)
        first_t = torch.sigmoid(
            first_t * float(settings.time_sampling_std)
            + float(settings.time_sampling_mean)
        )
        second_t = torch.sigmoid(
            second_t * float(settings.time_sampling_std)
            + float(settings.time_sampling_mean)
        )
        t_vec = torch.minimum(first_t, second_t)  # 区间起点 t = 两时刻较小者
        delta_t = (first_t - second_t).abs()  # 区间长度 Δt = 两时刻之差
        # 以 anchor_prob 抽中的样本将 Δt 置 0，转成瞬时速度 (anchor) 监督
        anchor_mask = torch.rand(batch_size, device=device, dtype=latent_dtype) < float(
            settings.anchor_prob
        )
        delta_t = torch.where(anchor_mask, torch.zeros_like(delta_t), delta_t)
        z0 = torch.randn_like(latents_sampled)  # 噪声端 x0 ~ N(0,I)
        xt = core.fm_helper.sample_x_t(  # 路径采样 x_t = μ_t + σ_t·z0
            z0,
            latents_sampled,
            t_vec.view(-1, 1, 1).to(latent_dtype),  # t reshape 成 (B,1,1) 便于广播
        )

        fused_cfg = settings.cfg_distill_mode == "fused"
        if fused_cfg:
            # fused：不丢条件 (mask 全 0)，CFG 改由 teacher 目标内部合成
            cfg_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            xvec_drop_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            # natural：按 droprate 随机丢前缀条件 / x-vector，学生自学条件与无条件两支
            cfg_mask = torch.empty(
                batch_size, device=device, dtype=torch.float32
            ).uniform_(0, 1) < float(core.cfg_droprate)
            xvec_drop_mask = torch.empty(
                batch_size, device=device, dtype=torch.float32
            ).uniform_(0, 1) < float(core.xvec_drop_rate)

        # 说话人音色条件：x-vector 投影后，对被 drop 且为人声的样本置零 (CFG 用)
        xvec_cond = core.xvec_proj(data["xvector"])
        vocal_mask = data.get("vocal_mask")
        if vocal_mask is None:
            vocal_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        # 只对 vocal 样本应用 xvec drop：非人声本就无说话人身份可言
        xvec_cond = util_module.mask_data(xvec_cond, xvec_drop_mask & vocal_mask)

        # DiT 段的 per-token 条件：输出 span 位用 LLM hidden，其余位用原始嵌入
        hiddens_for_fm = torch.where(
            output_span_mask.unsqueeze(-1),
            llm_hidden,
            inputs_embeds,
        )
        # 组装学生 DiT 的输入序列 (前缀条件 C + 含噪声 x_t 的生成段 Z)，含块对角 mask、pos_ids
        prefix_data = core.io_helper.prepare_meanflow_inputs_for_dit(
            hiddens=hiddens_for_fm,
            latents=latents_sampled,
            latent_lens=latent_lengths,
            hidden_proj=core.hidden_proj,
            latent_proj=core.latent_proj,
            noisy_proj=core.coordinate_proj,
            span_mask=output_span_mask,
            hidden_patch_size=core.hidden_patch_size,
            latent_patch_size=core.latent_patch_size,
            cfg_mask=cfg_mask,
            noise_latents=xt,
        )

        # teacher 侧输入在 no_grad 下构造 (teacher 不参与反传)；其投影层权重与 student 不同
        uncond_prefix_data = None
        uncond_g_cond = None
        with torch.no_grad():
            teacher_xvec_cond = teacher_core.xvec_proj(data["xvector"])
            teacher_xvec_cond = util_module.mask_data(
                teacher_xvec_cond,
                xvec_drop_mask & vocal_mask,
            )
            teacher_prefix_data = (
                teacher_core.io_helper.prepare_meanflow_inputs_for_dit(
                    hiddens=hiddens_for_fm,
                    latents=latents_sampled,
                    latent_lens=latent_lengths,
                    hidden_proj=teacher_core.hidden_proj,
                    latent_proj=teacher_core.latent_proj,
                    noisy_proj=teacher_core.coordinate_proj,
                    span_mask=output_span_mask,
                    hidden_patch_size=teacher_core.hidden_patch_size,
                    latent_patch_size=teacher_core.latent_patch_size,
                    cfg_mask=cfg_mask,
                    noise_latents=xt,
                )
            )
            if fused_cfg:
                # fused CFG 需要一份「全无条件」前缀：cfg_mask 全 1 (丢掉所有前缀条件)
                # 且 g_cond 置零，作为 CFG 的 uncond 分支
                uncond_prefix_data = (
                    teacher_core.io_helper.prepare_meanflow_inputs_for_dit(
                        hiddens=hiddens_for_fm,
                        latents=latents_sampled,
                        latent_lens=latent_lengths,
                        hidden_proj=teacher_core.hidden_proj,
                        latent_proj=teacher_core.latent_proj,
                        noisy_proj=teacher_core.coordinate_proj,
                        span_mask=output_span_mask,
                        hidden_patch_size=teacher_core.hidden_patch_size,
                        latent_patch_size=teacher_core.latent_patch_size,
                        cfg_mask=torch.ones(  # 全 1 = 全部丢条件，得到无条件预测
                            batch_size, device=device, dtype=torch.bool
                        ),
                        noise_latents=xt,
                    )
                )
                uncond_g_cond = torch.zeros_like(teacher_xvec_cond)  # 无条件分支不给说话人

        # teacher rollout 得回归目标 (平均速度 / anchor 处瞬时速度)
        teacher_target = self.compute_teacher_meanflow_target(
            xt=xt,
            t=t_vec,
            delta_t=delta_t,
            prefix_data=teacher_prefix_data,
            g_cond=teacher_xvec_cond,
            cfg_distill=fused_cfg,
            uncond_prefix_data=uncond_prefix_data,
            uncond_g_cond=uncond_g_cond,
        )
        # 若选 formula anchor，把 anchor 样本的目标换成闭式 u_t (更稳，绕过 teacher 误差)
        if anchor_mask.any() and settings.anchor_target == "formula":
            target = self.replace_anchor_targets_with_formula(
                teacher_target,
                z0=z0,
                latents_sampled=latents_sampled,
                latent_lengths=latent_lengths,
                anchor_mask=anchor_mask,
            )
        else:
            target = teacher_target

        # 学生 DiT 一次前向：除 timesteps(t) 外额外吃 duration(Δt) 条件，直接预测平均速度
        student_vt = core.velocity_field_predictor(
            x=prefix_data["fm_seq"],
            timesteps=t_vec,
            duration=delta_t,  # MeanFlow 关键：把区间长度 Δt 作为条件喂进 DiT
            pos_ids=prefix_data["fm_pos_ids"],
            mask=prefix_data["fm_seq_mask"],
            attn_mask=prefix_data["fm_attn_mask"],
            g_cond=xvec_cond,
        )
        # 从完整 DiT 序列输出里抽出生成段的 latent 预测 (丢掉前缀/hidden 占位部分)
        pred = core.io_helper.get_dit_outputs(
            pred_v=student_vt,
            fm_prefix_lengths=prefix_data["fm_prefix_lengths"],
            fm_gen_lengths=prefix_data["fm_gen_lengths"],
            fm_gen_patch_size=prefix_data["fm_gen_patch_size"],
            latent_patch_size=prefix_data["latent_patch_size"],
        )
        return pred, target

    def replace_anchor_targets_with_formula(
        self,
        teacher_target: torch.Tensor,
        *,
        z0: torch.Tensor,
        latents_sampled: torch.Tensor,
        latent_lengths: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """把 anchor 样本的目标替换成闭式公式 u_t / swap anchor targets for the closed-form u_t.

        anchor (Δt=0) 处的真值就是 flow-matching 的瞬时 velocity，可以直接由闭式公式
        ``u_t = x1 - (1-σ)·z0`` 算出 (compute_u_t)，无需依赖 teacher 预测，因此更精确也更稳。
        非 anchor 样本保留 teacher rollout 出来的平均速度目标。逐样本按 patch 拼回。

        Args:
            teacher_target: (sum_patches, patch, D) teacher 算出的目标 (含 anchor 与非 anchor)。
            z0/latents_sampled: 噪声端与数据端 latent，供闭式公式使用。
            latent_lengths/anchor_mask: 各样本有效长度与 anchor 标记。

        Returns:
            与 teacher_target 同形状、anchor 段已替换为公式目标的张量。
        """
        core = self.student.core
        formula_target = core.fm_helper.compute_u_t(z0, latents_sampled)  # 闭式 u_t (整 batch)
        chunks = []
        offset = 0  # 在 teacher_target 的 patch 维上的游标 (只对非 anchor 推进取片)
        for batch_idx in range(latents_sampled.size(0)):
            length = int(latent_lengths[batch_idx].item())
            if length <= 0:
                continue
            patch_count = length // core.latent_patch_size
            if bool(anchor_mask[batch_idx].item()):
                # anchor：用闭式 u_t，切回 (n_patch, patch, D)
                chunks.append(
                    rearrange(
                        formula_target[batch_idx, :length, :],
                        "(n p) d -> n p d",
                        p=core.latent_patch_size,
                    )
                )
            else:
                # 非 anchor：原样取 teacher 目标对应区段
                chunks.append(teacher_target[offset : offset + patch_count])
            offset += patch_count  # teacher_target 已是「过滤掉空样本」的紧凑排布，故按 patch_count 走
        if not chunks:
            raise RuntimeError("Anchor target replacement produced no target.")
        return torch.cat(chunks, dim=0)

    def dummy_fm_forward(
        self,
        core,
        llm_hidden: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """无音频 patch 时的占位 DiT 前向 / no-op DiT pass when a batch has no audio patches.

        若本 batch 全是纯文本 (无输出音频 patch),仍跑一次「乘 0」的假前向，让
        velocity_field_predictor 的参数出现在 autograd 图里。否则 DDP 会因为某些 rank 的 DiT
        参数「未参与 loss」而报 unused-parameter 错误。pred 与 target (=pred.detach()) 相等，
        回归 loss 为 0，不影响优化方向。
        """
        dummy_length = core.latent_patch_size
        # 三路投影各喂全零再乘 0：确保每个 proj 层权重都进图，但数值贡献严格为 0
        dummy_seq_h = llm_hidden.new_zeros((1, dummy_length, core.llm_hidden_size))
        dummy_seq_h = core.hidden_proj(dummy_seq_h) * 0.0
        dummy_seq_l = llm_hidden.new_zeros((1, dummy_length, core.latent_dim))
        dummy_seq_l = core.latent_proj(dummy_seq_l) * 0.0
        dummy_seq_c = llm_hidden.new_zeros((1, dummy_length, core.latent_dim))
        dummy_seq_c = core.coordinate_proj(dummy_seq_c) * 0.0
        dummy_seq = dummy_seq_h + dummy_seq_l + dummy_seq_c
        dummy_times = torch.zeros((1,), device=device, dtype=torch.float32)
        dummy_duration = torch.zeros((1,), device=device, dtype=torch.float32)
        dummy_attn_mask = torch.ones(
            (1, dummy_length, dummy_length),
            device=device,
            dtype=torch.bool,
        )
        dummy_out = core.velocity_field_predictor(
            x=dummy_seq,
            timesteps=dummy_times,
            duration=dummy_duration,
            attn_mask=dummy_attn_mask,
        )
        pred = dummy_out[:, -core.latent_patch_size :, :]
        return pred, pred.detach()  # target=detach(pred) → 回归 loss 恒为 0


class DotsTtsMeanFlowTrainingRun(DotsTtsTrainingRun):
    """MeanFlow 蒸馏的训练 run / training run wrapping accelerate, data, checkpoints.

    继承普通 ``DotsTtsTrainingRun`` 复用其训练循环 (run/_train_step/验证/日志等)，只重写：
    - ``__init__``：换上 MeanFlow 模型 (student+teacher)、按需冻结非 DiT 参数、装载数据。
    - ``_write_run_config`` / ``_save_checkpoint``：在 config 与 checkpoint 里额外记录蒸馏超参，
      保证恢复训练时蒸馏设置一致、且产物可复现。
    """

    def __init__(
        self,
        cfg: app_config.AppConfig,
        *,
        meanflow_settings: MeanFlowSettings,
        debug_enabled: bool = False,
    ):
        self.cfg = cfg
        self.meanflow_settings = meanflow_settings
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
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.grad_accumulation_steps,
            log_with="tensorboard",
            project_config=project_config,
            step_scheduler_with_optimizer=False,
        )

        util_module.seed_everything(self.cfg.train.seed)

        # student：从预训练权重加载，升级成 MeanFlow DiT (加 duration_embedder 并零初始化)
        student = dots_tts_model.DotsTtsModel.from_pretrained(
            self.cfg.train.pretrained_model_path
        )
        student.set_cfg_droprate(
            cfg_droprate=self.cfg.train.cfg_droprate,
            xvec_drop_rate=self.cfg.train.xvec_drop_rate,
        )
        enable_meanflow_student(student)
        # 默认只训 DiT 声学头：冻结全部，再单独解冻 velocity_field_predictor
        if not bool(meanflow_settings.train_all_parameters):
            for param in student.parameters():
                param.requires_grad_(False)
            for param in student.core.velocity_field_predictor.parameters():
                param.requires_grad_(True)
        model = MeanFlowDotsTtsModel(student, meanflow_settings)

        # teacher：另起一份未升级的预训练权重 (默认与 student 同源)，冻结后只用于产目标
        teacher_path = (
            meanflow_settings.teacher_model_path or self.cfg.train.pretrained_model_path
        )
        teacher = dots_tts_model.DotsTtsModel.from_pretrained(teacher_path)
        model.set_teacher(teacher)

        optimizer = AdamW(
            # 只把 requires_grad 的参数交给优化器 (teacher 不在 model.parameters() 里)
            (param for param in model.parameters() if param.requires_grad),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.train.warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        # 显式 .to(device)：触发上面重写的 to()，把 dict 里的 teacher 也搬到同一设备
        self.unwrapped_model.to(self.accelerator.device)

        # 一致性校验：数据侧采样率 / 每 LLM token 的音频样本数必须与预训练模型的契约对齐
        expected_sample_rate = int(self.unwrapped_model.config.vocoder.sample_rate)
        expected_audio_samples_per_llm_token = int(
            self.unwrapped_model.student.hop_size
        ) * int(self.unwrapped_model.config.patch_size)
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

        if self.accelerator.is_main_process:
            total_params = sum(
                param.numel() for param in self.unwrapped_model.parameters()
            )
            trainable_params = sum(
                param.numel()
                for param in self.unwrapped_model.parameters()
                if param.requires_grad
            )
            self.accelerator.print(f"Total parameters: {total_params:,}")
            self.accelerator.print(f"Trainable parameters: {trainable_params:,}")
            self.accelerator.print(
                f"MeanFlow teacher path: {Path(teacher_path).expanduser()}"
            )
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
        if self.cfg.train.eval_interval is not None or self.cfg.train.run_eval_on_start:
            if self.cfg.val_data is None:
                raise ValueError(
                    "Validation requires val_data when eval_interval or "
                    "run_eval_on_start is enabled."
                )
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
        self.train_loader.set_epoch(self.progress.epoch)

    def _write_run_config(self) -> None:
        # 把本次 run 的完整配置 (含 meanflow_train 段) 落盘到 output_dir/config.yml，便于复现
        if not bool(getattr(self.accelerator, "is_main_process", True)):
            return  # 仅主进程写，避免多卡并发写同一文件
        output_dir = Path(self.cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.yml"
        payload = self.cfg.to_dict()
        payload["meanflow_train"] = self.meanflow_settings.to_dict()
        with config_path.open("w", encoding="utf-8") as fout:
            yaml.safe_dump(
                payload,
                fout,
                sort_keys=False,
                allow_unicode=True,
            )

    def _save_checkpoint(self, learning_rate: float) -> None:
        # 存 checkpoint：除常规 scheduler 状态外，额外把 meanflow 蒸馏超参写进元数据
        train_checkpoint.save_train_checkpoint(
            self.accelerator,
            self.model,
            self.optimizer,
            self.progress,
            self.cfg.train.output_dir,
            self.cfg.train.max_checkpoints_to_keep,
            self.train_loader.state_dict(),
            {
                "type": "transformers_cosine_with_warmup_meanflow",
                "global_step": int(self.progress.global_step),
                "base_lr": float(self.cfg.train.learning_rate),
                "current_lr": float(learning_rate),
                "warmup_steps": int(self.cfg.train.warmup_steps),
                "max_train_steps": int(self.max_train_steps),
                "meanflow": self.meanflow_settings.to_dict(),
                "state_dict": self.scheduler.state_dict(),
            },
        )


def parse_args(argv=None):
    """解析 CLI：训练配置路径 + 全部 MeanFlow 蒸馏超参 / parse config path and distill flags."""
    parser = argparse.ArgumentParser(
        description="Accelerate MeanFlow distillation entrypoint for dots.tts."
    )
    parser.add_argument("--config", default=app_config.DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print training debug information.",
    )
    parser.add_argument(
        "--teacher-model-path",
        default=None,
        help=(
            "Frozen flow-matching teacher model path. Defaults to "
            "train.pretrained_model_path."
        ),
    )
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument(
        "--teacher-solver",
        choices=_ALLOWED_TEACHER_SOLVERS,
        default="euler",
    )
    parser.add_argument(
        "--cfg-distill-mode",
        choices=_ALLOWED_CFG_DISTILL_MODES,
        default="fused",
    )
    parser.add_argument("--distill-cfg-scale", type=float, default=1.2)
    parser.add_argument("--anchor-prob", type=float, default=0.5)
    parser.add_argument(
        "--anchor-target",
        choices=_ALLOWED_ANCHOR_TARGETS,
        default="formula",
    )
    parser.add_argument("--time-sampling-mean", type=float, default=-0.4)
    parser.add_argument("--time-sampling-std", type=float, default=1.0)
    parser.add_argument(
        "--train-all-parameters",
        action="store_true",
        help="Train all regular dots.tts parameters instead of only the DiT.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    # 入口：CLI → MeanFlowSettings → 加载 app config → 构造 run → 跑训练循环 (run())
    args = parse_args(argv)
    settings = MeanFlowSettings(
        teacher_model_path=args.teacher_model_path,
        teacher_steps=args.teacher_steps,
        teacher_solver=args.teacher_solver,
        cfg_distill_mode=args.cfg_distill_mode,
        distill_cfg_scale=args.distill_cfg_scale,
        anchor_prob=args.anchor_prob,
        anchor_target=args.anchor_target,
        time_sampling_mean=args.time_sampling_mean,
        time_sampling_std=args.time_sampling_std,
        train_all_parameters=args.train_all_parameters,
    )
    return DotsTtsMeanFlowTrainingRun(
        app_config.load_config(args.config),
        meanflow_settings=settings,
        debug_enabled=args.debug,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
