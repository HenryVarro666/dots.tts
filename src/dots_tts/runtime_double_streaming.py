"""Double-streaming（双流式）增量推理会话。

本文件做什么 / What this file does
----------------------------------
实现 dots.tts 的 **double-streaming** 在线推理：调用方一边喂入 text token、
本会话一边吐出 audio chunk，无需等整段文本到齐。"double" 指两条流同时在线 ——
**输入侧 text token 流** 与 **输出侧音频 chunk 流** —— 典型用于把上游 LLM 的逐
token 输出实时转成语音（边想边说）。

在推理数据流里的位置 / Position in the inference pipeline
--------------------------------------------------------
普通 ``DotsTtsRuntime.generate_stream`` 一次拿到完整文本，预先 build 出整条
generation schedule 再解码（见 ``runtime.py`` / ``model.py::_decode``）。本会话把
那条 batch 化的状态机拆成 **可逐 token 驱动的增量状态机**：复用同一套底层原语
（``_consume_text_schedule`` / ``_decode_next_audio`` / ``_consume_audio_patch``
/ vocoder streaming），但由外部按 token 节奏调用。整条链路是
``text token -> LLM（autoregressive 主干）-> flow-matching DiT 声学头 -> 连续
latent patch -> BigVGAN AudioVAE 流式 vocoder -> 波形 chunk``，全程 **无离散音频
token**，DiT 在连续 latent 空间用 velocity field + ODE solver 采样。

关键类 / Key classes
--------------------
- ``DoubleStreamingSession``：增量会话状态机本体。三大状态资源——LLM/FM 解码
  state（``_state``，含 KV cache 与 FM buffer）、vocoder 流式 state
  （``_vocoder_state``）、以及 prompt 说话人条件 ``_g_cond``（x-vector，可跨会话缓
  存复用）。对外暴露 ``push_text_token`` / ``finish_text``。
- ``DotsTtsRuntimeDoubleStreaming``：在 ``DotsTtsRuntime`` 之上加一个
  ``start_double_streaming`` 工厂方法，返回一个 ``DoubleStreamingSession``。

调用契约 / Call contract
------------------------
``start_double_streaming() -> 多次 push_text_token() -> finish_text()``。
``push_text_token`` 每次返回 0 或 1 个 audio chunk（可能为 ``None``，表示本步
vocoder 尚未攒够一个输出窗口）；``finish_text`` 是 generator，负责喂入 text-end
标记、把 EOS 之后的音频尾部 flush 干净，并 yield 剩余 chunk。
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from dots_tts.data.pipelines.tts_pipeline import TTS_INTERLEAVE_PREFIX
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.util import get_dtype


class DoubleStreamingSession:
    """Incremental interleave session for text-token to audio-chunk generation.

    增量交错（interleave）会话：把"喂文本 / 出音频"做成一个可被外部逐 token 驱动
    的状态机。"interleave" 指 text token 与 audio patch 在 LLM 序列里交替排列——
    每喂入若干 text token，就解码若干 audio patch，二者共用同一条 KV cache。

    状态机的几个关键标志位 / The state-machine flags
    ----------------------------------------------
    - ``_started``：是否已注入流式前缀 ``TTS_INTERLEAVE_PREFIX``。前缀只在第一次喂
      token（或 ``finish_text`` 提前触发）时随首批 token 一起进 LLM，之后不再重复。
    - ``_text_finished``：是否已喂入 text-end 标记（``text_cond_end_id``），喂入后
      不允许再 ``push_text_token``。
    - ``_state.end_flag``：底层解码 state 的 EOS 标志。由 EOS 概率超过
      ``eos_threshold`` 触发（见 ``_decode_audio_chunk``），表示模型自认为"说完了"。
    - ``_closed``：会话是否已彻底结束（``finish_text`` 跑完、vocoder 已 flush）。

    设计要点 / Design notes
    -----------------------
    - **不支持 prompt_text**：double-streaming 只用 reference audio 的说话人/音色条
      件（x-vector ``g_cond``），不做 prompt-prefill 的文本对齐，因此构造时若给了
      非空 ``prompt_text`` 直接报错。
    - **prompt 条件缓存**：从 reference audio 抽出的 ``g_cond`` 按
      (路径, device, dtype, speaker_scale) 缓存在 runtime 上，多个会话/请求复用，
      避免每次都跑一遍 ``_prepare_prompt_conditioning``（CAM++ x-vector 提取较贵）。
    - **initial silence**：前 ``_initial_silence_audio_tokens`` 个 audio patch 用预
      先算好的"静音 latent"替换模型输出，给开头垫一小段静音，缓解起播突兀/截音。
    """

    def __init__(
        self,
        runtime: DotsTtsRuntime,
        *,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        initial_silence_audio_tokens: int = 1,
    ) -> None:
        """初始化一个 double-streaming 会话并预热全部增量状态资源。

        关键参数 / Key params
        ----------------------
        - ``runtime``：已加载权重的 ``DotsTtsRuntime``，提供 model / device /
          tokenizer / vocoder 等。
        - ``prompt_audio_path``：reference audio 路径；若给出，则抽 x-vector 说话人
          条件 ``g_cond`` 用作音色克隆条件（带跨会话缓存）。``prompt_text`` 不支持，
          非空即报错。
        - ``ode_method`` / ``num_steps``：flow-matching DiT 声学头的 ODE solver 设置
          （如 "euler" + 步数），控制 velocity field 的积分精度/速度权衡。
        - ``guidance_scale``：classifier-free guidance (CFG) 强度，作用在声学头上。
        - ``speaker_scale``：说话人条件强度，影响 ``g_cond`` 的提取与作用。
        - ``eos_threshold``：EOS 概率阈值，超过则在当前 audio patch 后置 end_flag。
        - ``initial_silence_audio_tokens``：开头用静音 latent 替换的 patch 数，夹到
          [0, 10]。

        副作用 / Side effects
        ---------------------
        预分配 LLM/FM 解码 state（含 KV cache 与 FM 静态 buffer）与 vocoder 流式
        state；若有 reference audio，则填充（或命中缓存）``_g_cond``。
        """
        # double-streaming 不做 prompt-text 对齐：先归一化，若仍非空就拒绝。
        normalized_prompt_text = runtime._process_prompt_text(prompt_text)
        if normalized_prompt_text:
            raise ValueError("Double streaming does not support prompt_text.")

        self.runtime = runtime
        self.model = runtime.model
        self.device = runtime.device
        self.ode_method = ode_method
        self.num_steps = int(num_steps)
        self.guidance_scale = float(guidance_scale)
        self.speaker_scale = float(speaker_scale)
        self.eos_threshold = float(eos_threshold)
        self.max_generate_length = runtime.max_generate_length
        # 夹到 [0, 10]：与下方 silence latent 缓存的 cache_count=10 上限一致。
        self._initial_silence_audio_tokens = max(
            0,
            min(10, int(initial_silence_audio_tokens or 0)),
        )

        self._dtype = get_dtype(runtime.precision)
        # 仅在 CUDA + 半精度时启用 autocast (AMP)；CPU/fp32 路径关闭以保数值稳定。
        self._use_amp = self.device.type == "cuda" and self._dtype in {
            torch.float16,
            torch.bfloat16,
        }
        # 流式前缀（"[流式语音合成]"）的 token id，首批 text token 前注入一次。
        self._prefix_token_ids = tuple(
            self.model.tokenizer.encode(
                TTS_INTERLEAVE_PREFIX,
                add_special_tokens=False,
            )
        )
        # 预分配整条解码 state：LLM KV cache + FM 静态 buffer，容量按 max patch 数。
        self._state = self.model._allocate_generate_state(
            max_audio_patch_count=self.max_generate_length,
            device=self.device,
            dtype=self._dtype,
        )
        # vocoder 流式 state：保存 LSTM hidden 与解码滑动窗口，chunk_size = 一个
        # latent patch 的帧数，使"每出一个 audio patch 就推一步 vocoder"对齐。
        self._vocoder_state = self.model.vocoder.init_stream_state(
            batch_size=1,
            chunk_size=self.model.core.latent_patch_size,
        )
        self._g_cond = None  # 说话人/音色条件 (x-vector)，无 reference audio 时保持 None
        self._started = False  # 是否已注入流式前缀
        self._text_finished = False  # 是否已喂入 text-end 标记
        self._closed = False  # 会话是否已结束（flush 完毕）
        self._decoded_patch_count = 0  # 已解码 audio patch 计数（用于 silence 替换/上限检查）

        if prompt_audio_path is not None:
            # prompt 条件缓存挂在 runtime 上（而非本会话），以便多个会话/请求复用同一
            # reference audio 的 g_cond。缓存键含 device/dtype/speaker_scale，因为这些
            # 任一变化都会改变提取出的 x-vector 条件张量。
            cache = getattr(self.runtime, "_double_streaming_prompt_g_cond_cache", None)
            if cache is None:
                cache = {}
                setattr(self.runtime, "_double_streaming_prompt_g_cond_cache", cache)
            prompt_cache_key = (
                str(Path(prompt_audio_path).expanduser().resolve()),  # 绝对化路径，规避相对路径歧义
                str(self.device),
                str(self._dtype),
                self.speaker_scale,
            )
            cached_g_cond = cache.get(prompt_cache_key)
            if cached_g_cond is None:
                # cache miss：加载 reference audio 并提取说话人条件（较贵，仅首次执行）。
                prompt_audio = self.runtime._load_prompt_audio(prompt_audio_path)
                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self._dtype,
                        enabled=self._use_amp,
                    ):
                        # use_prompt_prefill=False：只取说话人条件 g_cond，不做文本
                        # 对齐的 prompt-prefill（double-streaming 不支持 prompt_text）。
                        prompt_conditioning = self.model._prepare_prompt_conditioning(
                            prompt_audio,
                            use_prompt_prefill=False,
                            speaker_scale=self.speaker_scale,
                        )
                # detach 切断计算图（纯推理条件，不需要梯度）后存入缓存。
                cached_g_cond = prompt_conditioning.g_cond.detach()
                cache[prompt_cache_key] = cached_g_cond
                logger.info(
                    "Double streaming prompt conditioning cached: path={} device={} "
                    "dtype={} speaker_scale={}",
                    prompt_cache_key[0],
                    self.device,
                    self._dtype,
                    self.speaker_scale,
                )
            else:
                logger.info(
                    "Double streaming prompt conditioning cache hit: path={} device={} "
                    "dtype={} speaker_scale={}",
                    prompt_cache_key[0],
                    self.device,
                    self._dtype,
                    self.speaker_scale,
                )
            self._g_cond = cached_g_cond

        logger.info(
            "Double streaming session started: prefix_token_count={} precision={} "
            "ode_method={} num_steps={} guidance_scale={} speaker_scale={} max_audio_patch_count={} "
            "initial_silence_audio_tokens={} has_ref_audio_only={}",
            len(self._prefix_token_ids),
            runtime.precision,
            self.ode_method,
            self.num_steps,
            self.guidance_scale,
            self.speaker_scale,
            self.max_generate_length,
            self._initial_silence_audio_tokens,
            self._g_cond is not None,
        )

    @property
    def is_finished(self) -> bool:
        """会话是否已彻底结束（``finish_text`` 跑完且 vocoder 已 flush）。"""
        return self._closed

    def push_text_token(self, text_token: int) -> torch.Tensor | None:
        """喂入一个 text token，推进状态机一步，并尝试吐出一个 audio chunk。

        语义 / Semantics
        ----------------
        每次只接收单个 text token：先把它（首次还要带上流式前缀）喂进 LLM 更新
        条件，再解码恰好一个 audio patch、过流式 vocoder。返回值是该步 vocoder 输出
        的波形 chunk，形状约 (B=1, 1, samples)；若本步 vocoder 还没攒够一个输出窗口
        则返回 ``None``——属正常情况，调用方应跳过、继续喂下一个 token。

        前置条件 / Preconditions
        ------------------------
        - 会话未 close（否则 ``_ensure_active`` 报错）。
        - 尚未 ``finish_text``（``_text_finished`` 为假）。
        - 尚未触发 EOS（``_state.end_flag`` 为假）；一旦 EOS，应改调 ``finish_text``
          把音频尾部 flush 出来。
        """
        self._ensure_active()
        if self._text_finished:
            raise RuntimeError("Cannot push text tokens after finish_text().")
        if self._state.end_flag:
            raise RuntimeError(
                "Double streaming generation has already reached EOS. "
                "Call finish_text() to flush the remaining audio tail."
            )

        token_id = int(text_token)
        if not self._started:
            # 首个 token：在它前面拼上流式前缀，一并进 LLM 建立交错语境。
            chunk_token_ids = [*self._prefix_token_ids, token_id]
            self._started = True
        else:
            chunk_token_ids = [token_id]

        self._consume_text_chunk(chunk_token_ids)  # 文本侧：更新 LLM KV cache + FM 条件
        return self._decode_audio_chunk()  # 音频侧：解码 1 个 patch，过 vocoder 出 chunk

    def finish_text(self):
        """收尾：喂入 text-end 标记，把 EOS 之后的音频尾部 flush 干净。

        这是一个 generator，按顺序产出剩余 audio chunk。三步 / Three phases：

        1. **告知模型文本已完**：喂入 ``text_cond_end_id``，让 LLM 知道没有更多文本
           条件、可以收尾。若一个 text token 都没 push 过（``_started`` 为假），这里
           顺带补上流式前缀。
        2. **尾部续解码（tail decode）**：在文本已结束但模型尚未自报 EOS 时，持续解
           码 audio patch 直到 ``end_flag`` 置位。这一阶段用
           ``continue_audio_span=True``，因为不再有新 text token 来推进交错序列，需要
           手动把上一步的 LLM hidden 追加进 FM buffer，让连续的纯音频 span 接得上。
        3. **vocoder flush + 关闭**：把 vocoder 滑动窗口里残留的尾帧解码出来，置
           ``_closed``。

        若进入本方法时 ``end_flag`` 已置位（push 阶段已自然到达 EOS），则跳过 1/2，
        只标记 ``_text_finished`` 并直接走 flush。
        """
        self._ensure_active()

        if not self._state.end_flag:
            if not self._text_finished:
                text_end_chunk = [self.model.core.text_cond_end_id]
                if not self._started:
                    # 极端情况：没 push 过任何 text token 就直接 finish——补前缀。
                    text_end_chunk = [*self._prefix_token_ids, *text_end_chunk]
                    self._started = True
                self._consume_text_chunk(text_end_chunk)
                self._text_finished = True

            # 尾部 flush：文本已停、模型还在出音频，逐 patch 解码到 EOS。
            while not self._state.end_flag:
                audio_chunk = self._decode_audio_chunk(continue_audio_span=True)
                if audio_chunk is not None:
                    yield audio_chunk
        else:
            # 进入时已 EOS：无需再喂 text-end / 续解码，仅记标志后走 vocoder flush。
            self._text_finished = True

        # 排空 vocoder 内部滑动窗口的残余样本（final=True，不再保留 lookahead）。
        final_chunk = self.model.vocoder.stream_flush(self._vocoder_state)
        self._closed = True
        logger.info(
            "Double streaming session finished: decoded_patch_count={}",
            self._decoded_patch_count,
        )
        if final_chunk.size(-1) > 0:
            yield final_chunk

    def _ensure_active(self) -> None:
        """守卫：会话已关闭则拒绝任何后续操作。"""
        if self._closed:
            raise RuntimeError("Double streaming session is already closed.")

    def _consume_text_chunk(self, token_ids: list[int]) -> None:
        """把一小段 text token 喂进 LLM，更新 KV cache 与 FM 条件 buffer。

        ``token_ids`` 包装成 (1, T) 的 long 张量（batch=1）。``position=0`` /
        ``next_audio_position=schedule.size(1)`` 表示把整段都当文本消费——增量场景下
        每个 chunk 都是一小段纯文本，没有内嵌 audio 占位，故文本窗口就是全长。底层
        ``_consume_text_schedule`` 会 step LLM 并把产出的 hidden 投影后追加到 FM
        sequence，作为后续声学头解码的条件。
        """
        schedule = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self._use_amp,
            ):
                self.model._consume_text_schedule(
                    schedule,
                    position=0,
                    next_audio_position=schedule.size(1),
                    state=self._state,
                )

    def _get_initial_silence_audio_patch(
        self,
        patch_index: int,
        audio_patch: torch.Tensor,
    ) -> torch.Tensor:
        """返回第 ``patch_index`` 个"静音 latent patch"，用于给开头垫静音。

        为什么这么设计 / Why
        --------------------
        起播时若直接用模型第一拍输出，常出现突兀起音或前几帧被截。这里把一段零波形
        过 AudioVAE 编码、再 ``normalize`` 到 DiT/FM 所在的 latent 空间，得到形状与真
        实 audio patch 完全一致的"静音 patch"，前几个 patch 用它顶替，让开头平滑。

        返回 / Returns
        --------------
        形状 (1, patch_size, latent_dim) 的张量（与 ``audio_patch`` 单帧对齐），``clone``
        出来以免调用方原地改动污染缓存。结果按
        (device, dtype, patch_size, latent_dim, cache_count) 缓存在 runtime 上复用。
        """
        cache = getattr(self.runtime, "_double_streaming_silence_audio_patch_cache", None)
        if cache is None:
            cache = {}
            setattr(self.runtime, "_double_streaming_silence_audio_patch_cache", cache)

        cache_count = 10  # 一次性预算 10 个静音 patch（与 initial_silence 上限对齐）
        patch_size = int(self.model.core.latent_patch_size)
        key = (
            str(self.device),
            str(self._dtype),
            patch_size,
            int(audio_patch.size(-1)),  # latent 维度（VAE 输出含 mean+logvar，下面只取前半）
            cache_count,
        )
        cached_patches = cache.get(key)
        if cached_patches is None:
            # 构造一段零波形：cache_count 个 patch × patch_size 帧 × hop_size 样本/帧。
            hop_size = int(getattr(self.model.vocoder, "hop_size", 1))
            zero_samples = cache_count * patch_size * hop_size
            zero_audio = torch.zeros(
                (1, 1, zero_samples),
                device=self.device,
                dtype=torch.float32,
            )
            silence_latents = self.model.vocoder.extract_latents(zero_audio)
            # VAE latent 沿通道维含 [mean | logvar]，只保留前半（mean 部分）作为确定性 latent。
            silence_latents, _ = torch.split(
                silence_latents,
                int(audio_patch.size(-1)),
                dim=1,
            )
            silence_latents = silence_latents.transpose(1, 2)  # (1, C, T) -> (1, T, C)
            target_frames = cache_count * patch_size
            if silence_latents.size(1) < target_frames:
                # 帧数不足（VAE 下采样可能少一两帧）：右侧补零凑到 target_frames。
                silence_latents = torch.cat(
                    [
                        silence_latents,
                        silence_latents.new_zeros(
                            (
                                silence_latents.size(0),
                                target_frames - silence_latents.size(1),
                                silence_latents.size(2),
                            )
                        ),
                    ],
                    dim=1,
                )
            silence_latents = silence_latents[:, :target_frames, :]  # 截多余帧（也可能恰好相等）
            # normalize：搬到 DiT/FM 的白化 latent 空间，与模型输出的 audio_patch 同空间。
            cached_patches = self.model.core.io_helper.normalize(silence_latents)
            cached_patches = cached_patches.to(device=self.device, dtype=audio_patch.dtype)
            # 折成 (1, cache_count, patch_size, latent_dim)，便于按 patch_index 取单个 patch。
            cached_patches = cached_patches.reshape(
                1,
                cache_count,
                patch_size,
                int(audio_patch.size(-1)),
            ).detach()
            cache[key] = cached_patches
            logger.info(
                "Double streaming initial silence cache built: patches={} patch_size={} "
                "hop_size={} device={} dtype={}",
                cache_count,
                patch_size,
                hop_size,
                self.device,
                audio_patch.dtype,
            )
        # clone：返回独立副本，避免下游原地写改污染共享缓存。
        return cached_patches[:, int(patch_index)].clone()

    def _consume_audio_patch(self, audio_patch: torch.Tensor) -> None:
        """把刚解码出的 audio patch 回灌进解码 state（更新 LLM/FM 历史）。

        薄封装：转调 ``model._consume_audio_patch``——它会把 patch 经 patch_encoder
        编成 LLM 输入嵌入、step 一次 LLM 更新 KV cache，并把 latent 追加进 FM 历史
        buffer，使下一拍声学解码能以"已生成音频"为条件（autoregressive 闭环）。
        """
        self.model._consume_audio_patch(self._state, audio_patch=audio_patch)

    def _decode_audio_chunk(self, *, continue_audio_span: bool = False) -> torch.Tensor | None:
        """解码恰好一个 audio patch 并过流式 vocoder，返回波形 chunk（或 None）。

        这是会话的核心单步：声学头采样 -> （可选）静音替换 -> 回灌 state ->
        denormalize -> vocoder 流式一步。

        参数 / Params
        -------------
        - ``continue_audio_span``：是否手动把当前 LLM hidden 再追加进 FM buffer。
          push 路径下 ``False``——下一个 text token 会自然推进交错序列；finish_text 的
          尾部续解码路径下 ``True``——没有新文本来推进，需要手动续上纯音频 span，否则
          下一拍声学解码缺少 hidden 条件。

        返回 / Returns
        --------------
        vocoder 这一步的波形 chunk，形如 (1, 1, samples)；窗口未攒够时返回 ``None``。

        关键步骤 / Key steps
        --------------------
        - **先判 EOS 再解码**：``_should_stop_after_current_audio`` 用"上一拍"的 LLM
          hidden 算 EOS 概率，决定"这一拍出完就停"，故先取标志、解码完再置 end_flag。
        - **声学头 + CFG**：``_decode_next_audio`` 在连续 latent 空间用 flow-matching
          DiT 预测 velocity field，按 ``ode_method``/``num_steps`` 走 ODE solver，并用
          ``guidance_scale`` 做 classifier-free guidance (CFG)、``g_cond`` 注入说话人。
        - **denormalize**：把 DiT 白化空间的 latent 还原到 AudioVAE latent 空间，再
          transpose 成 vocoder 期望的 (1, C, T) 喂 ``stream_step``。
        """
        if self._decoded_patch_count >= self.max_generate_length:
            # 安全阀：到达最大 patch 数仍未 EOS，宁可报错也不无限生成。
            raise RuntimeError(
                "Double streaming exceeded max_generate_length before reaching EOS."
            )

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self._use_amp,
            ):
                # 基于上一拍 hidden 预判"本拍出完即停"；end_flag 推迟到本拍结束才置。
                stop_after_current_audio = self.model._should_stop_after_current_audio(
                    self._state,
                    eos_threshold=self.eos_threshold,
                )
                # 声学头：flow-matching DiT + ODE solver + CFG，得到一个 latent patch。
                audio_patch = self.model._decode_next_audio(
                    self._state,
                    device=self.device,
                    g_cond=self._g_cond,
                    ode_method=self.ode_method,
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                )
                if self._decoded_patch_count < self._initial_silence_audio_tokens:
                    # 开头若干 patch 用静音 latent 顶替模型输出，平滑起播。
                    audio_patch = self._get_initial_silence_audio_patch(
                        self._decoded_patch_count,
                        audio_patch,
                    )
                self._consume_audio_patch(audio_patch)  # 回灌：把本 patch 作为下一拍的条件
                if continue_audio_span:
                    # 尾部续解码无新文本：手动续上纯音频 span 的 hidden 条件。
                    self.model._append_hidden_chunk(self._state, self._state.llm_hiddens)
                self._decoded_patch_count += 1
                # 还原到 VAE latent 空间，并转成 (1, C, T) 喂流式 vocoder。
                latent_patch = self.model.core.io_helper.denormalize(audio_patch)
                audio_chunk = self.model.vocoder.stream_step(
                    latent_patch.transpose(1, 2),
                    self._vocoder_state,
                )
                if stop_after_current_audio:
                    self._state.end_flag = True  # 本拍已出完，正式置 EOS

        # vocoder 可能因 lookahead 窗口尚未填满而本步无输出——返回 None，调用方跳过。
        if audio_chunk.size(-1) == 0:
            return None
        return audio_chunk


class DotsTtsRuntimeDoubleStreaming(DotsTtsRuntime):
    """在 ``DotsTtsRuntime`` 上追加 double-streaming 入口的运行时子类。

    继承全部权重加载与普通生成能力，仅新增 ``start_double_streaming`` 工厂方法用于
    开一个 ``DoubleStreamingSession``；其余推理 API（``generate`` / ``generate_stream``）
    与基类一致。
    """

    def start_double_streaming(
        self,
        *,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        initial_silence_audio_tokens: int = 1,
    ) -> DoubleStreamingSession:
        """构造并返回一个 ``DoubleStreamingSession``。

        参数原样转发给会话构造器（见 ``DoubleStreamingSession.__init__`` 的说明）。
        典型用法：``session = runtime.start_double_streaming(prompt_audio_path=...)``，
        随后多次 ``session.push_text_token(tok)``，最后 ``for chunk in
        session.finish_text(): ...``。
        """
        return DoubleStreamingSession(
            self,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            speaker_scale=speaker_scale,
            eos_threshold=eos_threshold,
            initial_silence_audio_tokens=initial_silence_audio_tokens,
        )


__all__ = ["DotsTtsRuntimeDoubleStreaming", "DoubleStreamingSession"]
