"""文本 tokenize 阶段 + 生成调度构建 / Text tokenization stage + generation-schedule builder.

本文件做什么 / What this file does
==================================
dots.tts 是连续潜在(continuous-latent)自回归 TTS:文本与音频在**同一条 token 序列**里交给
Qwen2.5-1.5B 自回归主干。但音频并不是离散 token —— 序列里出现的 `<|audio_gen_span|>` 只是一个
**占位 token(placeholder)**,标记"此处该由 DiT 声学头预测一个连续 latent patch",真正的声学信息
走 flow-matching 连续空间,不在文本 token 词表里。本文件负责把"一段文本 + 一个模板(template)"
铺成这样一条混合 token 序列,并产出训练所需的 input_ids / labels / loss_mask,或推理所需的
generation schedule(预先排好"哪几步该生成音频 patch、哪几步是文本条件")。

模板占位符 / Template placeholders
  - `{text}`       : 此处插入条件文本的 token_ids(loss=0,不参与 LM 损失,只作条件)。
  - `{audio}`      : 一整段连续音频区:start 占位 + N 个 span 占位 + end 占位(非流式 / non-streaming)。
  - `{interleave}` : 文本 token 与音频 span 占位**逐个交错**铺开(流式 / double-streaming 用)。
  模板里其余的普通字符串(如 `[文本]`、`[文本对应语音]` 前缀)按 literal 直接 encode。

两种产物 / Two outputs
  1. build_tokenized_example —— 训练样本:full_ids 错位一位切成 (input_ids, labels),并给每个
     位置打 loss_mask;只有"模型该自己生成的音频 span"位置 loss=1,文本/特殊符 loss=0。
  2. build_generation_schedule —— 推理调度:把模板展开成定长 schedule_ids(audio span 个数固定为
     max_audio_tokens),推理引擎据此知道每一步是"喂已知文本 token"还是"让 DiT 吐一个 latent patch"。

在数据流里的位置 / Position in the data flow
  [文本 + 波形] -> [preprocessing:对齐/补零, 算出 num_audio_tokens] -> [本文件:tokenize/排 schedule]
               -> [AR 主干消费 token 序列, DiT 在 audio span 位置预测 latent] -> [AudioVAE 解码成波形]

关键类/函数清单 / Key classes & functions
  - ParsedTemplate / TokenizedTemplatePart : 解析后的模板结构与单个已 tokenize 的片段(frozen dataclass)。
  - parse_template / _iter_tokenized_template_parts : 解析模板 + 逐片段产出 token_ids。
  - build_tokenized_example   : 训练样本(input_ids/labels/loss_mask)。
  - build_generation_schedule : 推理调度(schedule_ids + 是否 interleave)。
  - _append_interleave_* (两个) : interleave 模式下文本/音频交错铺设的内核(训练版带 loss_mask,推理版纯排序)。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from loguru import logger

from dots_tts.utils.tokenizer import (
    AUDIO_GEN_END_TOKEN,
    AUDIO_GEN_SPAN_TOKEN,
    AUDIO_GEN_START_TOKEN,
    TEXT_COND_END_TOKEN,
    require_token_id,
)

# 模板分词正则:优先整体匹配三个占位符,否则贪婪吞掉一段不含 '{' 的普通文本。
# 交替顺序保证占位符先于 `[^\{]+` 被尝试;`[^\{]+` 把占位符之间的 literal 文本整段切出来。
# Tokenize the template string into placeholders ({text}/{audio}/{interleave}) and literal runs.
TEMPLATE_PATTERN = re.compile(r"\{text\}|\{audio\}|\{interleave\}|[^\{]+")


@dataclass(frozen=True)
class ParsedTemplate:
    """解析后的模板 / Parsed template.

    职责 / Role
        把模板字符串切成有序片段(parts),并预先标好它属于哪种生成模式,
        让下游无需重新扫描即可分支。frozen 以便安全缓存/复用。

    字段 / Fields
        parts: 按出现顺序排列的片段元组,元素是占位符字面量("{text}"/"{audio}"/
               "{interleave}")或一段 literal 文本。
        has_audio_placeholder:       是否含 {audio}(非流式整段音频模式)。
        has_interleave_placeholder:  是否含恰好一个 {interleave}(流式交错模式)。
    """

    parts: tuple[str, ...]
    has_audio_placeholder: bool
    has_interleave_placeholder: bool


@dataclass(frozen=True)
class TokenizedTemplatePart:
    """已 tokenize 的单个模板片段 / A single tokenized template part.

    职责 / Role
        统一表示展开后的每一个片段,使 build_* 函数能按 kind 分支处理。

    字段 / Fields
        kind:      片段类型,取 "text" / "audio" / "interleave" / "literal" 之一。
        token_ids: 该片段对应的 token id 序列(audio/interleave 占位本身不带 token,留空元组)。
        raw_text:  仅 literal 片段保留原始文本,用于后续校验(如禁止 interleave 后出现非空后缀)。
    """

    kind: str
    token_ids: tuple[int, ...] = ()
    raw_text: str | None = None


def parse_template(template: str) -> ParsedTemplate:
    """解析模板字符串为 ParsedTemplate / Parse a template string.

    职责 / Role
        用 TEMPLATE_PATTERN 切分模板,统计占位符,并强制两条互斥约束:
        {audio} 与 {interleave} 不可混用;{interleave} 至多出现一次。
        这两种模式对应完全不同的 token 铺设方式,混用会让下游语义二义。

    参数 / Args
        template: 形如 "[文本]{text}[文本对应语音]{audio}" 的模板串。

    返回 / Returns
        ParsedTemplate;若违反互斥/计数约束则抛 ValueError。
    """
    parts = tuple(re.findall(TEMPLATE_PATTERN, template))
    has_audio_placeholder = "{audio}" in parts
    interleave_count = parts.count("{interleave}")
    if has_audio_placeholder and interleave_count:
        raise ValueError("Template cannot mix audio and interleave placeholders.")
    if interleave_count > 1:
        raise ValueError(
            "Interleave generation template must contain exactly one interleave placeholder."
        )
    return ParsedTemplate(
        parts=parts,
        has_audio_placeholder=has_audio_placeholder,
        has_interleave_placeholder=interleave_count == 1,
    )


def _prepare_template_tokens(
    *, text: str, tokenizer, template: str
) -> tuple[ParsedTemplate, list[int]]:
    """一次性解析模板并 encode 条件文本 / Parse template and encode the conditioning text.

    返回 (parsed_template, text_tokens)。注意 add_special_tokens=False:特殊符(BOS/EOS、
    音频占位符等)全部由本模块显式插入,不让 tokenizer 自动加,以免破坏 token 布局。
    """
    return parse_template(template), tokenizer.encode(text, add_special_tokens=False)


def _iter_tokenized_template_parts(
    *,
    parsed_template: ParsedTemplate,
    tokenizer,
    text_tokens: list[int],
):
    """按顺序产出每个模板片段的已 tokenize 形式 / Yield tokenized parts in order.

    生成器(generator):对解析出的每个 part 判型并 yield 一个 TokenizedTemplatePart。
        - "{text}":      复用已 encode 的 text_tokens(不重复 encode)。
        - "{audio}"/"{interleave}": 仅产出占位标记,token_ids 留空(实际占位 id 在调用方按
          require_token_id 取得并插入)。
        - 其它(literal):当场 encode,并保留 raw_text 供后续校验。
    """
    for part in parsed_template.parts:
        if part == "{text}":
            yield TokenizedTemplatePart(kind="text", token_ids=tuple(text_tokens))
            continue
        if part == "{audio}":
            yield TokenizedTemplatePart(kind="audio")
            continue
        if part == "{interleave}":
            yield TokenizedTemplatePart(kind="interleave")
            continue
        yield TokenizedTemplatePart(
            kind="literal",
            token_ids=tuple(tokenizer.encode(part, add_special_tokens=False)),
            raw_text=part,
        )


def _extend_tokens_with_loss(
    *, full_ids: list[int], loss_mask: list[float], token_ids: tuple[int, ...], loss: float
) -> None:
    """把一段 token 追加进 full_ids,并给每个位置写同一个 loss 权重 / Append tokens with a uniform loss weight.

    full_ids 与 loss_mask 始终等长、逐位置对齐;loss=0 表示该位置只作条件、不计 LM 损失,
    loss=1 表示该位置是模型应当生成的目标。
    """
    full_ids.extend(token_ids)
    loss_mask.extend([loss] * len(token_ids))


def build_tokenized_example(
    *, text: str, tokenizer, template: str, num_audio_tokens: int
) -> dict[str, Any]:
    """构建一条训练样本 / Build one tokenized training example.

    职责 / Role
        把"文本 + 模板 + 音频 token 数"展开成一条混合 token 序列,产出自回归 LM 训练所需的
        input_ids / labels / loss_mask(teacher-forcing 错位一位的标准做法)。

    参数 / Args
        text:             条件文本(尚未 encode)。
        tokenizer:        文本 tokenizer,须有 eos_token_id 与 (en/de)code、convert_tokens_to_ids。
        template:         模板串,决定 {text}/{audio}/{interleave} 的布局。
        num_audio_tokens: 该样本音频 latent 切出的 audio span 个数(由 preprocessing 算出)。

    返回 / Returns
        dict,含:
          input_ids       : full_ids[:-1] —— 模型输入(长度 L-1)。
          labels          : full_ids[1:]  —— 错位一位的预测目标(下一个 token)。
          loss_mask       : 与 labels 对齐的 0/1 权重(只在音频 span 目标处为 1)。
          text_token_count: 条件文本 token 数,供统计/调度上限校验。

    设计 / Why
        音频 span 占位虽是文本词表里的特殊 id,但训练时让模型"预测下一个是 span 占位",
        从而在这些位置触发 DiT 声学头去回归连续 latent;故只有 span 位置 loss=1。
        start/end 占位与文本/特殊符不计损失(loss=0),只作结构条件。
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer eos_token_id is required for generation targets.")

    parsed_template, text_tokens = _prepare_template_tokens(
        text=text,
        tokenizer=tokenizer,
        template=template,
    )

    full_ids: list[int] = []
    loss_mask: list[float] = []
    audio_tokens: list[int] | None = None
    if parsed_template.has_audio_placeholder:
        # 非流式整段音频 / non-streaming audio block:start + N×span + end。
        # 仅在含 {audio} 时才取占位 id,避免对不需要的模板做无谓的词表查询。
        audio_gen_start_id = require_token_id(tokenizer, AUDIO_GEN_START_TOKEN)
        audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)
        audio_gen_end_id = require_token_id(tokenizer, AUDIO_GEN_END_TOKEN)
        audio_tokens = (
            [audio_gen_start_id]
            + [audio_gen_span_id] * num_audio_tokens
            + [audio_gen_end_id]
        )
    elif parsed_template.has_interleave_placeholder:
        audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)
        audio_gen_end_id = require_token_id(tokenizer, AUDIO_GEN_END_TOKEN)
        text_cond_end_id = require_token_id(tokenizer, TEXT_COND_END_TOKEN)

    for part in _iter_tokenized_template_parts(
        parsed_template=parsed_template,
        tokenizer=tokenizer,
        text_tokens=text_tokens,
    ):
        if part.kind == "text":
            _extend_tokens_with_loss(
                full_ids=full_ids,
                loss_mask=loss_mask,
                token_ids=part.token_ids,
                loss=0.0,
            )
            continue

        if part.kind == "audio":
            if audio_tokens is None:
                raise RuntimeError("Audio placeholder tokens were not initialized.")
            full_ids.extend(audio_tokens)
            # loss_mask 与 [start, span×N, end] 逐位置对齐:start=0、N 个 span=1、end=0。
            # max(0, len-2) 防御 num_audio_tokens 退化为 0(只剩 start/end)时取负长度。
            loss_mask.extend([0.0])
            loss_mask.extend([1.0] * max(0, len(audio_tokens) - 2))
            loss_mask.append(0.0)
            continue

        if part.kind == "interleave":
            _append_interleave_generation_tokens(
                full_ids=full_ids,
                loss_mask=loss_mask,
                text_tokens=text_tokens,
                num_audio_tokens=num_audio_tokens,
                audio_span_id=audio_gen_span_id,
                audio_end_id=audio_gen_end_id,
                text_cond_end_id=text_cond_end_id,
            )
            continue

        _extend_tokens_with_loss(
            full_ids=full_ids,
            loss_mask=loss_mask,
            token_ids=part.token_ids,
            loss=0.0,
        )

    full_ids.append(tokenizer.eos_token_id)
    loss_mask.append(0.0)

    # 错位一位切分 / shift-by-one:input_ids 预测 labels(下一 token);loss_mask 跟随 labels
    # (丢掉首位的 mask),保证 labels[i] 与 loss_mask[i] 指向同一个被预测位置。
    return {
        "input_ids": full_ids[:-1],
        "labels": full_ids[1:],
        "loss_mask": loss_mask[1:],
        "text_token_count": len(text_tokens),
    }


def build_generation_schedule(
    *,
    text: str,
    tokenizer,
    template: str,
    max_audio_tokens: int,
) -> dict[str, Any]:
    """构建推理生成调度 / Build the inference generation schedule.

    职责 / Role
        推理时没有真实音频长度,故用 max_audio_tokens 作为"预算"把模板展开成一条**定长**
        schedule_ids。引擎逐位置走这条调度:遇到普通文本 id 就当作已知条件喂入;遇到
        audio span 占位 id 就让 DiT 在该步预测一个连续 latent patch。与训练版的区别是
        没有 labels/loss_mask —— 它只回答"接下来每一步该做什么"。

    参数 / Args
        text:             待合成文本(尚未 encode)。
        tokenizer:        文本 tokenizer。
        template:         模板串,决定 {audio}(非流式)还是 {interleave}(流式)路径。
        max_audio_tokens: 音频 span 个数预算(>0);非流式取满 max,流式见 interleave 约束。

    返回 / Returns
        dict:{"schedule_ids": [...], "interleave": bool}。
        interleave=False 走整段音频路径,True 走文本/音频逐 token 交错路径。

    设计 / Why
        日志里只打印"可见"序列(剔除大量重复的 span 占位),便于人读;span 占位本身不可解码成
        文本,保留只会刷屏。
    """
    if max_audio_tokens <= 0:
        raise ValueError("max_audio_tokens must be positive for generation.")

    parsed_template, text_tokens = _prepare_template_tokens(
        text=text,
        tokenizer=tokenizer,
        template=template,
    )
    schedule_ids: list[int] = []
    audio_gen_start_id = require_token_id(tokenizer, AUDIO_GEN_START_TOKEN)
    audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)

    if parsed_template.has_audio_placeholder:
        # 非流式路径 / non-streaming:文本条件全部铺完后,接 start + max_audio_tokens 个 span 占位。
        # 注意这里不追加 end 占位:推理由引擎/采样自行决定何时停,而非模板预设终点。
        for part in _iter_tokenized_template_parts(
            parsed_template=parsed_template,
            tokenizer=tokenizer,
            text_tokens=text_tokens,
        ):
            if part.kind == "audio":
                schedule_ids.append(audio_gen_start_id)
                schedule_ids.extend([audio_gen_span_id] * max_audio_tokens)
                continue
            schedule_ids.extend(part.token_ids)
        # 仅为日志准备的"可见"序列:滤掉成百上千个重复的 span 占位,只留可解码的文本/特殊符。
        visible_schedule_ids = [
            token_id for token_id in schedule_ids if token_id != audio_gen_span_id
        ]
        decoded_schedule = (
            tokenizer.decode(
                visible_schedule_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if hasattr(tokenizer, "decode")
            else repr(visible_schedule_ids)
        )
        logger.info(
            "Built generation schedule: interleave={} max_audio_tokens={} sequence={!r}",
            False,
            int(max_audio_tokens),
            decoded_schedule,
        )
        return {
            "schedule_ids": schedule_ids,
            "interleave": False,
        }

    if not parsed_template.has_interleave_placeholder:
        raise ValueError(
            "Generation template must contain either {audio} or {interleave}."
        )
    text_cond_end_id = require_token_id(tokenizer, TEXT_COND_END_TOKEN)
    # 流式交错的硬约束:每个文本 token 都要配至少一个音频 span,故预算不能少于文本 token 数,
    # 否则无法把文本逐字铺进交错序列。/ at least one audio span per text token.
    if max_audio_tokens < len(text_tokens):
        raise ValueError(
            "Interleave generation requires at least one audio span per text token: "
            f"text_token_count={len(text_tokens)} "
            f"max_audio_patch_count={max_audio_tokens}."
        )

    interleave_started = False
    for part in _iter_tokenized_template_parts(
        parsed_template=parsed_template,
        tokenizer=tokenizer,
        text_tokens=text_tokens,
    ):
        if part.kind == "interleave":
            _append_interleave_schedule_tokens(
                schedule_ids=schedule_ids,
                text_tokens=text_tokens,
                max_audio_tokens=max_audio_tokens,
                audio_span_id=audio_gen_span_id,
                text_cond_end_id=text_cond_end_id,
            )
            interleave_started = True
            continue
        if part.kind == "text":
            raise ValueError(
                "Generation schedule does not support {text} inside an interleave template."
            )
        if part.kind == "audio":
            raise ValueError(
                "Generation schedule does not support {audio} inside an interleave template."
            )
        if interleave_started:
            # interleave 之后只允许出现"空白"literal(纯空格);任何非空后缀文本都无处安放,
            # 因为交错段已用 text_cond_end 收尾,故直接报错。/ no non-empty suffix after interleave.
            if (part.raw_text or "").strip():
                raise ValueError(
                    "Generation schedule does not support non-empty suffix text after the interleave placeholder."
                )
            continue
        schedule_ids.extend(part.token_ids)

    # 同上:日志只显示可见(非 span 占位)序列。/ visible-only sequence for logging.
    visible_schedule_ids = [
        token_id for token_id in schedule_ids if token_id != audio_gen_span_id
    ]
    decoded_schedule = (
        tokenizer.decode(
            visible_schedule_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if hasattr(tokenizer, "decode")
        else repr(visible_schedule_ids)
    )
    logger.info(
        "Built generation schedule: interleave={} max_audio_tokens={} sequence={!r}",
        True,
        int(max_audio_tokens),
        decoded_schedule,
    )
    return {
        "schedule_ids": schedule_ids,
        "interleave": True,
    }


def _append_interleave_generation_tokens(
    *,
    full_ids: list[int],
    loss_mask: list[float],
    text_tokens: list[int],
    num_audio_tokens: int,
    audio_span_id: int,
    audio_end_id: int,
    text_cond_end_id: int,
) -> None:
    """训练版交错铺设内核 / Training-side interleave layout.

    职责 / Role
        把文本 token 与音频 token([span×N, end])**逐拍交错**写入 full_ids,并同步打 loss_mask:
        文本与 end/text_cond_end 处 loss=0,音频 span 处 loss=1(模型应生成的目标)。

    交错规则 / Interleaving
        每一轮(while)先放一个文本 token(放完后用一次 text_cond_end 收尾文本侧),再放一个音频
        token;两侧谁先放完就只剩另一侧继续,直到两侧都耗尽。text_cond_end 标记"文本条件结束",
        在文本先放完的那一拍补上,且全程只补一次(text_cond_end_added 守卫)。

    形状 / Shapes
        text_tokens 长 T,audio_tokens 长 (num_audio_tokens + 1);本函数原地 extend,不返回。

    设计 / Why
        交错布局让自回归主干在生成早期就能"边看文本边吐音频",是 double-streaming 低延迟的基础。
    """
    audio_tokens = [audio_span_id] * num_audio_tokens + [audio_end_id]
    text_index = 0
    audio_index = 0
    text_cond_end_added = False

    # 双指针交错:每轮各推进文本侧/音频侧各一步,直到两侧都耗尽。/ two-pointer interleave.
    while text_index < len(text_tokens) or audio_index < len(audio_tokens):
        if text_index < len(text_tokens):
            full_ids.append(text_tokens[text_index])
            loss_mask.append(0.0)
            text_index += 1
        elif not text_cond_end_added:
            # 文本侧刚好放完的那一拍:插入一次 text_cond_end 作为"文本条件结束"标记(仅一次)。
            full_ids.append(text_cond_end_id)
            loss_mask.append(0.0)
            text_cond_end_added = True

        if audio_index < len(audio_tokens):
            full_ids.append(audio_tokens[audio_index])
            # 只有真正的 span 目标(前 num_audio_tokens 个)loss=1;末尾的 end 占位 loss=0。
            loss_mask.append(1.0 if audio_index < num_audio_tokens else 0.0)
            audio_index += 1

    # 边界:若音频侧比文本侧先耗尽,循环可能没机会进入上面那个 elif,这里补齐 text_cond_end。
    if not text_cond_end_added:
        full_ids.append(text_cond_end_id)
        loss_mask.append(0.0)


def _append_interleave_schedule_tokens(
    *,
    schedule_ids: list[int],
    text_tokens: list[int],
    max_audio_tokens: int,
    audio_span_id: int,
    text_cond_end_id: int,
) -> None:
    """推理版交错铺设内核 / Inference-side interleave layout.

    职责 / Role
        与训练版同构,但没有 loss_mask:每个文本 token 后紧跟一个 span 占位(text→span 成对),
        文本铺完后插 text_cond_end,再用剩余预算补满纯 span 占位。

    预算分配 / Budget
        前 len(text_tokens) 个 span 与文本一一配对,剩下 (max_audio_tokens - len(text_tokens)) 个
        span 作为"文本说完后继续生成的音频"。调用方已保证 max_audio_tokens >= len(text_tokens),
        故 remaining 不会为负;仍判 >0 以处理恰好相等(无尾部 span)的情形。
    """
    # text→span 逐对铺设:每个文本 token 立刻配一个待生成的 audio span 占位。
    for token_id in text_tokens:
        schedule_ids.append(token_id)
        schedule_ids.append(audio_span_id)
    schedule_ids.append(text_cond_end_id)
    remaining_audio_tokens = max_audio_tokens - len(text_tokens)
    if remaining_audio_tokens > 0:
        schedule_ids.extend([audio_span_id] * remaining_audio_tokens)
