"""特殊 token 字典 + 取 id 辅助 / Special-token vocabulary and id lookup.

本文件做什么 / What this file does
    集中声明 dots.tts 在"文本词表里复用一批特殊占位 token"来表达音频结构,并提供
    require_token_id() 把这些字符串 token 解析成整数 id(找不到就报错,fail-fast)。
    这里只放常量与一个小工具,不含模型逻辑;真正构造序列在 data/pipelines/tokenizing.py,
    真正在 forward 里把 span 占位换成 latent patch embedding 在 models/dots_tts/{core,model}.py。

    dots.tts 是连续潜在(continuous-latent)TTS:音频本身没有离散 token,这些特殊 token
    仅作"结构标记/占位"——告诉自回归主干(Qwen2.5-1.5B)序列里哪一段是音频、是待生成的还是
    参考的,而音频内容由 flow-matching DiT 声学头在 span 位置回归连续 latent。

序列中的语义角色 / Token roles in the sequence
    两套平行的 span:gen=要生成的音频(generate),comp=已知/参考音频(complete,用于音色克隆条件)。
    每套都是 start + N×span + end 三段式:
      - *_START_TOKEN : 音频段开始标记 / opening marker,只作结构条件(训练时 loss=0)。
      - *_SPAN_TOKEN  : 一个 latent patch 的占位 / one latent-patch placeholder,共 N 个;
                        forward 时这些位置的 embedding 会被换成 latent patch embedding,
                        gen-span 也是 LM 训练目标(loss=1)→ 在此触发 DiT 回归连续 latent。
      - *_END_TOKEN   : 音频段结束标记 / closing marker。
    另有 TEXT_COND_END_TOKEN:交错(interleave)流式模板里"一段文本条件结束、音频开始"的分界,
    用于 double-streaming 下文本与音频交替排布。

关键清单 / Key symbols
    AUDIO_COMP_{START,SPAN,END}_TOKEN : 参考/已知音频段(comp)的三段式标记。
    AUDIO_GEN_{START,SPAN,END}_TOKEN  : 待生成音频段(gen)的三段式标记。
    TEXT_COND_END_TOKEN               : 交错模板中文本条件段的结束分界。
    require_token_id(tokenizer, token) : 把上述字符串 token 解析为整数 id(缺失即抛错)。
"""

from __future__ import annotations

# 下方 7 个常量是字符串字面量(<|...|>),需预先加入 tokenizer 的特殊 token 词表;
# 模型构建时会把 vocab_size 扩到 len(tokenizer) 以容纳这些新增 id。
AUDIO_COMP_START_TOKEN = "<|audio_comp_start|>"  # comp(参考音频)段开始 / start of reference-audio block
AUDIO_COMP_SPAN_TOKEN = "<|audio_comp_span|>"  # comp 段单 patch 占位,replaced by latent patch embedding(克隆条件,不计 loss)
AUDIO_COMP_END_TOKEN = "<|audio_comp_end|>"  # comp 段结束 / end of reference-audio block
AUDIO_GEN_START_TOKEN = "<|audio_gen_start|>"  # gen(待生成音频)段开始 / start of to-generate block
AUDIO_GEN_SPAN_TOKEN = "<|audio_gen_span|>"  # gen 段单 patch 占位:LM 目标 + 在此触发 DiT 回归 latent
AUDIO_GEN_END_TOKEN = "<|audio_gen_end|>"  # gen 段结束 / end of to-generate block
TEXT_COND_END_TOKEN = "<|text_cond_end|>"  # 交错流式中"文本条件段结束、音频开始"的分界 / text-cond boundary for interleaving


def require_token_id(tokenizer, token: str) -> int:
    """把特殊 token 字符串解析为整数 id,缺失即报错 / Resolve a special token to its id, fail-fast.

    职责 / Role
        查 token 在 tokenizer 词表里的整数 id;若该特殊 token 没被加进词表(返回 None 或
        哨兵负值/常见的 unk),立即抛错而非静默回退,避免后续把"找不到"当成合法 id 用,
        从而错位整条 token 序列。

    参数 / Args
        tokenizer : HF tokenizer,需提供 convert_tokens_to_ids。
        token     : 待查的特殊 token 字符串(本文件顶部那批 <|...|> 之一)。

    返回 / Returns
        int : 该 token 的词表 id。

    为什么这么设计 / Why
        span/start/end 占位的 id 是后续"按位置插入占位、再在 forward 里替换 latent
        patch embedding"的锚点;一旦取到错误 id,整个混合序列布局都会崩,所以这里宁可
        在加载期(load time)硬失败,也不让坏 id 流入训练/推理。
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    # None 或负值都视作"词表里没有这个特殊 token"(HF 对未知 token 可能返回 unk_id 或 -1)。
    if token_id is None or token_id < 0:
        raise ValueError(f"Artifact tokenizer is missing required special token: {token}")
    return int(token_id)


__all__ = [
    "AUDIO_COMP_END_TOKEN",
    "AUDIO_COMP_SPAN_TOKEN",
    "AUDIO_COMP_START_TOKEN",
    "AUDIO_GEN_END_TOKEN",
    "AUDIO_GEN_SPAN_TOKEN",
    "AUDIO_GEN_START_TOKEN",
    "TEXT_COND_END_TOKEN",
    "require_token_id",
]
