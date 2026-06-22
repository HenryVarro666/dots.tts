"""文本前处理工具 / Text front-end utilities.

本文件负责把"原始用户文本"加工成可直接喂给 TTS 主干 (Qwen2.5-1.5B 自回归
backbone) 的 prompt 文本。它是推理数据流里最靠前的一环：

    用户文本 → [本文件] 语言检测 + TN 规整 + 语言标签注入 → tokenizer → AR backbone
             → flow-matching DiT 声学头 → AudioVAE 解码 → 波形

它只做"文本层面"的处理，不碰任何 latent / audio token / 张量；下游的声学建模
在别处。这里干三件事：

  1. 语言识别 (language detection)：用 ``lingua`` 统计模型从纯文本里猜语种，
     输出 ISO 639 语言码 (如 "zh" / "en" / "yue")。
  2. 语言码规范化 (normalization)：用 ``langcodes`` 把五花八门的语言写法
     (大小写、区域后缀、宏语言/macrolanguage 等) 收敛成统一大写码 (如 "ZH")。
  3. TN (text normalization / 文本规整)：用 WeTextProcessing 的中/英 Normalizer
     把数字、符号、缩写等"非朗读形式"展开成"可朗读形式"
     (如 "3.14" → "三点一四"，"$5" → "five dollars")。

关键函数清单 / Key functions:
  - ``detect``                 : 纯文本 → ISO 639 语言码 (lingua)。
  - ``normalize_language_code``: 任意语言写法 → 统一大写码 (langcodes)。
  - ``attach_language_tag``    : 把 ``[语言码]`` 前缀注入 prompt，给模型语种提示。
  - ``detect_text_language``   : 粗分类成 zh / en / unknown，用于选 TN 走哪条。
  - ``normalize_text`` 及中英专用版本 : 调用对应 Normalizer 做 TN。

设计上各重对象 (Normalizer / detector) 都用 ``lru_cache`` 懒加载并复用，因为
它们构建一次代价不小 (要加载规则/模型)，而推理过程会被反复调用。
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

# langcodes：把任意语言标识 (BCP-47 / ISO 639 / 自然语言名) 解析成结构化 Language，
# 用于把"语言写法"规范化成统一码；与 lingua 的 Language 同名，故 import 时改名避歧义。
from langcodes import Language as LangcodesLanguage
# lingua：基于统计 n-gram 的离线语言检测器，从纯文本里猜语种 (无需联网/API)。
from lingua import Language, LanguageDetectorBuilder
# WeTextProcessing 的中/英 TN (text normalization) Normalizer：把数字、日期、符号等
# 展开成可朗读文本。两套规则相互独立，按检测出的语种二选一调用。
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

# 粗粒度语种标签：只区分中文 / 英文 / 其它，用来决定 TN 走哪套 Normalizer。
TextLanguage = Literal["zh", "en", "unknown"]

# 把任意连续空白 (空格/换行/制表符) 折叠成单个空格，TN 输出后的清洗用。
_WHITESPACE_PATTERN = re.compile(r"\s+")


@lru_cache(maxsize=1)
def get_chinese_text_normalizer() -> ZhNormalizer:
    """懒加载并缓存中文 TN Normalizer (单例)。

    构建 Normalizer 需加载一整套规则，代价不小；推理中会被反复调用，故用
    ``lru_cache(maxsize=1)`` 保证全进程只构建一次、之后命中缓存直接复用。
    """
    return ZhNormalizer()


@lru_cache(maxsize=1)
def get_english_text_normalizer() -> EnNormalizer:
    """懒加载并缓存英文 TN Normalizer (单例)。理由同中文版。"""
    return EnNormalizer()


@lru_cache(maxsize=1)
def get_language_detector():
    """懒加载并缓存 lingua 语言检测器 (单例)。

    用 ``Language.all()`` 启用 lingua 支持的全部语种参与判别，覆盖面最广。
    按 ``language.name`` 排序只是为了让传参顺序确定 (结果与顺序无关)，便于复现。
    检测器构建同样昂贵 (要装载各语种的 n-gram 统计模型)，故同样缓存为单例。
    """
    supported_languages = tuple(
        sorted(Language.all(), key=lambda language: language.name)
    )
    return LanguageDetectorBuilder.from_languages(*supported_languages).build()


def _lingua_language_to_code(language: Language | None) -> str | None:
    """把 lingua 的 ``Language`` 枚举转成小写 ISO 639 字符串码。

    优先用两字母 639-1 码 (如 "zh"/"en")，缺失时退到三字母 639-3 码，再不行
    才退回语种英文名。``iso_code_639_*`` 是枚举对象，其 ``.name`` 才是码字符串，
    故用 ``getattr(..., "name", None)`` 取值并防御性兜底。
    """
    if language is None:
        return None
    iso_code_639_1 = getattr(language.iso_code_639_1, "name", None)
    if iso_code_639_1:
        return iso_code_639_1.lower()
    iso_code_639_3 = getattr(language.iso_code_639_3, "name", None)
    if iso_code_639_3:
        return iso_code_639_3.lower()
    return language.name.lower()


def detect(text: str) -> str | None:
    """检测文本语种，返回小写 ISO 639 码 (如 "zh"/"en"/"yue") 或 ``None``。

    空白文本无从判别，直接返回 ``None``。否则交给缓存的 lingua 检测器，
    把它给出的最可能语种映射成字符串码。纯离线、无副作用。
    """
    stripped = text.strip()
    if not stripped:
        return None
    language = get_language_detector().detect_language_of(stripped)
    return _lingua_language_to_code(language)


def normalize_language_code(language: str | None) -> str | None:
    """把任意语言写法规范化成统一的大写 ISO 639 码，失败返回 ``None``。

    接受的输入很杂：BCP-47 标签 ("zh-CN")、纯码 ("EN"/"en")、甚至自然语言名
    ("中文"/"English")。本函数把它们都收敛成大写主码 (如 "ZH"/"EN"/"YUE")，
    供 ``attach_language_tag`` 拼成 prompt 前缀。

    处理顺序与几个特判:
      - 空 / "none" / "unknown" → 视为"无明确语种"，返回 ``None``。
      - 已是"口音:xxx"形式 → 这是上层自定义的口音提示 (非标准 ISO 码)，原样透传，
        不交给 langcodes (它解析不了)。
      - 否则依次尝试 ``langcodes`` 的两个解析器:
          ``get``  按规范 (BCP-47) 严格解析；
          ``find`` 按语言名等模糊查找作兜底。
        任一抛异常就跳到下一个 (try/except continue)。
      - ``prefer_macrolanguage()``：把方言/个体语言折叠到宏语言 (macrolanguage)，
        例如普通话个体码会归并到中文宏语言 "zh"，保证标签粒度一致。
      - 取 ``.language`` 主码并大写；"UND" (undetermined/未定) 视同失败，继续兜底。
    """
    if language is None:
        return None

    stripped = language.strip()
    if not stripped or stripped.lower() in {"none", "unknown"}:
        return None
    if stripped.startswith("口音:"):  # 自定义口音提示，绕过 langcodes 直接透传
        return stripped

    for resolver in (LangcodesLanguage.get, LangcodesLanguage.find):
        try:
            # prefer_macrolanguage: 把个体语言/方言折叠到宏语言，统一标签粒度。
            normalized_language = resolver(stripped).prefer_macrolanguage()
        except Exception:
            continue  # 该解析器解析不了就换下一个，不让异常冒泡

        language_code = (normalized_language.language or "").strip().upper()
        if language_code and language_code != "UND":  # "UND"=未能确定，不算有效结果
            return language_code
    return None


def attach_language_tag(text: str, language: str | None) -> str:
    """在文本前注入 ``[语言码]`` 前缀，给 AR backbone 一个显式的语种提示。

    这是把"语言信息"喂进模型的方式：训练时同样在文本前带这种标签，所以推理时
    加上它能让模型从第一个 token 起就锚定到目标语种 (类似一个软性 control token)。

    规则:
      - 空文本 / 无法规范化出语种 → 原样返回，不加任何前缀。
      - 粤语 (YUE) 特例：改用 "口音:粤语" 而非 "[YUE]"，因为模型把粤语当作中文的
        一种"口音"来建模 (与 ``normalize_language_code`` 里"口音:"透传分支呼应)。
      - 幂等性：若文本已带该前缀则不重复添加 (上层可能多次调用)。

    Returns:
        形如 ``"[ZH]你好"`` 的带标签文本；不满足条件时返回原 ``text``。
    """
    if not text:
        return text

    language_code = normalize_language_code(language)
    if language_code is None:  # 解析不出语种就不强加标签，避免误导模型
        return text

    if language_code == "YUE":  # 粤语按"口音"建模，而非独立 [YUE] 标签
        language_code = "口音:粤语"

    language_tag = f"[{language_code}]"
    if text.startswith(language_tag):  # 幂等：已带前缀则不重复拼接
        return text
    return f"{language_tag}{text}"


def detect_text_language(text: str) -> TextLanguage:
    """把 ``detect`` 的细粒度语言码收敛成 zh / en / unknown 三类。

    只关心"该走哪套 TN Normalizer"，所以非中英语种一律归到 "unknown"
    (对应不做语种特定 TN、原样返回的分支)。
    """
    language_code = detect(text)
    if language_code == "zh":
        return "zh"
    if language_code == "en":
        return "en"
    return "unknown"


def _normalize_with(normalizer, text: str) -> str:
    """用给定 Normalizer 做 TN，并把输出里的连续空白折叠成单空格、去首尾空白。

    TN 展开数字/符号后可能引入多余空白 (尤其英文)，统一清洗保证 prompt 整洁。
    """
    normalized = normalizer.normalize(text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_chinese_text(text: str) -> str:
    """对文本强制走中文 TN (调用方已知是中文时用)。空白文本返回空串。"""
    stripped = text.strip()
    if not stripped:
        return ""
    return _normalize_with(get_chinese_text_normalizer(), stripped)


def normalize_english_text(text: str) -> str:
    """对文本强制走英文 TN (调用方已知是英文时用)。空白文本返回空串。"""
    stripped = text.strip()
    if not stripped:
        return ""
    return _normalize_with(get_english_text_normalizer(), stripped)


def normalize_text(text: str) -> str:
    """自动判语种再做 TN 的总入口：检测 → 选对应 Normalizer。

    流程：去首尾空白 → ``detect_text_language`` 粗分类 → 中文走中文 TN、
    英文走英文 TN；其它语种 (unknown) 没有专用规则，原样返回去白后的文本，
    避免用错语种的 TN 规则把文本改坏。空白输入统一返回空串。
    """
    stripped = text.strip()
    if not stripped:
        return ""

    language = detect_text_language(stripped)
    if language == "zh":
        return _normalize_with(get_chinese_text_normalizer(), stripped)
    if language == "en":
        return _normalize_with(get_english_text_normalizer(), stripped)
    return stripped  # 非中英：无专用 TN 规则，保持原文不动
