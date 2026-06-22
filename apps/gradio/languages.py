"""Gradio 演示界面的语言/口音清单与映射 (UI language & accent registry for the Gradio demo).

本文件做什么 (What this file does)
    纯数据 + 一个辅助函数,定义 dots.tts Gradio Web UI 里"语言"下拉框的全部可选项。
    它把**给人看的中文显示名** (如 "普通话"、"英语") 映射到**模型内部用的语言/口音代码**
    (如 "ZH"、"EN", 或口音串 "口音:粤语")。这些代码会被拼进送给 TTS 模型的文本/提示
    (prompt) 里,用来条件化 (condition) 生成对应语种/口音的语音。

数据流里的位置 (Where it sits in the pipeline)
    属于推理侧的**前端配置层**,不参与 flow-matching / DiT / AudioVAE 等任何核心计算。
    调用方 (Gradio app) 读取本文件得到下拉选项 -> 用户选一项 -> 选中的 code 字符串随
    用户输入文本一起交给自回归主干 (Qwen2.5-1.5B backbone) 做条件生成。改这里只影响
    "界面能选哪些语言/口音"以及"选项对应什么代码",不改任何声学建模逻辑。

关键对象清单 (Key objects)
    - SUPPORTED_LANGUAGE_CODE_BY_NAME: dict[str, str], 中文显示名 -> 内部语言/口音代码。
    - build_language_choice_items(): 把上面的 dict 转成 Gradio Dropdown 需要的
      (label, value) 元组列表,并在最前面补一个"不指定"空选项。
"""

from __future__ import annotations

# 中文显示名 -> 内部语言/口音代码的映射表 (display name -> internal language/accent code)。
# 取值有两类:
#   1) ISO 639 风格的语言代码,如 "ZH"(中文)、"EN"(英语)、"ES"(西班牙语) —— 选定语种;
#   2) "口音:xxx" 形式的口音串(如 "口音:粤语""口音:东北话") —— 在中文语种下进一步指定口音/方言。
# 该 dict 的**插入顺序即下拉框里的显示顺序**(Python 3.7+ dict 保序),
# 因此这里把普通话/常见口音和高频语种排在前面,长尾小语种排在后面。
SUPPORTED_LANGUAGE_CODE_BY_NAME = {
    "普通话": "ZH",
    "粤语": "口音:粤语",
    "北京话": "口音:北京官话",
    "东北话": "口音:东北话",
    "四川话": "口音:四川话",
    "闽南话": "口音:闽南话",
    "吴语": "口音:吴语",
    "英语": "EN",
    "西班牙语": "ES",
    "印地语": "HI",
    "阿拉伯语": "AR",
    "孟加拉语": "BN",
    "葡萄牙语": "PT",
    "俄语": "RU",
    "日语": "JA",
    "法语": "FR",
    "德语": "DE",
    "韩语": "KO",
    "意大利语": "IT",
    "土耳其语": "TR",
    "越南语": "VI",
    "印尼语": "ID",
    "乌尔都语": "UR",
    "波斯语": "FA",
    "泰米尔语": "TA",
    "泰卢固语": "TE",
    "菲律宾语": "FIL",
    "马来语": "MS",
    "旁遮普语": "PA",
    "马拉地语": "MR",
    "古吉拉特语": "GU",
    "马拉雅拉姆语": "ML",
    "卡纳达语": "KN",
    "波兰语": "PL",
    "乌克兰语": "UK",
    "荷兰语": "NL",
    "泰语": "TH",
    "罗马尼亚语": "RO",
    "斯瓦希里语": "SW",
    "希伯来语": "HE",
    "捷克语": "CS",
    "希腊语": "EL",
    "匈牙利语": "HU",
    "瑞典语": "SV",
    "丹麦语": "DA",
    "芬兰语": "FI",
    "书面挪威语": "NB",
    "斯洛伐克语": "SK",
    "斯洛文尼亚语": "SL",
    "塞尔维亚语": "SR",
    "波斯尼亚语": "BS",
    "克罗地亚语": "HR",
    "保加利亚语": "BG",
    "马其顿语": "MK",
    "立陶宛语": "LT",
    "拉脱维亚语": "LV",
    "爱沙尼亚语": "ET",
    "冰岛语": "IS",
    "爱尔兰语": "GA",
    "威尔士语": "CY",
    "加泰罗尼亚语": "CA",
    "加利西亚语": "GL",
    "奥克语": "OC",
    "阿斯图里亚斯语": "AST",
    "尼泊尔语": "NE",
    "信德语": "SD",
    "奥里亚语": "OR",
    "阿萨姆语": "AS",
    "普什图语": "PS",
    "缅甸语": "MY",
    "高棉语": "KM",
    "老挝语": "LO",
    "哈萨克语": "KK",
    "乌兹别克语": "UZ",
    "吉尔吉斯语": "KY",
    "塔吉克语": "TG",
    "阿塞拜疆语": "AZ",
    "格鲁吉亚语": "KA",
    "亚美尼亚语": "HY",
    "白俄罗斯语": "BE",
    "卢森堡语": "LB",
    "马耳他语": "MT",
    "毛利语": "MI",
    "南非荷兰语": "AF",
    "祖鲁语": "ZU",
    "科萨语": "XH",
    "约鲁巴语": "YO",
    "豪萨语": "HA",
    "伊博语": "IG",
    "阿姆哈拉语": "AM",
    "奥罗莫语": "OM",
    "北索托语": "NSO",
    "尼扬贾语": "NY",
    "修纳语": "SN",
    "索马里语": "SO",
    "卢干达语": "LG",
    "林加拉语": "LN",
    "卢奥语": "LUO",
    "坎巴语": "KAM",
    "翁本杜语": "UMB",
    "富拉语": "FF",
    "沃洛夫语": "WO",
    "中库尔德语": "CKB",
    "宿务语": "CEB",
    "佛得角克里奥尔语": "KEA",
    "蒙古语": "MN",
    "爪哇语": "JV",
}


def build_language_choice_items() -> list[tuple[str, str]]:
    """构建 Gradio Dropdown 的 (label, value) 选项列表 (build the dropdown choice list).

    职责 (Responsibility)
        把 SUPPORTED_LANGUAGE_CODE_BY_NAME 这个 dict 摊平成 Gradio Dropdown 的
        ``choices`` 期望的格式 —— 一串 (显示文本, 实际取值) 元组,并在最前面插入一个
        "不指定" -> "" 的默认/兜底项。

    返回 (Returns)
        list[tuple[str, str]]: 每个元素是 (label, code)。第 0 项固定为 ("不指定", "")
        —— 即"不显式指定语言/口音",code 为空串,交由模型自行判断;其余项按映射表
        的插入顺序排列。

    设计要点 (Design note)
        空串 "" 作为哨兵值 (sentinel) 表示"未选语言",下游据此跳过语言条件化;
        放在列表首位让它成为下拉框的默认选中项。这里用展开 (``*[...]``) 是为了把
        "不指定"项与映射表项拼成同一层列表,而非嵌套列表。
    """
    return [("不指定", ""), *[(name, code) for name, code in SUPPORTED_LANGUAGE_CODE_BY_NAME.items()]]
