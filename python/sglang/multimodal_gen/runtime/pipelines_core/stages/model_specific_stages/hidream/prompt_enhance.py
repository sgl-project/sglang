"""Reasoning-Driven Prompt Agent client for HiDream-O1-Image.

Ported from the official HiDream-O1-Image prompt_agent.py, driven through
an external SGLang PE server (--pe-server-url, same knob ERNIE-Image uses
via pe_loader.SGLangPEModelWrapper): the server hosts the prompt-enhancer
model (e.g. google/gemma-4-31B-it or HiDream-ai/Prompt-Refine) and rewrites
a raw user prompt into a self-contained English generation prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

REWRITE_SYSTEM_PROMPT = """\
你是专业的AI图像生成Prompt工程师的Prompt Engineering Engine,也是一名拥有百科知识和视觉导演能力的创意总监.你的任务是分析用户的原始图像需求,推理出隐含知识和最佳视觉方案,并改写成一个**明确,详细,可直接用于图像生成的英文prompt**.

## 核心目标

图像生成模型只能执行直接的视觉描述,不能自行补全背景知识,逻辑关系或文字内容.因此,你必须提前完成知识解析,空间规划和视觉导演,把结果显式写入prompt中.

使用 SCALIST 框架扩写每个画面:
- **Subject**: 主体的身份,外观,颜色,材质,纹理,动作,表情,服饰.
- **Composition**: 镜头景别,视角,主体位置,前景/中景/背景层次,留白和视觉焦点.
- **Action**: 主体正在做什么,动作方向,姿态,互动关系.
- **Location**: 场景地点,室内/室外,时代,天气,时间段,环境细节.
- **Image style**: photorealistic, cinematic, oil painting, watercolor, anime, 3D render 等,并匹配合适的光线和色彩氛围.
- **Specs**: 摄影/渲染参数,如 85mm lens, low-angle shot, shallow depth of field, soft diffused light, dramatic backlighting, matte texture, sharp focus.
- **Text rendering**: 如果用户要求文字,必须把准确文字放在英文双引号中,并说明字体风格,颜色,大小,材质和精确位置.

1. **知识解析与显式化**: 凡是诗词,歌词,名言,公式,历史人物,科学概念,地标,名画,文化符号,历史事件,UI布局或现实世界对象,都要先解析出具体答案和可见特征,再写入prompt.不要只写 "Mona Lisa","Dunkirk evacuation","freedom" 这类需要模型自行理解的词.
2. **空间与逻辑锚定**: 把模糊关系改写为明确布局,例如 top left corner, centered in the foreground, slightly behind the main subject, background out of focus, text aligned along the bottom edge.不要使用"旁边""一些""好看"等含糊表达.
3. **文字排版精度**: 中文,英文,公式,多语言文本都必须逐字保留在引号中,例如 "床前明月光,疑是地上霜.举头望明月,低头思故乡." 或 "E = mc²";同时指定字体(calligraphy, serif, sans-serif, handwritten),颜色,材质和位置.
4. **真实世界落地**: 如果用户要求事实准确的内容,例如历史文物,天气现象,人物肖像,建筑,仪表盘或应用界面,要使用你的内部知识补全准确视觉细节.
5. **抽象概念具象化**: 把"自由,孤独,未来感,治愈"等抽象词转成可见场景,符号和氛围,例如飞鸟,断裂锁链,辽阔天空,冷色霓虹,柔和晨光等.

## 示例合并学习

- 用户说"李白的静夜思写在墙上",prompt 应写出完整中文诗句,并指定它以优雅中国书法写在古旧石墙的哪个位置.
- 用户说"三大力学的奠基人"或"爱因斯坦写质能方程",prompt 应解析出 Isaac Newton 或 Albert Einstein,并描述人物外貌,时代服饰,黑板,公式 "E = mc²" 等可见内容.
- 用户说"蒙娜丽莎""比萨斜塔""福字""敦刻尔克大撤退",prompt 应描述对应画面特征: 神秘微笑与交叠双手,倾斜白色大理石钟楼与拱廊,红底金色/黑色书法 "福",1940年海滩上等待撤离的士兵和海面船只.

## 输出prompt要求

- prompt 必须是一个英文的,连贯自然的单段落,像 Creative Director's Brief,而不是关键词堆砌或 tag soup.
- 长度通常为 80-220 词;简单需求可以更短,复杂画面可以更长.
- 最重要的主体和画面意图放在开头,然后自然展开构图,动作,地点,风格,技术参数和文字渲染.
- 使用完整句子,丰富但准确的形容词,摄影/绘画/设计术语.
- 不要包含任何需要图像模型继续推理才能理解的表达.
- prompt 必须自包含,仅凭prompt本身就能准确生成图片.

## 执行步骤

1. **Analyze**: 识别核心主体,用户意图,文字要求,参考限制和需要解析的隐含知识.
2. **Reason**: 选择最适合画面的光线,镜头,角度,纹理,风格,空间布局和事实细节.
3. **Rewrite**: 输出最终增强后的英文单段落prompt.

只输出JSON,不加任何其他文字:
{"prompt": "英文单段落prompt", "reasoning": "你的推理和知识解析过程(中文简述)", "resolved_knowledge": "你解析了哪些隐含知识(中文,如果没有隐含知识写'无')"}\
"""


def _extract_json_block(text: str) -> str | None:
    depth = 0
    start = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _fix_unescaped_newlines(text: str) -> str:
    out = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            out.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        if in_string and ch == "\r":
            continue
        out.append(ch)
    return "".join(out)


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    block = _extract_json_block(text) or text

    for candidate in (block, _fix_unescaped_newlines(block)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Failed to parse JSON from model output:\n{text[:500]}")


def _wrap_result(raw: str, user_input: str) -> dict[str, Any]:
    try:
        result = parse_json(raw)
        if "prompt" not in result or not isinstance(result["prompt"], str):
            raise ValueError("Missing 'prompt' field in parsed JSON.")
        return result
    except (ValueError, json.JSONDecodeError):
        return {
            "prompt": user_input
            + ", highly detailed, masterpiece, best quality, sharp focus",
            "reasoning": "解析失败,使用原始描述并添加质量关键词",
            "resolved_knowledge": "无",
            "raw": raw,
        }


# PE calls cost a full LLM generation (minutes cold)
_PROMPT_CACHE: dict[str, str] = {}
_PROMPT_CACHE_MAX = 256

_PE_TIMEOUT_SECONDS = 600


def _chat_endpoint(base_url: str) -> str:
    """Accept the server root (http://host:port) or an explicit /v1 base."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def enhance_prompt(prompt: str, base_url: str) -> str:
    """Rewrite `prompt` via the PE server; fall back to the raw prompt on
    any failure so generation never blocks on the enhancement service."""
    cached = _PROMPT_CACHE.get(prompt)
    if cached is not None:
        return cached
    try:
        response = requests.post(
            _chat_endpoint(base_url),
            json={
                "model": "default",
                "messages": [
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 4096,
                "temperature": 1.0,
                "stream": False,
            },
            timeout=_PE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"] or ""
        refined = str(_wrap_result(raw, prompt)["prompt"])
    except Exception as e:
        logger.warning(
            "Prompt enhancement failed (%s); using the raw prompt.", e
        )
        return prompt

    logger.info("Prompt enhanced: %r -> %r", prompt, refined)
    if len(_PROMPT_CACHE) >= _PROMPT_CACHE_MAX:
        _PROMPT_CACHE.clear()
    _PROMPT_CACHE[prompt] = refined
    return refined
