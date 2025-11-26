import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

logger = logging.getLogger(__name__)


from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
)

try:
    import torch
except Exception:
    torch = None  # type: ignore


import base64
from io import BytesIO

logger = logging.getLogger(__name__)


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs


def parse_openai_message_content(message: dict):
    """
    输入：OpenAI 风格的 message（dict）
    输出：
      processed_msg: 去掉多模态 payload 后、仅含文本/工具的消息 dict（可给 apply_chat_template 用）
      img_items: 要传给 Engine 的 image_data 列表（PIL.Image 或 dict）
      vid_items: 要传给 Engine 的 video_data 列表（占位，当前可空）
      aud_items: 要传给 Engine 的 audio_data 列表（dict，格式为 {"format":"audio_url","url":...} 等）
      modalities: ["image", "audio", ...] 的并集
    约定：
      - image: 用 PIL.Image 直接传（Engine 会处理）
      - audio: 不去提前解码，交给后端处理，保证 item 里不要混入 image 字段，避免被当作图像分支处理
    """
    role = message.get("role", "user")
    content = message.get("content", "")

    processed_msg = {"role": role, "content": ""}
    img_items, vid_items, aud_items = [], [], []
    modalities = []

    # 文本消息（纯 str）
    if isinstance(content, str):
        processed_msg["content"] = content
        return processed_msg, img_items, vid_items, aud_items, modalities

    # 结构化 content（list）
    out_text_chunks = []
    if isinstance(content, list):
        for part in content:
            ptype = part.get("type")
            # ------- 文本 -------
            if ptype == "text":
                t = part.get("text", "")
                if t:
                    out_text_chunks.append(t)

            # ------- 图片 -------
            elif ptype == "image_url":
                image_url = (part.get("image_url") or {}).get("url")
                image_b64 = (part.get("image_url") or {}).get("b64_json")

                if image_b64:  # base64 内联
                    try:
                        raw = base64.b64decode(image_b64)
                        img = Image.open(BytesIO(raw)).convert("RGB")
                        img_items.append(img)
                        modalities.append("image")
                    except Exception as e:
                        logger.warning(f"decode image b64 failed: {e}")
                elif image_url:
                    try:
                        # 让后端离线/无网也能跑：如果失败就把 URL 回退成占位 dict，后端可自行下载/报错
                        resp = requests.get(image_url, timeout=10)
                        resp.raise_for_status()
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                        img_items.append(img)
                        modalities.append("image")
                    except Exception as e:
                        logger.warning(f"download image failed: {e}; pass URL through")
                        # 兜底：有的后端支持 {"format":"image_url", "url": ...}
                        img_items.append({"format": "image_url", "url": image_url})
                        modalities.append("image")

            # ------- 音频 -------
            elif ptype == "audio_url":
                audio_url = (part.get("audio_url") or {}).get("url")
                if audio_url:
                    # 关键：音频必须进入 audio_data，而不是 image_data
                    # 交给下游解码；不要带 pixel_values/feature 字段，避免走图像 pad 推断分支
                    aud_items.append({"format": "audio_url", "url": audio_url})
                    modalities.append("audio")

            # 其它类型（例如 video_url 等）：按需扩展
            else:
                # 忽略未知 part 或在此扩展
                pass

    processed_msg["content"] = "\n".join(out_text_chunks) if out_text_chunks else ""
    return processed_msg, img_items, vid_items, aud_items, modalities


def process_hidden_states_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[List]:
    """Process hidden states from a ret item in non-streaming response.

    Args:
        ret_item: Response item containing meta_info
        request: The original request object

    Returns:
        Processed hidden states for the last token, or None
    """
    if not request.return_hidden_states:
        return None

    hidden_states = ret_item["meta_info"].get("hidden_states", None)
    if hidden_states is not None:
        hidden_states = hidden_states[-1] if len(hidden_states) > 1 else []
    return hidden_states


def normalize_openai_messages_for_mm(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    规范化多模态消息的 content 顺序：
      - 把每条 message 中的图片段(image_url/input_image)移到文本段之后
      - 为所有图片段补充 modalities="multi-images"（用于下游选择 pad 策略/占位策略）
      - 严格保留“出现顺序”→ 注入 model_specific_data.original_index（若缺失）
    这样，后续按 content 顺序插入 <image> 占位时，offset 将位于文本之后（>= prefix_len）。
    """
    normed = []
    for msg in messages:
        content = msg.get("content", None)
        if not isinstance(content, list):
            normed.append(msg)
            continue

        text_parts, image_parts, other_parts = [], [], []
        local_img_idx = 0
        for part in content:
            ptype = part.get("type")
            if ptype in ("image_url", "input_image"):
                # 标注 multi-images
                part = dict(part)
                part.setdefault("modalities", "multi-images")
                # 注入 original_index，用于与 mm_items 对齐（如果上层没给）
                msd = part.setdefault("model_specific_data", {})
                msd.setdefault("original_index", local_img_idx)
                image_parts.append(part)
                local_img_idx += 1
            elif ptype == "text":
                text_parts.append(part)
            else:
                other_parts.append(part)

        # 文本优先，其次其它（如tool等），最后图片
        new_content = text_parts + other_parts + image_parts
        new_msg = dict(msg)
        new_msg["content"] = new_content
        normed.append(new_msg)
    return normed


def build_mm_inputs_from_openai_messages(messages: List[dict]) -> List[dict]:
    import os
    from io import BytesIO

    import requests
    from PIL import Image

    mm_items: List[dict] = []
    for m in messages or []:
        content = m.get("content", [])
        if not isinstance(content, list):
            continue

        local_img_idx = 0
        for part in _iter_structured_content(content):
            ptype = _get(part, "type", None)

            if ptype in ("image_url", "input_image"):
                image_url_obj = _get(part, "image_url", None) or _get(part, "url", None)
                url = (
                    _get_url_from_image_obj(image_url_obj)
                    if image_url_obj is not None
                    else _get(part, "url", None)
                )
                if not url:
                    continue

                def _append_image(img):
                    mm_items.append(
                        {
                            "format": "image",
                            "image": img,
                            "model_specific_data": {"original_index": local_img_idx},
                        }
                    )

                try:
                    # 在线下载
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    _append_image(img)
                except Exception:
                    # 离线回退1：如果 url 看起来是本地路径，直接打开
                    if os.path.exists(url):
                        try:
                            img = Image.open(url).convert("RGB")
                            _append_image(img)
                        except Exception:
                            mm_items.append(
                                {
                                    "format": "image_url",
                                    "url": url,
                                    "model_specific_data": {
                                        "original_index": local_img_idx
                                    },
                                }
                            )
                    else:
                        # 离线回退2：支持镜像目录，把 URL 文件名映射到本地
                        mirror = os.getenv("SGLANG_IMAGE_MIRROR_DIR")
                        fname = os.path.basename(url)
                        local_path = os.path.join(mirror, fname) if mirror else None
                        if local_path and os.path.exists(local_path):
                            try:
                                img = Image.open(local_path).convert("RGB")
                                _append_image(img)
                            except Exception:
                                mm_items.append(
                                    {
                                        "format": "image_url",
                                        "url": url,
                                        "model_specific_data": {
                                            "original_index": local_img_idx
                                        },
                                    }
                                )
                        else:
                            # 实在不行，仍保留 URL，但一定要带 original_index
                            mm_items.append(
                                {
                                    "format": "image_url",
                                    "url": url,
                                    "model_specific_data": {
                                        "original_index": local_img_idx
                                    },
                                }
                            )
                local_img_idx += 1

            elif ptype in ("audio_url", "input_audio"):
                url = _get_nested(part, "audio_url", "url", default=None) or _get(
                    part, "url", None
                )
                mm_items.append(
                    {
                        "format": "audio",
                        "feature": None,
                        "url": url,
                    }
                )

    return mm_items


def split_text_and_images_messages(messages):
    # 不重排、不合并：仅把 image_url 统一成 {"type":"image_url","image_url":{"url":...}}
    out = []
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, list):
            out.append({"role": role, "content": content})
            continue
        normalized_parts = []
        for part in content:
            if not isinstance(part, dict):
                normalized_parts.append(part)
                continue
            t = part.get("type")
            if t in ("image_url", "input_image"):
                url_obj = part.get("image_url") or part.get("url")
                url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
                if url:
                    p = {"type": "image_url", "image_url": {"url": url}}
                    normalized_parts.append(p)
                else:
                    normalized_parts.append(part)
            else:
                normalized_parts.append(part)
        out.append({"role": role, "content": normalized_parts})
    return out


def normalize_openai_messages_for_mm(messages):
    """轻量重排：把每条 message 内部的图片段移到文本之后（不拆 message）"""
    normed = []
    for msg in messages:
        content = msg.get("content", None)
        if not isinstance(content, list):
            normed.append(msg)
            continue
        text_parts, image_parts, other_parts = [], [], []
        local_img_idx = 0
        for part in content:
            t = part.get("type")
            if t in ("image_url", "input_image"):
                part = dict(part)
                part.setdefault("modalities", "multi-images")
                msd = part.setdefault("model_specific_data", {})
                msd.setdefault("original_index", local_img_idx)
                image_parts.append(part)
                local_img_idx += 1
            elif t == "text":
                text_parts.append(part)
            else:
                other_parts.append(part)
        new_msg = dict(msg)
        new_msg["content"] = text_parts + other_parts + image_parts
        normed.append(new_msg)
    return normed


def _maybe_to_dict(obj):
    if isinstance(obj, dict):
        return obj
    # pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return None


def _get(obj, key, default=None):
    """Safe get for dict/Pydantic/attrs-like objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    # attribute access
    try:
        val = getattr(obj, key)
        return val if val is not None else default
    except Exception:
        pass
    # fallback to dict-like
    d = _maybe_to_dict(obj)
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _get_nested(obj, *keys, default=None):
    cur = obj
    for k in keys:
        cur = _get(cur, k, default=None)
        if cur is None:
            return default
    return cur


def _ensure_openai_message(role, content):
    """Return a minimal OpenAI-style message dict."""
    return {"role": role, "content": content}


def _iter_structured_content(content):
    """Yield parts from a content that is either list[dict/obj] or a single str."""
    if isinstance(content, list):
        for p in content:
            yield p
    return


def _get_url_from_image_obj(image_obj):
    """
    image_obj could be:
      - {"url": "..."} dict
      - Pydantic object with .url
    """
    if image_obj is None:
        return None
    if isinstance(image_obj, dict):
        return image_obj.get("url", None)
    # attr style
    try:
        return getattr(image_obj, "url", None)
    except Exception:
        pass
    # fallback to dict()
    d = _maybe_to_dict(image_obj)
    if isinstance(d, dict):
        return d.get("url", None)
    return None
