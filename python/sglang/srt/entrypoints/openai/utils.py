import io
import logging
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

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


def _download_bytes(url: str, timeout: float = 10.0) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def _image_from_url(url: str) -> Image.Image:
    raw = _download_bytes(url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _audio_wave_from_url(url: str, target_sr: int = 16000):
    """Return (wave: torch.FloatTensor[1, T], sr) or (None, None) on fatal error."""
    if torch is None:
        return None, None

    # 1) try soundfile
    try:
        import soundfile as sf  # type: ignore

        raw = _download_bytes(url)
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)  # mono
        wave = torch.from_numpy(data).unsqueeze(0)  # [1, T]
        if sr != target_sr:
            import librosa  # type: ignore

            data_rs = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            wave = torch.from_numpy(data_rs).unsqueeze(0)
            sr = target_sr
        return wave.contiguous(), sr
    except Exception as e:
        logger.warning(f"[openai.adapter] soundfile decode failed: {e}")

    # 2) try torchaudio (http stream → bytes)
    try:
        import torchaudio  # type: ignore

        raw = _download_bytes(url)
        wav, sr = torchaudio.load(io.BytesIO(raw))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono [1,T]
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
            sr = target_sr
        return wav.contiguous().float(), sr
    except Exception as e:
        logger.warning(f"[openai.adapter] torchaudio decode failed: {e}")

    return None, None


def build_mm_inputs_from_openai_messages(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Parse OpenAI-style chat messages and build SGLang mm_inputs list.
    Only image/audio parts are materialized here; text is handled by prompt builder.

    Returns: List[MultimodalDataItem-like dict]
    """
    mm_items: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            ptype = part.get("type")
            # ---------- image ----------
            if ptype in ("image_url", "input_image"):
                try:
                    url = (
                        part["image_url"]["url"]
                        if "image_url" in part
                        else part.get("url")
                    )
                    img = _image_from_url(url)
                    mm_items.append(
                        {
                            "format": "image",
                            "image": img,  # schedule_batch.py 会识别 PIL.Image
                        }
                    )
                except Exception as e:
                    logger.warning(f"[openai.adapter] image_url load failed: {e}")
            # ---------- audio ----------
            elif ptype in ("audio_url", "input_audio"):
                url = None
                if "audio_url" in part and isinstance(part["audio_url"], dict):
                    url = part["audio_url"].get("url")
                elif "url" in part:
                    url = part["url"]
                if url is None:
                    logger.warning("[openai.adapter] audio part missing url")
                    continue

                wave, sr = _audio_wave_from_url(url, target_sr=16000)
                if wave is None:
                    # minimal placeholder to keep pipeline alive
                    if torch is not None:
                        wave = torch.zeros(1, 16000, dtype=torch.float32)
                        sr = 16000
                        logger.warning(
                            "[openai.adapter] use zero-audio placeholder(1s@16k)"
                        )
                    else:
                        logger.warning(
                            "[openai.adapter] torch not available; skip audio item"
                        )
                        continue

                mm_items.append(
                    {
                        "format": "audio",
                        # 关键字段：调度器需要能“看见一个张量”，用于 pad / dtype 判断
                        "feature": wave,  # shape [1, T], float32
                        "sampling_rate": sr,
                    }
                )
            # text 等别的类型留给 prompt 侧处理
    return mm_items
