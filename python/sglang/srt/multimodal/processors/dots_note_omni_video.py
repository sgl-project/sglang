"""Request-scoped, train-consistent video preprocessing for dots.note.omni.

The submitted ``video_url`` is decoded into memory and passed directly to the
vendored train-consistent pipeline. No uploaded video or derived feature is
written to local or shared storage; all request data becomes collectible after
this function returns.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import threading
from typing import Any

from sglang.srt.utils import VideoData, get_video_bytes

_TOKEN_RE = re.compile(r"(<image_\d+>|<audio_\d+>)")
_PREPROCESS_LOCK = threading.Lock()


def _build_cfg(
    *,
    seq: int,
    output_reserve: int | None,
    audio_cap: float,
    audio_sr: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    if seq <= 0:
        raise ValueError(f"seq must be positive, got {seq}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    configured_reserve = seq // 4 if output_reserve is None else output_reserve
    effective_reserve = max(configured_reserve, max_new_tokens)
    if effective_reserve >= seq:
        raise ValueError(
            "output_reserve/max_new_tokens must leave room for input: "
            f"reserve={effective_reserve}, seq={seq}"
        )
    if audio_cap < 0:
        raise ValueError(f"audio_cap must be non-negative, got {audio_cap}")
    if audio_sr <= 0:
        raise ValueError(f"audio_sr must be positive, got {audio_sr}")

    process_audio = audio_cap > 0
    return {
        "process_audio": process_audio,
        "seq_length": seq - effective_reserve,
        "reserve_interleave": True,
        "audio_token_ratio_cap": float(audio_cap),
        "audio_sample_rate": int(audio_sr),
        "video_jpeg_quality": int(os.environ.get("XHS_VIDEO_JPEG_QUALITY", "85")),
    }


def _video_payload(raw_video) -> tuple[bytes, str]:
    if isinstance(raw_video, VideoData):
        raw_video = raw_video.url
    raw_url = raw_video.get("url") if isinstance(raw_video, dict) else raw_video
    video_bytes = get_video_bytes(raw_url)
    return video_bytes, hashlib.sha1(video_bytes).hexdigest()


def _cfg_for_pure_visual(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["process_audio"] = False
    return cfg


def _flat_to_content(flat: dict[str, Any]) -> list[dict[str, Any]]:
    meta = flat.get("meta", {})
    user_value = next(
        (
            conv.get("value", "")
            for conv in flat.get("conversations", [])
            if (conv.get("from") or conv.get("role")) == "user"
        ),
        "",
    )
    content: list[dict[str, Any]] = []
    last = 0
    for match in _TOKEN_RE.finditer(user_value):
        if match.start() > last:
            content.append({"type": "text", "text": user_value[last : match.start()]})
        key = match.group(1)[1:-1]
        encoded = meta.get(key)
        if encoded and key.startswith("image_"):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        elif encoded:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{encoded}"},
                }
            )
        last = match.end()
    if last < len(user_value):
        content.append({"type": "text", "text": user_value[last:]})
    return content


def preprocess_dots_video(
    raw_video,
    question: str,
    *,
    tokenizer,
    seq: int = 131072,
    output_reserve: int | None = None,
    audio_cap: float = 1.0,
    audio_sr: int = 16000,
    k_mode: str = "eval_ek",
    max_new_tokens: int = 0,
) -> list[dict[str, Any]]:
    """Return in-memory timestamp/image/audio content using the server tokenizer."""
    if not k_mode:
        raise ValueError("k_mode must not be empty")
    with _PREPROCESS_LOCK:
        video_bytes, video_id = _video_payload(raw_video)
        cfg = _build_cfg(
            seq=seq,
            output_reserve=output_reserve,
            audio_cap=audio_cap,
            audio_sr=audio_sr,
            max_new_tokens=max_new_tokens,
        )
        from sglang.srt.multimodal.processors.dots_note_omni_video_core import (
            flatten_runner,
        )
        from sglang.srt.multimodal.processors.dots_note_omni_video_core import (
            preprocess as pp,
        )

        pp.set_tokenizer(tokenizer)
        video_b64 = base64.b64encode(video_bytes).decode()
        sample = {
            "meta": {"video_0": video_b64},
            "conversations": [{"from": "user", "value": f"<video_0>{question}"}],
        }
        record_key = hashlib.sha1(f"{video_id}|{question}".encode()).hexdigest()

        def run(run_cfg):
            new_meta, conversations = pp.process_sample_video(sample, run_cfg)
            plan = flatten_runner.build_plan(
                new_meta,
                conversations,
                record_key,
                k_mode=k_mode,
                process_audio=run_cfg["process_audio"],
                audio_sample_rate=audio_sr,
            )
            return _flat_to_content(flatten_runner.render_flat(plan))

        try:
            return run(cfg)
        except pp.SkipSample as exc:
            if "audio_token_ratio_exceed" not in str(exc):
                raise
            return run(_cfg_for_pure_visual(cfg))
