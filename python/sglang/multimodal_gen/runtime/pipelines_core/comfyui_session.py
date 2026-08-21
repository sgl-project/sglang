# SPDX-License-Identifier: Apache-2.0
"""Run-scoped cache for ComfyUI conditioning that does not change every step.

The ComfyUI process keeps the sampler loop. After the first DiT call, text
embeddings / packed extras stay on the worker; later steps only need to send
latents and the timestep.
"""

from __future__ import annotations

from typing import Any

_CONDITIONING_FIELDS = (
    "prompt_embeds",
    "negative_prompt_embeds",
    "prompt_seq_lens",
    "negative_prompt_seq_lens",
    "pooled_embeds",
    "neg_pooled_embeds",
    "image_latent",
    "vae_image_sizes",
    "prompt_attention_mask",
    "negative_attention_mask",
    "prompt_embeds_mask",
    "negative_prompt_embeds_mask",
)

_SESSIONS: dict[str, dict[str, Any]] = {}


def session_id_from_req(req) -> str | None:
    extra = getattr(req, "extra", None) or {}
    sid = extra.get("comfyui_session_id")
    return sid if sid else None


def bind_comfyui_session(req):
    """Restore cached conditioning, then refresh the cache from whatever is set."""
    sid = session_id_from_req(req)
    if not sid:
        return req

    cached = _SESSIONS.get(sid)
    if cached:
        for name, value in cached.items():
            current = getattr(req, name, None)
            if _is_empty(current):
                setattr(req, name, value)

    snapshot = {}
    for name in _CONDITIONING_FIELDS:
        value = getattr(req, name, None)
        if not _is_empty(value):
            snapshot[name] = value
    if snapshot:
        _SESSIONS[sid] = snapshot
    return req


def release_comfyui_session(session_id: str | None) -> None:
    if session_id:
        _SESSIONS.pop(session_id, None)


def _is_empty(value) -> bool:
    return value is None or value == [] or value == ()
