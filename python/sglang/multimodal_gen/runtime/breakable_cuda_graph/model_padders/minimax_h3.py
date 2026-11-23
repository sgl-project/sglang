# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""MiniMax H3 breakable CUDA graph packed-prompt padding."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)


@dataclass
class H3PackedPrompt:
    """Static inputs for one request branch; only step-varying kwargs are copied on replay."""

    bounds: tuple[int, ...]
    text_len: int
    capture_varlen: bool = False
    offsets: tuple[torch.Tensor, torch.Tensor] | None = None
    by_bucket: dict[int, dict] = field(default_factory=dict)
    source_psp: Any = None


live_packed_prompt: ContextVar[H3PackedPrompt | None] = ContextVar(
    "h3_live_packed_prompt", default=None
)


def is_minimax_h3_transformer(current_model: Any, call_kwargs: dict) -> bool:
    return (
        bcg_utils.transformer_class_name_matches(current_model, "minimaxh3")
        and "prompt_embeds" in call_kwargs
        and "packed_seq_params" in call_kwargs
        and "refiner_packed_seq_params" in call_kwargs
        and "text_pos_info" in call_kwargs
    )


def _position_ids(info: Any) -> torch.Tensor | None:
    if isinstance(info, dict):
        ids = info.get("position_ids")
    else:
        ids = getattr(info, "position_ids", None)
    return ids if torch.is_tensor(ids) else None


def _replace_position_ids(info: Any, ids: torch.Tensor) -> dict[str, Any]:
    if isinstance(info, dict):
        return {**info, "position_ids": ids}
    # H3 currently passes dictionaries. Avoid mutating an unknown request
    # object if an alternate frontend supplies one.
    return {"position_ids": ids}


def _single_sequence(psp, used: int, total: int) -> bool:
    if not isinstance(psp, dict):
        return False
    host = psp.get("cu_seqlens_q_host")
    cu = psp.get("cu_seqlens_q")
    return (
        isinstance(host, (tuple, list))
        and all(type(v) is int for v in host)
        and (
            tuple(host) == (0, used, total)
            or (used == total and tuple(host) == (0, used))
        )
        and torch.is_tensor(cu)
        and cu.ndim == 1
        and len(host) == cu.numel()
        and cu.dtype == torch.int32
        and type(psp.get("max_seqlen_q")) is int
        and psp["max_seqlen_q"] in (used, max(used, total - used))
    )


def supports_live_packed_bounds(call_kwargs: dict) -> bool:
    prompt = call_kwargs.get("prompt_embeds")
    text_pos = _position_ids(call_kwargs.get("text_pos_info"))
    img_pos = _position_ids(call_kwargs.get("img_pos_info"))
    audio_pos = _position_ids(call_kwargs.get("audio_pos_info"))
    x = call_kwargs.get("x")
    if (
        not torch.is_tensor(prompt)
        or prompt.dim() < 2
        or not torch.is_tensor(text_pos)
        or not torch.is_tensor(img_pos)
        or not torch.is_tensor(audio_pos)
        or not torch.is_tensor(x)
        or x.dim() != 3
    ):
        return False

    text_len = int(prompt.shape[0])
    # All used H3 rows are disjoint text, image/video, or audio rows. Derive
    # the used/media lengths from tensor shapes so padding itself does not
    # perform a GPU-to-host .item() synchronization on every denoising step.
    media_rows = int(img_pos.numel()) + int(audio_pos.numel())
    used = text_len + media_rows
    source_seq = int(x.shape[1])
    # H3's native branch is one live sequence and an optional alignment tail.
    # Keep independently packed documents and alternate frontends intact.
    packed = call_kwargs.get("packed_seq_params")
    refiner = call_kwargs.get("refiner_packed_seq_params")
    return not (
        x.shape[0] != 1
        or text_len <= 0
        or text_pos.numel() != text_len
        or source_seq < used
        or not _single_sequence(packed, used, source_seq)
        or not _single_sequence(refiner, text_len, text_len)
        or packed.keys()
        - {
            "cu_seqlens_q",
            "cu_seqlens_q_host",
            "max_seqlen_q",
            "txt_len",
            "video_spans",
        }
        or refiner.keys() - {"cu_seqlens_q", "cu_seqlens_q_host", "max_seqlen_q"}
    )


def _with_tail(cu: torch.Tensor) -> torch.Tensor:
    return cu if cu.numel() == 3 else torch.cat((cu, cu[-1:]))


def pad_minimax_h3_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    branch = live_packed_prompt.get()
    if branch is None:
        # Unscoped callers must retain all host metadata in their signature.
        return call_kwargs
    text_len = branch.text_len
    bucket = bcg_utils.select_text_bucket(text_len, buckets)
    if bucket is None:
        return call_kwargs

    if bucket not in branch.by_bucket:
        prompt = call_kwargs["prompt_embeds"]
        text_pos = _position_ids(call_kwargs["text_pos_info"])
        used, total = branch.bounds[1], branch.bounds[-1]
        # Canonical [0, live, total] buffers allow aligned and padded prompts
        # to share a graph. The same offsets serve every warmup bucket.
        if branch.offsets is None:
            packed_cu = call_kwargs["packed_seq_params"]["cu_seqlens_q"]
            refiner_cu = call_kwargs["refiner_packed_seq_params"]["cu_seqlens_q"]
            branch.offsets = (_with_tail(packed_cu), _with_tail(refiner_cu))
        packed_cu, refiner_cu = branch.offsets
        updates = {
            "packed_seq_params": {
                "cu_seqlens_q": packed_cu,
                "max_seqlen_q": total,
                "bcg_live_bounds": True,
                "bcg_capture_varlen": branch.capture_varlen,
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": refiner_cu,
                "max_seqlen_q": bucket,
            },
        }
        if call_kwargs.get("refined_prompt_embeds_length") is not None:
            # Only the presence of refined embeddings affects the graph. The
            # eager embedding break reads the request's actual host length.
            updates["refined_prompt_embeds_length"] = bucket
        if text_len < bucket:
            updates["prompt_embeds"] = bcg_utils.pad_tensor_dim(
                prompt, dim=0, target=bucket
            )
            dummy = torch.arange(
                used,
                used + bucket - text_len,
                dtype=text_pos.dtype,
                device=text_pos.device,
            )
            updates["text_pos_info"] = _replace_position_ids(
                call_kwargs["text_pos_info"], torch.cat((text_pos.view(-1), dummy))
            )
        branch.by_bucket[bucket] = updates

    out = {**call_kwargs, **branch.by_bucket[bucket]}
    out.pop("local_embedding_layout", None)
    return out


bcg_utils.register_prompt_padder(
    is_minimax_h3_transformer, pad_minimax_h3_prompt_kwargs
)
