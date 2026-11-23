# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""MiniMax H3 breakable CUDA graph packed-prompt padding."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)

# Read only inside the attention graph break. The host boundaries belong to
# the live CFG branch and must never be closed over by a captured segment.
live_packed_bounds: ContextVar[tuple[int, ...] | None] = ContextVar(
    "minimax_h3_live_packed_bounds", default=None
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


def pad_minimax_h3_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    if not supports_live_packed_bounds(call_kwargs):
        return call_kwargs
    prompt = call_kwargs["prompt_embeds"]
    text_len = prompt.shape[0]
    bucket = bcg_utils.select_text_bucket(text_len, buckets)
    if bucket is None:
        return call_kwargs
    packed = call_kwargs["packed_seq_params"]
    used = packed["cu_seqlens_q_host"][1]
    source_seq = call_kwargs["x"].shape[1]
    text_pos = _position_ids(call_kwargs["text_pos_info"])
    out = dict(call_kwargs)
    # request-local row lists have prompt-dependent shapes, so keep them out
    # of bucketed BCG signatures and rebuild the rows in the eager break
    out.pop("local_embedding_layout", None)
    # Request-static H3 denoising normally carries the live refined-text
    # length as a host integer to avoid a per-step device scalar read. Host
    # integers are baked into BCG signatures, however, so different prompt
    # lengths would miss the same text bucket. Make only the BCG-padded copy a
    # scalar tensor; the eager embedding break reads its updated replay value.
    refined_len = out.get("refined_prompt_embeds_length")
    if refined_len is not None and not torch.is_tensor(refined_len):
        out["refined_prompt_embeds_length"] = torch.tensor(
            int(refined_len),
            dtype=torch.int64,
            device=prompt.device,
        )
    if text_len < bucket:
        out["prompt_embeds"] = bcg_utils.pad_tensor_dim(prompt, dim=0, target=bucket)
        # These rows exist only to stabilize the BCG input signature. The
        # model's eager embedding break trims prompt/text_pos/refiner metadata
        # back to ``text_len`` before any projection or attention, so their
        # values never enter the model.
        dummy_text_pos = torch.arange(
            used,
            used + (bucket - text_len),
            dtype=text_pos.dtype,
            device=text_pos.device,
        )
        out["text_pos_info"] = _replace_position_ids(
            out["text_pos_info"], torch.cat((text_pos.view(-1), dummy_text_pos))
        )

    # Do not grow the main packed sequence to media_rows + bucket. Changing
    # the SP row partition changes GEMM shapes and is measurably non-bitwise
    # even though dummy rows live in an independent attention segment. A
    # capture is therefore reusable only inside the request's existing
    # packed sequence shape. Image and video releases use different alignments.
    # Host-only txt_len/video_spans are used by the stage to prepare attention
    # metadata before the DiT call. Baking them into a graph key specializes
    # every prompt; the eager attention break reads this branch's live bounds.
    out["packed_seq_params"] = {
        "cu_seqlens_q": packed["cu_seqlens_q"],
        "max_seqlen_q": source_seq,
        "bcg_live_bounds": True,
    }
    refiner_cu = torch.tensor(
        [0, text_len, bucket], dtype=torch.int32, device=prompt.device
    )
    out["refiner_packed_seq_params"] = {
        "cu_seqlens_q": refiner_cu,
        "max_seqlen_q": bucket,
    }
    return out


bcg_utils.register_prompt_padder(
    is_minimax_h3_transformer, pad_minimax_h3_prompt_kwargs
)
