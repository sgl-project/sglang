# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""MiniMax H3 breakable CUDA graph packed-prompt padding."""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)
from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
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


def _replace_psp(
    psp: Any,
    *,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> dict[str, Any]:
    if isinstance(psp, dict):
        return {
            **psp,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": max_seqlen_q,
        }
    return {
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": max_seqlen_q,
    }


def _aligned(value: int) -> int:
    alignment = MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    return (value + alignment - 1) // alignment * alignment


def pad_minimax_h3_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    prompt = bcg_utils.first_tensor(call_kwargs.get("prompt_embeds"))
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
        return call_kwargs

    text_len = int(prompt.shape[0])
    bucket = bcg_utils.select_text_bucket(text_len, buckets)
    if bucket is None:
        return call_kwargs

    # All used H3 rows are disjoint text, image/video, or audio rows. Derive
    # the used/media lengths from tensor shapes so padding itself does not
    # perform a GPU-to-host .item() synchronization on every denoising step.
    media_rows = int(img_pos.numel()) + int(audio_pos.numel())
    used = text_len + media_rows
    source_seq = int(x.shape[1])
    if source_seq < _aligned(used):
        return call_kwargs

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
    # 64-row packed-sequence alignment group; other groups safely miss the
    # signature and run eager.
    packed_cu = torch.tensor([0, used, source_seq], dtype=torch.int32, device=x.device)
    out["packed_seq_params"] = _replace_psp(
        out["packed_seq_params"],
        cu_seqlens_q=packed_cu,
        # FA accepts an upper bound; keeping this bucket-stable is required
        # because non-tensor values are baked into the BCG signature.
        max_seqlen_q=source_seq,
    )
    refiner_cu = torch.tensor(
        [0, text_len, bucket],
        dtype=torch.int32,
        device=prompt.device,
    )
    out["refiner_packed_seq_params"] = _replace_psp(
        out["refiner_packed_seq_params"],
        cu_seqlens_q=refiner_cu,
        max_seqlen_q=bucket,
    )
    return out


bcg_utils.register_prompt_padder(
    is_minimax_h3_transformer, pad_minimax_h3_prompt_kwargs
)
