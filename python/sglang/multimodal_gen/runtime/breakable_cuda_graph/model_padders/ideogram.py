# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""Ideogram-4 breakable CUDA graph (BCG) prompt padding."""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)
from sglang.multimodal_gen.runtime.layers.attention import DynamicVarlenMaskMeta

_SEQUENCE_PADDING_INDICATOR = -1
_OUTPUT_IMAGE_INDICATOR = 2
_LLM_TOKEN_INDICATOR = 3
_DYNAMIC_MASK_META_ATTR = "_sglang_bcg_ideogram_attn_mask_meta"


def is_ideogram_transformer(current_model: Any, call_kwargs: dict) -> bool:
    return (
        bcg_utils.transformer_class_name_matches(current_model, "ideogram")
        and "llm_features" in call_kwargs
        and "x" in call_kwargs
        and "indicator" in call_kwargs
        and "position_ids" in call_kwargs
    )


def _unwrap_model(current_model: Any) -> Any:
    for attr in ("module", "_orig_mod"):
        wrapped = getattr(current_model, attr, None)
        if wrapped is not None:
            current_model = wrapped
    return current_model


def _dynamic_mask_meta(current_model: Any) -> DynamicVarlenMaskMeta:
    model = _unwrap_model(current_model)
    meta = getattr(model, _DYNAMIC_MASK_META_ATTR, None)
    if not isinstance(meta, DynamicVarlenMaskMeta):
        meta = DynamicVarlenMaskMeta()
        setattr(model, _DYNAMIC_MASK_META_ATTR, meta)
    return meta


def _first_indicator(call_kwargs: dict) -> torch.Tensor | None:
    indicator = bcg_utils.first_tensor(call_kwargs.get("indicator"))
    if not torch.is_tensor(indicator) or indicator.dim() < 2:
        return None
    return indicator


def _text_and_image_lengths(indicator: torch.Tensor) -> tuple[int, int] | None:
    row = indicator[0]
    if not torch.any(row == _LLM_TOKEN_INDICATOR):
        return None
    image_positions = (row == _OUTPUT_IMAGE_INDICATOR).nonzero(as_tuple=False)
    if image_positions.numel() == 0:
        return None
    text_seq = int(image_positions[0].item())
    if text_seq <= 0:
        return None
    image_seq = int(row.numel()) - text_seq
    if image_seq <= 0:
        return None
    return text_seq, image_seq


def _pad_total_dim(obj: Any, *, source: int, target: int, value: float = 0) -> Any:
    return bcg_utils.pad_nested_dim(
        obj, dim=1, source=source, target=target, value=value
    )


def pad_ideogram_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    indicator = _first_indicator(call_kwargs)
    if indicator is None:
        return call_kwargs

    lengths = _text_and_image_lengths(indicator)
    if lengths is None:
        return call_kwargs
    text_seq, image_seq = lengths

    bucket = bcg_utils.select_text_bucket(text_seq, buckets)
    if bucket is None:
        return call_kwargs

    source_total = text_seq + image_seq
    target_total = bucket + image_seq
    out = dict(call_kwargs)

    if source_total < target_total:
        for key in ("llm_features", "x"):
            if key in out and out[key] is not None:
                out[key] = _pad_total_dim(
                    out[key], source=source_total, target=target_total
                )
        if out.get("position_ids") is not None:
            out["position_ids"] = _pad_total_dim(
                out["position_ids"], source=source_total, target=target_total
            )
        if out.get("segment_ids") is not None:
            out["segment_ids"] = _pad_total_dim(
                out["segment_ids"],
                source=source_total,
                target=target_total,
                value=_SEQUENCE_PADDING_INDICATOR,
            )
        if out.get("indicator") is not None:
            out["indicator"] = _pad_total_dim(
                out["indicator"], source=source_total, target=target_total
            )
        if out.get("attn_mask") is not None:
            out["attn_mask"] = _pad_total_dim(
                out["attn_mask"], source=source_total, target=target_total
            )

    if out.get("attn_mask") is not None:
        out["attn_mask_meta"] = _dynamic_mask_meta(current_model)

    return out


bcg_utils.register_prompt_padder(is_ideogram_transformer, pad_ideogram_prompt_kwargs)
