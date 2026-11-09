# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""SANA-Video breakable CUDA graph (BCG) prompt handling."""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)

_DEFAULT_PROMPT_LENGTH = 300
_COMPACT_PROMPT_LENGTH = 64


def is_sana_video_transformer(current_model: Any, call_kwargs: dict) -> bool:
    encoder_hidden_states = bcg_utils.first_tensor(
        call_kwargs.get("encoder_hidden_states")
    )
    return (
        bcg_utils.transformer_class_name_matches(current_model, "sanavideo")
        and torch.is_tensor(encoder_hidden_states)
        and encoder_hidden_states.dim() >= 2
        and encoder_hidden_states.shape[1]
        in (_COMPACT_PROMPT_LENGTH, _DEFAULT_PROMPT_LENGTH)
    )


def keep_sana_video_prompt_shape(
    call_kwargs: dict, _current_model: Any, _buckets: tuple[int, ...]
) -> dict:
    """Keep SANA-Video's supported 64- and 300-token widths unchanged.

    Expanding either width to generic buckets adds cross-attention work and
    captures signatures the SANA-Video text stage does not use.
    """
    return call_kwargs


def sana_video_warmup_kwargs(call_kwargs: dict, _current_model: Any) -> dict | None:
    """Add the 300-token capture input for a compact 64-token SANA prompt."""
    embeddings = bcg_utils.first_tensor(call_kwargs.get("encoder_hidden_states"))
    if (
        not torch.is_tensor(embeddings)
        or embeddings.ndim != 3
        or embeddings.shape[1] != _COMPACT_PROMPT_LENGTH
    ):
        return None

    out = dict(call_kwargs)
    for key, ndim in (
        ("encoder_hidden_states", 3),
        ("encoder_attention_mask", 2),
        ("encoder_hidden_states_mask", 2),
    ):
        value = call_kwargs.get(key)
        tensor = bcg_utils.first_tensor(value)
        if (
            not torch.is_tensor(tensor)
            or tensor.ndim != ndim
            or tensor.shape[:2] != embeddings.shape[:2]
        ):
            return None
        out[key] = bcg_utils.pad_nested_dim(
            value,
            dim=1,
            source=_COMPACT_PROMPT_LENGTH,
            target=_DEFAULT_PROMPT_LENGTH,
            value=0,
        )
    return out


bcg_utils.register_prompt_padder(
    is_sana_video_transformer,
    keep_sana_video_prompt_shape,
    warmup_kwargs_provider=sana_video_warmup_kwargs,
)
