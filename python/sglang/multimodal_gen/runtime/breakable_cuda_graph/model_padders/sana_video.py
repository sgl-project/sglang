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


def is_sana_video_transformer(current_model: Any, call_kwargs: dict) -> bool:
    encoder_hidden_states = bcg_utils.first_tensor(
        call_kwargs.get("encoder_hidden_states")
    )
    return (
        bcg_utils.transformer_class_name_matches(current_model, "sanavideo")
        and torch.is_tensor(encoder_hidden_states)
        and encoder_hidden_states.dim() >= 2
        and encoder_hidden_states.shape[1] == _DEFAULT_PROMPT_LENGTH
    )


def keep_sana_video_prompt_shape(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    """Keep the pipeline's fixed 300-token prompt shape unchanged.

    The SANA-Video text stage pads both CFG branches to 300 tokens by default.
    Expanding that fixed shape to the generic 512/1024 buckets only increases
    cross-attention work and captures unused graph signatures.
    """
    return call_kwargs


bcg_utils.register_prompt_padder(
    is_sana_video_transformer, keep_sana_video_prompt_shape
)
