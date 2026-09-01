# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""LongCat-Image breakable CUDA graph (BCG) prompt handling."""

from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)


def is_longcat_image_transformer(current_model: Any, call_kwargs: dict) -> bool:
    return (
        bcg_utils.transformer_class_name_matches(current_model, "longcatimage")
        and "encoder_hidden_states" in call_kwargs
        and "txt_ids" in call_kwargs
    )


def keep_longcat_prompt_shape(
    call_kwargs: dict, _current_model: Any, _buckets: tuple[int, ...]
) -> dict:
    # LongCat always supplies the complete 512-token prompt body to its DiT.
    # Generic bucket padding would only create larger, unused graph signatures.
    return call_kwargs


bcg_utils.register_prompt_padder(
    is_longcat_image_transformer, keep_longcat_prompt_shape
)
