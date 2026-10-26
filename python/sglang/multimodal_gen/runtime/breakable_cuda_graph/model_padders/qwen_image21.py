# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""Qwen-Image 2.1 breakable CUDA graph (BCG) prompt padding.

Only cached steps replay graphs: prefill runs eagerly and stores each layer's
prefix K/V at its exact length, and a cached step reads nothing else of the
prompt. The padder left-pads that K/V to the text bucket, stacks it into one
tensor (one input copy per replay) and drops the prefill-only inputs, so prompts
of different lengths share one signature. The pad rows never reach attention or
RoPE: the DiT attends past them at a graph break, so replays stay bit-exact with
the unpadded eager forward.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size


def is_qwen_image21_transformer(current_model: Any, call_kwargs: dict) -> bool:
    return (
        bcg_utils.transformer_class_name_matches(current_model, "qwenimage21")
        and "prefix_caches" in call_kwargs
    )


def pad_qwen_image21_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    caches = call_kwargs["prefix_caches"]
    # only cached single-sample steps replay: prefill runs eagerly, warmup
    # captures one sample, and the SP attention path has no pad-aware break
    if (
        caches is None
        or len(caches) != 1
        or not caches[0][0]
        or get_sp_world_size() != 1
    ):
        return call_kwargs
    seq = caches[0][0]["key"].shape[1]
    # a longer prefix (e.g. an edit's condition-image tokens) runs eagerly; the
    # runner reports that miss once, so skip the per-step bucket warning
    bucket = next((bucket for bucket in buckets if seq <= bucket), None)
    if bucket is None:
        return call_kwargs
    rows = torch.stack(
        [tensor for cache in caches[0] for tensor in (cache["key"], cache["value"])]
    )
    prefix_kv = rows.new_zeros(*rows.shape[:2], bucket, *rows.shape[3:])
    prefix_kv[:, :, bucket - seq :] = rows
    return dict(
        call_kwargs,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        condition_latents=None,
        layouts=[{"target_rope": call_kwargs["layouts"][0]["target_rope"]}],
        prefix_caches=None,
        prefix_kv=prefix_kv,
        prefix_pad=torch.tensor(bucket - seq),
    )


bcg_utils.register_prompt_padder(
    is_qwen_image21_transformer, pad_qwen_image21_prompt_kwargs
)
