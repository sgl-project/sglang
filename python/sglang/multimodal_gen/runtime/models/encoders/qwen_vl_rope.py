# SPDX-License-Identifier: Apache-2.0
"""Shared SRT rotary embedding adapter for Qwen-VL text encoders."""

from typing import Any

import torch

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.utils.hf_transformers.common import get_rope_config


def build_qwen_vl_text_rope(
    config: Any, *, mrope_interleaved: bool = False
) -> RotaryEmbedding:
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    rope_theta, rope_scaling = get_rope_config(config)
    rope_scaling = dict(rope_scaling or {})
    rope_scaling["mrope_interleaved"] = mrope_interleaved
    return get_rope(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position=config.max_position_embeddings,
        base=rope_theta,
        is_neox_style=True,
        rope_scaling=rope_scaling,
    )


def apply_qwen_vl_text_rope(
    rotary_emb: RotaryEmbedding,
    position_ids: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply three-axis MRoPE to batched attention tensors."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(
            "Qwen-VL query and key must have shape [batch, heads, sequence, head_dim]"
        )
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(
            "Qwen-VL text position_ids must have shape [3, batch, sequence]"
        )
    batch_size, num_query_heads, sequence_length, head_dim = query.shape
    key_batch_size, num_key_value_heads, key_sequence_length, key_head_dim = key.shape
    if (key_batch_size, key_sequence_length, key_head_dim) != (
        batch_size,
        sequence_length,
        head_dim,
    ):
        raise ValueError("Qwen-VL query and key shapes are incompatible")
    if tuple(position_ids.shape[1:]) != (batch_size, sequence_length):
        raise ValueError("Qwen-VL position_ids do not match the attention input")

    query = query.transpose(1, 2).reshape(-1, num_query_heads * head_dim)
    key = key.transpose(1, 2).reshape(-1, num_key_value_heads * head_dim)
    # Preserve HF's bf16 arithmetic order; fused MRoPE changes generated images.
    query, key = rotary_emb.forward_native(position_ids.reshape(3, -1), query, key)
    query = query.view(batch_size, sequence_length, num_query_heads, head_dim)
    key = key.view(batch_size, sequence_length, num_key_value_heads, head_dim)
    return query.transpose(1, 2), key.transpose(1, 2)
