# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Modulate by shift and scale."""
    if scale is None and shift is None:
        return x
    if shift is None:
        return x * (1 + scale.unsqueeze(1))  # type: ignore[union-attr]
    if scale is None:
        return x + shift.unsqueeze(1)  # type: ignore[union-attr]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_qkv_projections(
    attn: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    *,
    make_contiguous: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Shared fused/unfused QKV (+ optional added-KV) projection helper.

    Used by FLUX / FLUX.2 / Qwen-Image attention blocks that expose the same
    ``to_qkv`` / ``to_added_qkv`` packing flags. ``use_fused_qkv`` is always
    set by those blocks' constructors, and ``use_fused_added_qkv`` whenever
    ``added_kv_proj_dim`` is not ``None`` — direct attribute access so a
    renamed flag fails loudly instead of silently unfusing.

    ``make_contiguous=False`` preserves zero-copy views for a caller that can
    consume packed projection output strides directly.
    """
    if attn.use_fused_qkv:
        qkv, _ = attn.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        if make_contiguous:
            query, key, value = [t.contiguous() for t in (query, key, value)]
    else:
        query, _ = attn.to_q(hidden_states)
        key, _ = attn.to_k(hidden_states)
        value, _ = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        if attn.use_fused_added_qkv:
            added_qkv, _ = attn.to_added_qkv(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = added_qkv.chunk(3, dim=-1)
            if make_contiguous:
                encoder_query, encoder_key, encoder_value = [
                    t.contiguous() for t in (encoder_query, encoder_key, encoder_value)
                ]
        else:
            encoder_query, _ = attn.add_q_proj(encoder_hidden_states)
            encoder_key, _ = attn.add_k_proj(encoder_hidden_states)
            encoder_value, _ = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value
