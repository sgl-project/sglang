# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_customkernel import (
    apply_convrot_int8_shared_input,
    convrot_int8_shares_input,
)


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


def _project_shared_input(
    x: torch.Tensor, layers: Sequence[torch.nn.Module]
) -> list[torch.Tensor]:
    if convrot_int8_shares_input(layers):
        return apply_convrot_int8_shared_input(x=x, layers=layers)
    return [layer(x)[0] for layer in layers]


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
        query, key, value = _project_shared_input(
            x=hidden_states, layers=(attn.to_q, attn.to_k, attn.to_v)
        )

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
            encoder_query, encoder_key, encoder_value = _project_shared_input(
                x=encoder_hidden_states,
                layers=(attn.add_q_proj, attn.add_k_proj, attn.add_v_proj),
            )

    return query, key, value, encoder_query, encoder_key, encoder_value
