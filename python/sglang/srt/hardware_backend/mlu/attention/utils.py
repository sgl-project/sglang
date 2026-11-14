"""Shared validation for MLU attention backends."""

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.radix_attention import AttentionType

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


def validate_attention(layer: "RadixAttention", sinks: torch.Tensor | None) -> None:
    """Reject unsupported attention options before writing the KV cache."""
    if sinks is not None:
        raise NotImplementedError("MLU attention does not support attention sinks.")
    if layer.logit_cap != 0:
        raise NotImplementedError("MLU attention does not support nonzero logit_cap.")
    if layer.is_cross_attention:
        raise NotImplementedError("MLU attention does not support cross-attention.")
    if layer.attn_type != AttentionType.DECODER:
        raise NotImplementedError(
            "MLU attention only supports causal decoder self-attention."
        )
