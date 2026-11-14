"""Normalization weights shared by fused boundary providers."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm


def fused_norm_gamma(layernorm: torch.nn.Module) -> Optional[torch.Tensor]:
    """The multiplier as applied -- GemmaRMSNorm's pre-folded w + 1. None declines."""
    if isinstance(layernorm, GemmaRMSNorm):
        return layernorm.gemma_weight
    if isinstance(layernorm, RMSNorm) and layernorm.has_weight:
        return layernorm.weight
    return None
