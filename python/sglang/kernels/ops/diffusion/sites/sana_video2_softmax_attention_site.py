"""SANA-Video 2.0 BF16-input softmax attention, gated by request quality.

The reference path promotes Q, K and V to FP32 before scaled-dot-product
attention, which limits PyTorch to the FP32 memory-efficient kernel.  Those
inputs are BF16-valued already (RMSNorm and RoPE both cast back to the
activation dtype), so for ``quality="high"`` the anchor layers hand them to
SDPA as BF16 and flash attention becomes eligible.  QK^T, softmax and the PV
accumulation stay FP32 inside the kernel; the probabilities are rounded to
BF16 before the PV GEMM where the reference keeps them in FP32.

The default ``quality="exact"`` path remains the FP32-input chain bit-for-bit.
Only BF16 CUDA inputs are eligible; anything else falls back to the reference
path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion


def _site_reject_reason(_site: nn.Module) -> str | None:
    if torch.version.cuda is None:
        return "CUDA is unavailable"
    return None


SANA_VIDEO2_SOFTMAX_ATTENTION = QualityGatedFusion(
    name="SANA-Video 2.0 BF16-input softmax attention",
    marker_attr="_sgl_sana_video2_softmax_attention_site",
    enabled_attr="_sgl_sana_video2_softmax_attention_enabled",
    reject_reason=_site_reject_reason,
)


def try_sana_video2_softmax_attention(
    site: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor | None:
    """Return the quality-gated attention result, or ``None`` to fall back."""
    if not (
        SANA_VIDEO2_SOFTMAX_ATTENTION.is_enabled(site)
        and query.is_cuda
        and query.dtype == torch.bfloat16
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.dim() == 4
        and query.shape == key.shape == value.shape
    ):
        return None
    return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
