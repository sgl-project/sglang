"""SANA-Video BF16-input linear attention, gated by request quality.

The reference path promotes rotated Q/K and V to FP32 before both attention
GEMMs.  For ``quality="high"``, the first GEMM keeps its BF16 inputs while
requesting an FP32 output/accumulator from cuBLAS.  The second GEMM stays in
FP32.  This removes two large dtype-conversion kernels and lets the first GEMM
use BF16 Tensor Cores, at the cost of half-precision input rounding.

The default ``quality="lossless"`` path remains the original FP32-input chain
bit-for-bit.  Only the single-batch CUDA layout used by native SANA-Video is
eligible; unsupported dtypes and layouts fall back to the reference path.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

logger = logging.getLogger(__name__)

_FUSION = QualityGatedFusion(
    name="SANA-Video BF16-input linear attention",
    marker_attr="_sgl_sana_video_linear_attention_site",
    enabled_attr="_sgl_sana_video_linear_attention_enabled",
)


def mark_sana_video_linear_attention_site(module: nn.Module) -> None:
    """Mark a SANA-Video linear-attention module; it starts unmounted."""
    _FUSION.mark(module)


def sana_video_linear_attention_active(module: nn.Module) -> bool:
    """Whether the request-scoped BF16-input path is mounted on ``module``."""
    return _FUSION.is_enabled(module)


def _site_reject_reason(_site: nn.Module) -> str | None:
    if torch.version.cuda is None:
        return "CUDA is unavailable"
    return None


def mount_sana_video_linear_attention(root: nn.Module) -> bool:
    return _FUSION.mount(root, reject_reason=_site_reject_reason, logger=logger)


def unmount_sana_video_linear_attention(root: nn.Module) -> None:
    _FUSION.unmount(root)


def try_sana_video_linear_attention(
    site: nn.Module,
    query_rotate: torch.Tensor,
    key_rotate: torch.Tensor,
    value: torch.Tensor,
    normalizer: torch.Tensor,
) -> torch.Tensor | None:
    """Return the quality-gated attention result, or ``None`` to fall back."""
    if not (
        _FUSION.is_enabled(site)
        and query_rotate.is_cuda
        and query_rotate.dtype == torch.bfloat16
        and key_rotate.dtype == query_rotate.dtype
        and value.dtype == query_rotate.dtype
        and query_rotate.dim() == 4
        and query_rotate.shape == key_rotate.shape == value.shape
        and query_rotate.shape[0] == 1
        and normalizer.is_cuda
    ):
        return None

    batch_size, num_heads, head_dim, _ = value.shape
    scores = torch.bmm(
        value.flatten(0, 1),
        key_rotate.transpose(-1, -2).flatten(0, 1),
        out_dtype=torch.float32,
    ).view(batch_size, num_heads, head_dim, head_dim)
    return (scores @ query_rotate.float()) * normalizer
