"""Per-token gated-residual fusion for Helios and LingBot Video, gated by quality.

Each block applies ``residual + (gate * update).to(residual.dtype)`` at its
attention and FFN updates, where ``update`` and the per-token ``gate``
(``[B, S, 1]``) stay in FP32 while ``residual`` is BF16. The shared
``residual_gate_add`` kernel computes ``residual + update * gate`` in a single
pass but requires one dtype, so the gate and update are first cast to BF16.
That reordering of the FP32 multiply is numerically equivalent only at
half-precision rounding level (not bit-exact), so the fusion is opt-in:
``quality="lossless"`` and ``quality="high"`` mount it, while the default
``quality="exact"`` keeps the reference FP32-multiply form bit-for-bit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

HELIOS_GATED_RESIDUAL = QualityGatedFusion(
    name="Helios per-token gated residual",
    marker_attr="_sgl_helios_gated_residual_site",
    enabled_attr="_sgl_helios_gated_residual_enabled",
)
LINGBOT_VIDEO_GATED_RESIDUAL = QualityGatedFusion(
    name="LingBot Video per-token gated residual",
    marker_attr="_sgl_lingbot_video_gated_residual_site",
    enabled_attr="_sgl_lingbot_video_gated_residual_enabled",
)


def try_gated_residual(
    fusion: QualityGatedFusion,
    site: nn.Module,
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor | None:
    """Return the fused ``residual + update * gate`` when ``site`` is enabled.

    Returns ``None`` (caller runs the reference path) when the site is off or
    the tensors are not eligible for the per-token fast path.
    """
    if not fusion.is_enabled(site):
        return None
    from sglang.kernels.ops.diffusion import (
        can_use_residual_gate_add_cuda,
        residual_gate_add,
    )

    if residual.dtype != update.dtype or residual.dtype != gate.dtype:
        update = update.to(residual.dtype)
        gate = gate.to(residual.dtype)
    if not can_use_residual_gate_add_cuda(residual, update, gate):
        return None
    return residual_gate_add(residual, update, gate)
