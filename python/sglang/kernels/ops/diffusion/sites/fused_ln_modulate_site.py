"""LayerNorm + adaLN modulate folded into one affine LN call.

``layer_norm(x, weight=(1 + scale), bias=shift)`` replaces the affine-free
LayerNorm + modulate pair: one kernel and one HBM pass per site instead of
two.  ``1 + scale`` keeps the eager rounding of the [1, D] modulation row,
but scale/shift then apply in fp32 to the *unrounded* normalized value, so
the result is not bit-exact vs the reference (half-precision rounding-order
differences only).

Because it is not bit-exact the fold is opt-in per batch: model code marks
its LN+modulate sites with ``FUSED_LN_MODULATE.mark`` (default off,
reference path), and the denoising stage mounts / unmounts
``FUSED_LN_MODULATE`` at batch boundaries for ``quality="high"`` requests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

FUSED_LN_MODULATE = QualityGatedFusion(
    name="fused LN+modulate",
    marker_attr="_sgl_fused_ln_modulate_site",
    enabled_attr="_sgl_fused_ln_modulate_enabled",
)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def can_use_ln_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:
    """Per-call guard: the folded affine is a [D] row, so batch must be 1."""
    return (
        x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and scale.dtype == x.dtype
        and shift.dtype == x.dtype
        and x.dim() == 3
        and x.shape[0] == 1
        and scale.dim() == 2
        and scale.shape == shift.shape
        and scale.shape == (1, x.shape[-1])
        and x.numel() > 0
    )


def fused_ln_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    """``layer_norm(x) * (1 + scale) + shift`` as one affine-folded LN kernel."""
    return F.layer_norm(
        x,
        (x.shape[-1],),
        weight=(1 + scale).reshape(-1),
        bias=shift.reshape(-1),
        eps=eps,
    )
