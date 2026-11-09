"""Quality-gated fused RMSNorm modulate/gate sites.

Adaln-style DiT blocks (Ideogram 4) spend four elementwise chains per block on
modulate/gate around each RMSNorm: ``RMSNorm(x) * scale`` before
attention/FFN and ``x + tanh(gate) * RMSNorm(out)`` after. Shared BF16-native
Triton kernels
(:mod:`sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton`) fuse each
chain into a single kernel (RMSNorm + tanh + mul + add in one pass).

Z-Image mounts those kernels unconditionally because they reproduce its own
native-bf16 reference RMSNorm. Ideogram's reference norm is ``F.rms_norm``
(fp32 internal statistics), so the fused path is numerically close (the norm
statistics round through bf16) but **not bit-exact**. Following the
fused-linear-GELU precedent, sites are therefore mounted only for
``quality="high"`` requests at batch boundaries; the default ``"exact"`` path
keeps the unmodified reference chain bit-for-bit.

Mounting is all-or-nothing per transformer: if any marked site fails the
static guards (non-bf16 norm weight, hidden size above the kernel limit, ...)
every site on that transformer stays on the reference path.
"""

from __future__ import annotations

from importlib import import_module

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

# The Triton kernels mask a single block over the hidden dim.
_MAX_HIDDEN_SIZE = 8192


def fused_rmsnorm_scale(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, eps: float
) -> torch.Tensor | None:
    """``RMSNorm(x, weight, eps) * scale`` in one Triton kernel (or None)."""
    from sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton import (
        rmsnorm_scale,
    )

    return rmsnorm_scale(x, weight, scale, eps)


def fused_rmsnorm_tanh_residual(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """``residual + tanh(gate) * RMSNorm(x, weight, eps)`` fused (or None)."""
    from sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton import (
        rmsnorm_tanh_residual,
    )

    return rmsnorm_tanh_residual(x, gate, residual, weight, eps)


def _static_reject_reason(site: nn.Module) -> str | None:
    """Why ``site`` may never use the fused kernels, or None if it may."""
    try:
        import_module("triton")
    except ImportError:
        return "triton unavailable"
    for attr in FUSED_GATE_RMSNORM.metadata(site, ()):
        norm = getattr(site, attr, None)
        weight = getattr(norm, "weight", None)
        if weight is None or weight.dim() != 1:
            return f"{attr}: missing or non-1D norm weight"
        if weight.dtype != torch.bfloat16:
            return f"{attr}: non-bf16 norm weight dtype {weight.dtype}"
        if weight.numel() > _MAX_HIDDEN_SIZE:
            return f"{attr}: hidden size {weight.numel()} above kernel limit"
    return None


# A site's marker metadata is the tuple of its RMSNorm submodule names.
FUSED_GATE_RMSNORM = QualityGatedFusion(
    name="fused gate RMSNorm",
    marker_attr="_sgl_fused_gate_rmsnorm_norm_attrs",
    enabled_attr="_sgl_fused_gate_rmsnorm_enabled",
    reject_reason=_static_reject_reason,
)
