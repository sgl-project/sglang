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
``quality="high"`` requests via :func:`mount_fused_gate_rmsnorm` /
:func:`unmount_fused_gate_rmsnorm` at batch boundaries; the default
``"lossless"`` path keeps the unmodified reference chain bit-for-bit.

Mounting is all-or-nothing per transformer: if any marked site fails the
static guards (non-bf16 norm weight, hidden size above the kernel limit, ...)
every site on that transformer stays on the reference path.
"""

from __future__ import annotations

import logging
from importlib import import_module

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

logger = logging.getLogger(__name__)

# Attributes of the site protocol (set by ``mark_fused_gate_rmsnorm_site``).
_SITE_NORM_ATTRS = "_sgl_fused_gate_rmsnorm_norm_attrs"
_SITE_ENABLED_ATTR = "_sgl_fused_gate_rmsnorm_enabled"
_FUSION = QualityGatedFusion(
    name="fused gate RMSNorm",
    marker_attr=_SITE_NORM_ATTRS,
    enabled_attr=_SITE_ENABLED_ATTR,
)

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
    for attr in _FUSION.metadata(site, ()):
        norm = getattr(site, attr, None)
        weight = getattr(norm, "weight", None)
        if weight is None or weight.dim() != 1:
            return f"{attr}: missing or non-1D norm weight"
        if weight.dtype != torch.bfloat16:
            return f"{attr}: non-bf16 norm weight dtype {weight.dtype}"
        if weight.numel() > _MAX_HIDDEN_SIZE:
            return f"{attr}: hidden size {weight.numel()} above kernel limit"
    return None


def mark_fused_gate_rmsnorm_site(module: nn.Module, norm_attrs: tuple[str, ...]):
    """Declare ``module`` as a fused RMSNorm modulate/gate site.

    ``norm_attrs`` names the site's RMSNorm submodules (checked by the static
    guards at mount time). The site starts unmounted
    (``_sgl_fused_gate_rmsnorm_enabled = False``): the module's forward must
    keep the reference path bit-exact until :func:`mount_fused_gate_rmsnorm`
    enables it.
    """
    _FUSION.mark(module, tuple(norm_attrs))


def fused_gate_rmsnorm_active(module: nn.Module) -> bool:
    """Whether the quality-gated fused path is mounted on ``module``."""
    return _FUSION.is_enabled(module)


def mount_fused_gate_rmsnorm(root: nn.Module) -> bool:
    """Enable the fused kernels on every marked site under ``root``.

    All-or-nothing: if any marked site fails the static guards, every site is
    left (or reset) on the reference path and False is returned. Returns False
    as well when ``root`` has no marked sites.
    """
    return _FUSION.mount(root, reject_reason=_static_reject_reason, logger=logger)


def unmount_fused_gate_rmsnorm(root: nn.Module) -> None:
    """Reset every marked site under ``root`` to the bit-exact reference path."""
    _FUSION.unmount(root)
