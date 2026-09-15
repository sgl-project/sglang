"""Weightless RMSNorm + adaLN modulate folded into one kernel for LTX-2.

``rms_norm(x) * (1 + scale) + shift`` at the LTX-2 transformer-block adaLN
sites is otherwise an aten ``F.rms_norm`` plus a separate ``mul``/``add``
modulate (one reduction kernel plus several pointwise passes per site). This
folds the whole chain into a single ``fused_rmsnorm_scale_shift_bitexact``
launch.

The fused kernel reproduces the RMSNorm math via ``rsqrt.approx`` (the
flashinfer CuTe form), which differs from aten's refined ``rsqrtf`` by at
most one bf16 ULP on a small fraction of elements. It is therefore *not*
bit-exact vs the eager reference, so the fold is opt-in per batch: model code
marks its adaLN sites with :func:`mark_ltx2_rms_norm_modulate_site` (default
off, reference path) and the denoising stage calls
:func:`mount_ltx2_rms_norm_modulate` / :func:`unmount_ltx2_rms_norm_modulate`
at batch boundaries for ``quality="high"`` requests.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_bitexact import (
    can_use_fused_rmsnorm_scale_shift,
    fused_rmsnorm_scale_shift_bitexact,
)
from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

_SITE_MARKER_ATTR = "_sgl_ltx2_rms_norm_modulate_site"
_SITE_ENABLED_ATTR = "_sgl_ltx2_rms_norm_modulate_enabled"
_FUSION = QualityGatedFusion(
    name="LTX-2 RMSNorm+modulate",
    marker_attr=_SITE_MARKER_ATTR,
    enabled_attr=_SITE_ENABLED_ATTR,
)

# ``RMSNormNoWeight`` applies no scale, so a ones weight reproduces it exactly.
_ONES_WEIGHT_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def mark_ltx2_rms_norm_modulate_site(module: nn.Module) -> None:
    """Mark ``module`` as an LTX-2 RMSNorm+modulate fusion site (mounted off)."""
    _FUSION.mark(module)


def ltx2_rms_norm_modulate_active(module: nn.Module) -> bool:
    return _FUSION.is_enabled(module)


def mount_ltx2_rms_norm_modulate(root: nn.Module) -> bool:
    return _FUSION.mount(root)


def unmount_ltx2_rms_norm_modulate(root: nn.Module) -> None:
    _FUSION.unmount(root)


def _ones_weight(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, int(x.shape[-1]))
    w = _ONES_WEIGHT_CACHE.get(key)
    if w is None:
        w = torch.ones(x.shape[-1], device=x.device, dtype=torch.bfloat16)
        _ONES_WEIGHT_CACHE[key] = w
    return w


def can_use_ltx2_rms_norm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:
    if x.dtype is not torch.bfloat16 or not x.is_cuda:
        return False
    return can_use_fused_rmsnorm_scale_shift(x, _ones_weight(x), scale, shift)


def fused_ltx2_rms_norm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    """``rms_norm(x) * (1 + scale) + shift`` as one kernel (weightless RMSNorm)."""
    return fused_rmsnorm_scale_shift_bitexact(x, _ones_weight(x), scale, shift, eps)
