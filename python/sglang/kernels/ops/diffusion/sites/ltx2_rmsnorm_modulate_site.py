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
marks its adaLN sites with ``LTX2_RMS_NORM_MODULATE.mark`` (default off,
reference path) and the denoising stage mounts / unmounts
``LTX2_RMS_NORM_MODULATE`` at batch boundaries for ``quality="high"`` requests.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_bitexact import (
    fused_rmsnorm_scale_shift_bitexact,
)
from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

LTX2_RMS_NORM_MODULATE = QualityGatedFusion(
    name="LTX-2 RMSNorm+modulate",
    marker_attr="_sgl_ltx2_rms_norm_modulate_site",
    enabled_attr="_sgl_ltx2_rms_norm_modulate_enabled",
)

# ``RMSNormNoWeight`` applies no scale, so a ones weight reproduces it exactly.
_ONES_WEIGHT_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _ones_weight(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, int(x.shape[-1]))
    w = _ONES_WEIGHT_CACHE.get(key)
    if w is None:
        w = torch.ones(x.shape[-1], device=x.device, dtype=torch.bfloat16)
        _ONES_WEIGHT_CACHE[key] = w
    return w


def fused_ltx2_rms_norm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    """``rms_norm(x) * (1 + scale) + shift`` as one kernel (weightless RMSNorm)."""
    return fused_rmsnorm_scale_shift_bitexact(x, _ones_weight(x), scale, shift, eps)
