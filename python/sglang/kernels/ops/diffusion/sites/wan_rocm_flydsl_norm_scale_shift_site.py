"""Wan's BF16 FlyDSL normalization is approximate and requires quality=high.

The reference preserves FP32 gate/modulation operands and residuals; FlyDSL
converts them to BF16. Mount all eligible sites at the request batch boundary,
and restore the reference path for exact/lossless requests.
"""

from torch import nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion
from sglang.multimodal_gen.runtime.platforms import current_platform


def _static_reject_reason(site: nn.Module) -> str | None:
    if not current_platform.is_rocm():
        return "requires ROCm"
    # Read the same process-level permission as the layer dispatch. This
    # import runs only at mount time, after layer construction has finished.
    from sglang.multimodal_gen.runtime.layers.layernorm import _use_rocm_flydsl

    if not _use_rocm_flydsl:
        return "disabled by SGLANG_USE_ROCM_FLYDSL"
    # The existing kernels use a fixed epsilon, despite exposing an eps argument.
    if site.eps != 1e-6:
        return "FlyDSL requires epsilon 1e-6"
    try:
        from sglang.kernels.ops.diffusion import FLYDSL_NORM_MIN_ALIGNED_DIM
    except ImportError:
        return "FlyDSL unavailable"
    if WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.metadata(site) % FLYDSL_NORM_MIN_ALIGNED_DIM:
        return "unaligned hidden size"
    return None


WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT = QualityGatedFusion(
    name="Wan ROCm FlyDSL norm + scale/shift",
    marker_attr="_sgl_wan_rocm_flydsl_norm_scale_shift_hidden_size",
    enabled_attr="_sgl_wan_rocm_flydsl_norm_scale_shift_enabled",
    reject_reason=_static_reject_reason,
)
