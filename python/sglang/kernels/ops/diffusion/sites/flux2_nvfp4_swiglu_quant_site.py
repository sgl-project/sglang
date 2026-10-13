"""Request-scoped gate for the FLUX.2 NVFP4 SwiGLU fusion.

The fused FC1 + SwiGLU + FC2-input quantization path changes the rounding
order by quantizing before the reference BF16 intermediate is materialized.
Keep it disabled for the lossless default and mount it only for
``quality="high"`` requests at denoising batch boundaries.
"""

from __future__ import annotations

from torch import nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

_FUSION = QualityGatedFusion(
    name="FLUX.2 NVFP4 FC1+SwiGLU+quant",
    marker_attr="_sgl_flux2_nvfp4_swiglu_quant_site",
    enabled_attr="_sgl_flux2_nvfp4_swiglu_quant_enabled",
)


def mark_flux2_nvfp4_swiglu_quant_site(module: nn.Module) -> None:
    """Mark an eligible FLUX.2 feed-forward site, disabled by default."""
    _FUSION.mark(module)


def flux2_nvfp4_swiglu_quant_active(module: nn.Module) -> bool:
    return _FUSION.is_enabled(module)


def mount_flux2_nvfp4_swiglu_quant(root: nn.Module) -> bool:
    return _FUSION.mount(root)


def unmount_flux2_nvfp4_swiglu_quant(root: nn.Module) -> None:
    _FUSION.unmount(root)
