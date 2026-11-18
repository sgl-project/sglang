"""Request-scoped gate for the FLUX.2 NVFP4 SwiGLU fusion.

The fused FC1 + SwiGLU + FC2-input quantization path changes the rounding
order by quantizing before the reference BF16 intermediate is materialized.
Keep it disabled below the approximate tier and mount it only for
``quality="high"`` requests at denoising batch boundaries.
"""

from __future__ import annotations

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

FLUX2_NVFP4_SWIGLU_QUANT = QualityGatedFusion(
    name="FLUX.2 NVFP4 FC1+SwiGLU+quant",
    marker_attr="_sgl_flux2_nvfp4_swiglu_quant_site",
    enabled_attr="_sgl_flux2_nvfp4_swiglu_quant_enabled",
)
