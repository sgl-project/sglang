"""Quality gate for the LTX-2 Q/K RMSNorm + split-RoPE Hopper path.

The fused CUDA kernel is already the default on SM100+, but its fused
rounding differs from the Hopper eager chain.  LTX-2 attention sites therefore
enable the SM90 path only for requests whose quality policy allows approximate
kernel fusions.
"""

from __future__ import annotations

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

LTX2_QKNORM_SPLIT_ROPE = QualityGatedFusion(
    name="LTX-2 Hopper QKNorm+split-RoPE",
    marker_attr="_sgl_ltx2_qknorm_split_rope_site",
    enabled_attr="_sgl_ltx2_qknorm_split_rope_enabled",
)
