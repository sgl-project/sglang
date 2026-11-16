"""Qwen-Image added-QKV GEMM packing, gated by request quality.

Packing the three BF16 text projections into one GEMM changes the reduction
association and is therefore not bit-exact.  The packed weights stay resident
for checkpoint compatibility, but ``quality="exact"`` applies their three
slices independently.  ``quality="high"`` mounts the single-GEMM path.
"""

from __future__ import annotations

import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion


def _site_reject_reason(site: nn.Module) -> str | None:
    linear = getattr(site, "to_added_qkv", None)
    if linear is None:
        return "missing to_added_qkv"
    if getattr(linear, "quant_config", None) is not None:
        return "quantized packed projection"
    if len(getattr(linear, "output_partition_sizes", ())) != 3:
        return "packed projection does not contain three shards"
    return None


QWEN_IMAGE_ADDED_QKV = QualityGatedFusion(
    name="Qwen-Image fused added-QKV",
    marker_attr="_sgl_qwen_image_added_qkv_site",
    enabled_attr="_sgl_qwen_image_added_qkv_enabled",
    reject_reason=_site_reject_reason,
)
