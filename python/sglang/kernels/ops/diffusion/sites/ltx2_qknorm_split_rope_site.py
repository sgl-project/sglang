"""Quality gate for the LTX-2 Q/K RMSNorm + split-RoPE Hopper path.

The fused CUDA kernel is already the default on SM100+, but its fused
rounding differs from the Hopper eager chain.  LTX-2 attention sites therefore
enable the SM90 path only for requests whose quality policy allows approximate
kernel fusions.
"""

from __future__ import annotations

from torch import nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

_FUSION = QualityGatedFusion(
    name="LTX-2 Hopper QKNorm+split-RoPE",
    marker_attr="_sgl_ltx2_qknorm_split_rope_site",
    enabled_attr="_sgl_ltx2_qknorm_split_rope_enabled",
)


def mark_ltx2_qknorm_split_rope_site(module: nn.Module) -> None:
    _FUSION.mark(module)


def ltx2_qknorm_split_rope_active(module: nn.Module) -> bool:
    return _FUSION.is_enabled(module)


def mount_ltx2_qknorm_split_rope(root: nn.Module) -> bool:
    return _FUSION.mount(root)


def unmount_ltx2_qknorm_split_rope(root: nn.Module) -> None:
    _FUSION.unmount(root)
