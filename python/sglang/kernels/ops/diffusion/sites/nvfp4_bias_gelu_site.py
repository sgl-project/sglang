"""Request-scoped Wan NVFP4 bias+GELU fusion.

The fused JIT kernel is bit-exact with the local eager ``add + GELU`` chain for
eligible ModelOpt FP4 linears. It still changes the model's kernel schedule, so
keep the default ``quality="exact"`` path unchanged and mount this fast path
only for the existing ``quality="high"`` contract.
"""

from __future__ import annotations

import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion


def _site_reject_reason(site: nn.Module) -> str | None:
    if not getattr(site, "fuse_bias_gelu_tanh", False):
        return "site is not an NVFP4 fused-GELU target"
    linear = getattr(site, "fc_in", None)
    if linear is None:
        return "missing fc_in"

    from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
        ModelOptFp4LinearMethod,
    )

    if not isinstance(getattr(linear, "quant_method", None), ModelOptFp4LinearMethod):
        return "fc_in is not ModelOpt NVFP4"
    if getattr(linear, "bias", None) is None:
        return "fc_in has no bias"
    return None


class _NVFP4BiasGeluFusion(QualityGatedFusion):
    """A mounted site defers its ``fc_in`` bias add to the fused GELU."""

    __slots__ = ()

    def mount(self, root: nn.Module) -> bool:
        mounted = super().mount(root)
        for site in self.iter_sites(root):
            site.fc_in.skip_bias_add = mounted
        return mounted

    def unmount(self, root: nn.Module) -> None:
        super().unmount(root)
        for site in self.iter_sites(root):
            site.fc_in.skip_bias_add = False


# Marks an MLP whose ``fc_in`` bias can be deferred for this fusion.
NVFP4_BIAS_GELU = _NVFP4BiasGeluFusion(
    name="Wan NVFP4 bias+GELU",
    marker_attr="_sgl_nvfp4_bias_gelu_site",
    enabled_attr="_sgl_nvfp4_bias_gelu_enabled",
    reject_reason=_site_reject_reason,
)
