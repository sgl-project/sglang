"""Request-scoped Wan NVFP4 bias+GELU fusion.

The fused JIT kernel is bit-exact with the local eager ``add + GELU`` chain for
eligible ModelOpt FP4 linears. It still changes the model's kernel schedule, so
keep the default ``quality="lossless"`` path unchanged and mount this fast path
only for the existing ``quality="high"`` contract.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

logger = logging.getLogger(__name__)

_FUSION = QualityGatedFusion(
    name="Wan NVFP4 bias+GELU",
    marker_attr="_sgl_nvfp4_bias_gelu_site",
    enabled_attr="_sgl_nvfp4_bias_gelu_enabled",
)


def mark_nvfp4_bias_gelu_site(module: nn.Module) -> None:
    """Mark an MLP whose ``fc_in`` bias can be deferred for this fusion."""
    _FUSION.mark(module)


def nvfp4_bias_gelu_active(module: nn.Module) -> bool:
    """Whether the quality-gated fusion is mounted on ``module``."""
    return _FUSION.is_enabled(module)


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


def mount_nvfp4_bias_gelu(root: nn.Module) -> bool:
    """Enable every eligible marked site under ``root``."""
    sites = list(_FUSION.iter_sites(root))
    mounted = _FUSION.mount(root, reject_reason=_site_reject_reason, logger=logger)
    for site in sites:
        site.fc_in.skip_bias_add = mounted
    return mounted


def unmount_nvfp4_bias_gelu(root: nn.Module) -> None:
    """Restore every marked site to its original linear+GELU path."""
    _FUSION.unmount(root)
    for site in _FUSION.iter_sites(root):
        site.fc_in.skip_bias_add = False
