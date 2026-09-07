"""LingBot Video fused RMSNorm, gated by request quality.

LingBot Video keeps norm weights and statistics in FP32. Its reference module
spells RMSNorm as separate cast, square, mean, rsqrt, multiply, weight, and
output-cast operations. Some checkpoint norm weights remain BF16 after load;
the reference formula still promotes their multiplication through the FP32
hidden states. For ``quality="high"``, existing diffusion Triton RMSNorm
kernels replace that chain. Wide rows with FP32 weights use the one-row
``norm_infer`` kernel; the remaining sites use the tiled one-pass kernel. Their
reduction order is not bit-exact, so the default ``quality="lossless"`` path
remains unchanged.
"""

from __future__ import annotations

import logging
from importlib import import_module

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

logger = logging.getLogger(__name__)

_FUSION = QualityGatedFusion(
    name="LingBot Video fused RMSNorm",
    marker_attr="_sgl_lingbot_video_rmsnorm_site",
    enabled_attr="_sgl_lingbot_video_rmsnorm_enabled",
)


def mark_lingbot_video_rmsnorm_site(module: nn.Module) -> None:
    """Mark a LingBot RMSNorm module; it starts on the reference path."""
    _FUSION.mark(module)


def lingbot_video_rmsnorm_active(module: nn.Module) -> bool:
    return _FUSION.is_enabled(module)


def _site_reject_reason(site: nn.Module) -> str | None:
    try:
        import_module("triton")
    except ImportError:
        return "triton unavailable"
    weight = getattr(site, "weight", None)
    if not torch.is_tensor(weight) or weight.dim() != 1:
        return "missing or non-1D norm weight"
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"unsupported norm weight dtype {weight.dtype}"
    if weight.stride(0) != 1:
        return "non-contiguous norm weight"
    return None


def mount_lingbot_video_rmsnorm(root: nn.Module) -> bool:
    return _FUSION.mount(root, reject_reason=_site_reject_reason, logger=logger)


def unmount_lingbot_video_rmsnorm(root: nn.Module) -> None:
    _FUSION.unmount(root)


def try_lingbot_video_rmsnorm(
    site: nn.Module,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Return the quality-gated RMSNorm result, or ``None`` to fall back."""
    if not (
        _FUSION.is_enabled(site)
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and hidden_states.is_contiguous()
        and hidden_states.shape[-1] == weight.numel()
        and weight.is_cuda
        and weight.device == hidden_states.device
        and weight.dtype in (hidden_states.dtype, torch.float32)
    ):
        return None

    hidden_size = hidden_states.shape[-1]
    if weight.dtype == torch.float32 and hidden_size > 128:
        from sglang.kernels.ops.diffusion.norm.norm_triton import norm_infer

        shape = hidden_states.shape
        return norm_infer(
            hidden_states.view(-1, hidden_size),
            weight,
            bias=None,
            eps=eps,
            is_rms_norm=True,
        ).view(shape)

    from sglang.kernels.ops.diffusion.norm.rmsnorm_onepass_triton import (
        triton_one_pass_rms_norm,
    )

    return triton_one_pass_rms_norm(hidden_states, weight, eps)
