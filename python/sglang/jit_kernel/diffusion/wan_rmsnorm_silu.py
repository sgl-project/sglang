from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

_WAN_RMSNORM_SILU_DISABLED = False


def _bias_tensor(norm: nn.Module) -> torch.Tensor | None:
    bias = getattr(norm, "bias", None)
    return bias if isinstance(bias, torch.Tensor) else None


def _can_apply_wan_rmsnorm_silu(
    x: torch.Tensor,
    norm: nn.Module,
    activation: nn.Module,
) -> bool:
    if not (
        isinstance(activation, nn.SiLU)
        and not activation.inplace
        and getattr(norm, "channel_first", False)
        and hasattr(norm, "gamma")
        and hasattr(norm, "scale")
    ):
        return False

    gamma = getattr(norm, "gamma")
    if not isinstance(gamma, torch.Tensor):
        return False

    bias = _bias_tensor(norm)
    from sglang.jit_kernel.diffusion.triton.wan_rmsnorm_silu import (
        can_use_triton_wan_rmsnorm_silu,
    )

    return can_use_triton_wan_rmsnorm_silu(x, gamma, bias)


def apply_wan_rmsnorm_silu(
    x: torch.Tensor,
    norm: nn.Module,
    activation: nn.Module,
) -> torch.Tensor:
    global _WAN_RMSNORM_SILU_DISABLED

    if not _WAN_RMSNORM_SILU_DISABLED and _can_apply_wan_rmsnorm_silu(
        x, norm, activation
    ):
        try:
            from sglang.jit_kernel.diffusion.triton.wan_rmsnorm_silu import (
                triton_wan_rmsnorm_silu,
            )

            return triton_wan_rmsnorm_silu(
                x,
                norm.gamma,
                _bias_tensor(norm),
                rms_scale=float(norm.scale),
            )
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            logger.warning("Disabling Wan RMSNorm+SiLU Triton fast path: %s", exc)
            _WAN_RMSNORM_SILU_DISABLED = True

    return activation(norm(x))


__all__ = ["apply_wan_rmsnorm_silu"]
