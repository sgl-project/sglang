"""Fail-closed adapter for the gfx950 Kimi-K3 FP8 latent MoE tail."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Optional

import torch

from sglang.srt.utils import is_hip

# Token counts the tail kernel is compiled for.
TOKEN_BUCKETS = (1, 2, 4)


@functools.cache
def _module() -> Optional[ModuleType]:
    if not is_hip():
        return None
    try:
        from sglang.kernels.ops.attention.kda_flydsl.source import load_module

        return load_module(
            "sglang.kernels.ops.moe.kimi_k3_flydsl.latent_moe_tail_fp8",
            "aiter.ops.flydsl.latent_moe_tail_fp8",
        )
    except ImportError:
        return None


def available() -> bool:
    return _module() is not None


def pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 weight and per-row scale for a BF16 [7168, 3584] up-projection."""
    return _module().quantize_latent_moe_tail_weight(weight.contiguous())


def covered(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
    prefix: Optional[torch.Tensor] = None,
) -> bool:
    return available() and _module().supports_latent_moe_tail_fp8(
        routed, shared, rms_weight, up_weight, up_scale, epsilon, prefix
    )


def run(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
    prefix: Optional[torch.Tensor] = None,
    skip_rms: bool = False,
) -> torch.Tensor:
    """up_proj(rms_norm(routed)) + shared (+ prefix); skip_rms when already normed."""
    return _module().latent_moe_tail_fp8(
        routed,
        shared,
        rms_weight,
        up_weight,
        up_scale,
        epsilon,
        prefix=prefix,
        skip_rms=skip_rms,
    )


def warmup(
    *,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
) -> None:
    """Compile every token bucket and epilogue variant before graph capture."""
    device = up_weight.device
    for num_tokens in TOKEN_BUCKETS:
        routed = torch.zeros((num_tokens, 3584), dtype=torch.bfloat16, device=device)
        shared = torch.zeros((num_tokens, 7168), dtype=torch.bfloat16, device=device)
        if not covered(routed, shared, rms_weight, up_weight, up_scale, epsilon):
            continue
        for prefix in (None, torch.zeros_like(shared)):
            for skip_rms in (False, True):
                run(
                    routed,
                    shared,
                    rms_weight,
                    up_weight,
                    up_scale,
                    epsilon,
                    prefix=prefix,
                    skip_rms=skip_rms,
                )
    torch.cuda.synchronize(device)
