"""Fail-closed adapter for AITER's gfx950 Kimi-K3 FP8 latent tail."""

from __future__ import annotations

import os

import torch

from sglang.srt.utils import is_hip


def enabled() -> bool:
    return is_hip() and os.environ.get(
        "SGLANG_K3_AITER_LATENT_TAIL_FP8", "0"
    ).lower() in ("1", "true")


def _ops():
    try:
        from sglang.kernels.ops.kimi_k3.flydsl.source import load_module

        module = load_module(
            "sglang.kernels.ops.kimi_k3.flydsl.latent_moe_tail_fp8",
            "aiter.ops.flydsl.latent_moe_tail_fp8",
        )
    except (ImportError, ModuleNotFoundError):
        return None, None, None
    return (
        module.latent_moe_tail_fp8,
        module.quantize_latent_moe_tail_weight,
        module.supports_latent_moe_tail_fp8,
    )


def pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, quantize, _ = _ops()
    if quantize is None:
        raise RuntimeError("AITER latent-tail quantizer is unavailable")
    return quantize(weight.contiguous())


def covered(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
) -> bool:
    if not enabled():
        return False
    _, _, supports = _ops()
    return bool(
        supports is not None
        and supports(
            routed,
            shared,
            rms_weight,
            up_weight,
            up_scale,
            epsilon,
        )
    )


def run(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    op, _, _ = _ops()
    if op is None:
        raise RuntimeError("AITER latent-tail fusion is unavailable")
    return op(
        routed,
        shared,
        rms_weight,
        up_weight,
        up_scale,
        epsilon,
    )


def warmup(
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
) -> None:
    if not enabled():
        return
    routed = torch.zeros((1, 3584), dtype=torch.bfloat16, device=up_weight.device)
    shared = torch.zeros((1, 7168), dtype=torch.bfloat16, device=up_weight.device)
    if covered(routed, shared, rms_weight, up_weight, up_scale, epsilon):
        run(routed, shared, rms_weight, up_weight, up_scale, epsilon)
        torch.cuda.synchronize(up_weight.device)
