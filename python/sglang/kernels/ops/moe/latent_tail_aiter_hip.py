"""Fail-closed adapter for AITER's gfx950 Kimi-K3 FP8 latent tail."""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip


def enabled() -> bool:
    return is_hip() and envs.SGLANG_ROCM_K3_AITER_LATENT_TAIL_FP8.get()


def _ops():
    try:
        from sglang.kernels.ops.attention.kda_flydsl.source import load_module

        module = load_module(
            "sglang.kernels.ops.moe.kimi_k3_flydsl.latent_moe_tail_fp8",
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
    prefix: torch.Tensor | None = None,
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
            prefix,
        )
    )


def run(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
    prefix: torch.Tensor | None = None,
    skip_rms: bool = False,
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
        prefix=prefix,
        skip_rms=skip_rms,
    )


def warmup(
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    epsilon: float,
) -> None:
    if not enabled():
        return
    for num_tokens in (1, 2, 4):
        routed = torch.zeros(
            (num_tokens, 3584), dtype=torch.bfloat16, device=up_weight.device
        )
        shared = torch.zeros(
            (num_tokens, 7168), dtype=torch.bfloat16, device=up_weight.device
        )
        if covered(routed, shared, rms_weight, up_weight, up_scale, epsilon):
            run(routed, shared, rms_weight, up_weight, up_scale, epsilon)
            run(
                routed,
                shared,
                rms_weight,
                up_weight,
                up_scale,
                epsilon,
                prefix=torch.zeros_like(shared),
            )
            run(
                routed,
                shared,
                rms_weight,
                up_weight,
                up_scale,
                epsilon,
                skip_rms=True,
            )
            run(
                routed,
                shared,
                rms_weight,
                up_weight,
                up_scale,
                epsilon,
                prefix=torch.zeros_like(shared),
                skip_rms=True,
            )
        torch.cuda.synchronize(up_weight.device)
