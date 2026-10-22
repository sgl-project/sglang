"""Fail-closed adapter for the gfx950 Kimi-K3 FP8 MoE pre-route kernels."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Optional

import torch

from sglang.srt.utils import is_hip


@functools.cache
def _module() -> Optional[ModuleType]:
    if not is_hip():
        return None
    try:
        from sglang.kernels.ops.attention.kda_flydsl.source import load_module

        return load_module(
            "sglang.kernels.ops.moe.kimi_k3_flydsl.kimi_k3_moe_preroute_fp8",
            "aiter.ops.flydsl.kimi_k3_moe_preroute_fp8",
        )
    except ImportError:
        return None


def available() -> bool:
    return _module() is not None


def cooperative_preactivated_available() -> bool:
    # Only the SGLang copy ships the cooperative kernel; the AITER module lacks it.
    return hasattr(_module(), "kimi_k3_moe_tri_projection_cooperative_preactivated_fp8")


def cooperative_preactivated_tri_covered(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    return (
        hidden.shape[0] in (2, 4)
        and cooperative_preactivated_available()
        and _module().supports_kimi_k3_moe_tri_projection_cooperative_preactivated_fp8(
            hidden,
            routed_weight,
            routed_scale,
            shared_weight,
            shared_scale,
            router_weight,
        )
    )


def run_tri_cooperative_preactivated(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(routed input, SiTU-activated shared gate-up, router logits)."""
    return _module().kimi_k3_moe_tri_projection_cooperative_preactivated_fp8(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_weight,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        fast_situ=True,
    )


def tri_covered(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    return (
        hidden.shape[0] == 1
        and available()
        and _module().supports_kimi_k3_moe_tri_projection_fp8(
            hidden,
            routed_weight,
            routed_scale,
            shared_weight,
            shared_scale,
            router_weight,
        )
    )


def run_tri(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(routed input, shared gate-up, router logits)."""
    return _module().kimi_k3_moe_tri_projection_fp8(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_weight,
    )


def shared_down_covered(
    gate_up: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> bool:
    return (
        gate_up.shape[0] == 1
        and available()
        and _module().supports_kimi_k3_shared_down_fp8(gate_up, weight, scale)
    )


def run_shared_down(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float,
    out: torch.Tensor,
) -> torch.Tensor:
    return _module().kimi_k3_shared_down_fp8(
        gate_up,
        weight,
        scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        out=out,
    )


def warmup(
    *,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    shared_interleaved_weight: Optional[torch.Tensor],
    shared_interleaved_scale: Optional[torch.Tensor],
    situ_beta: float,
    situ_linear_beta: float,
) -> None:
    """Compile every token bucket the model will launch, before graph capture."""
    device = routed_weight.device
    hidden = torch.zeros((1, 7168), dtype=torch.bfloat16, device=device)
    if tri_covered(
        hidden, routed_weight, routed_scale, shared_weight, shared_scale, router_weight
    ):
        _, gate_up, _ = run_tri(
            hidden,
            routed_weight,
            routed_scale,
            shared_weight,
            shared_scale,
            router_weight,
        )
        if shared_down_covered(gate_up, shared_down_weight, shared_down_scale):
            run_shared_down(
                gate_up,
                shared_down_weight,
                shared_down_scale,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                out=hidden.new_empty((1, 7168)),
            )
    if shared_interleaved_weight is not None:
        for num_tokens in (2, 4):
            hidden = torch.zeros(
                (num_tokens, 7168), dtype=torch.bfloat16, device=device
            )
            if cooperative_preactivated_tri_covered(
                hidden,
                routed_weight,
                routed_scale,
                shared_interleaved_weight,
                shared_interleaved_scale,
                router_weight,
            ):
                run_tri_cooperative_preactivated(
                    hidden,
                    routed_weight,
                    routed_scale,
                    shared_interleaved_weight,
                    shared_interleaved_scale,
                    router_weight,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                )
    torch.cuda.synchronize(device)
