"""Fail-closed adapter for AITER's gfx950 Kimi-K3 FP8 MoE front."""

from __future__ import annotations

import os

import torch

from sglang.srt.utils import is_hip


def enabled() -> bool:
    return is_hip() and os.environ.get(
        "SGLANG_K3_AITER_MOE_PREROUTE_FP8", "0"
    ).lower() in ("1", "true")


def cooperative_preactivated_enabled() -> bool:
    return (
        enabled()
        and os.environ.get("SGLANG_K3_FLYDSL_SOURCE", "auto").lower()
        != "aiter"
        and os.environ.get(
            "SGLANG_K3_PREROUTE_PREACTIVATED_SHARED", "0"
        ).lower()
        in ("1", "true")
    )


def _ops():
    try:
        from sglang.kernels.ops.kimi_k3.flydsl.source import load_module

        module = load_module(
            "sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_moe_preroute_fp8",
            "aiter.ops.flydsl.kimi_k3_moe_preroute_fp8",
        )
    except (ImportError, ModuleNotFoundError):
        return None, None, None, None
    return (
        module.kimi_k3_moe_tri_projection_fp8,
        module.kimi_k3_shared_down_fp8,
        module.supports_kimi_k3_moe_tri_projection_fp8,
        module.supports_kimi_k3_shared_down_fp8,
    )


def _preactivated_ops():
    try:
        from sglang.kernels.ops.kimi_k3.flydsl.source import load_module

        module = load_module(
            "sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_moe_preroute_fp8",
            "aiter.ops.flydsl.kimi_k3_moe_preroute_fp8",
        )
    except (ImportError, ModuleNotFoundError):
        return None, None
    return (
        getattr(
            module,
            "kimi_k3_moe_tri_projection_cooperative_preactivated_fp8",
            None,
        ),
        getattr(
            module,
            "supports_kimi_k3_moe_tri_projection_cooperative_preactivated_fp8",
            None,
        ),
    )


def cooperative_preactivated_tri_covered(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    if not cooperative_preactivated_enabled() or hidden.shape[0] not in (2, 4):
        return False
    _, supports = _preactivated_ops()
    return bool(
        supports is not None
        and supports(
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
    op, _ = _preactivated_ops()
    if op is None:
        raise RuntimeError(
            "Kimi-K3 cooperative preactivated projection is unavailable"
        )
    return op(
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
    if not enabled():
        return False
    num_tokens = hidden.shape[0]
    if num_tokens != 1:
        return False
    _, _, supports, _ = _ops()
    return bool(
        supports is not None
        and supports(
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
    op, _, _, _ = _ops()
    if op is None:
        raise RuntimeError("AITER Kimi-K3 pre-route projection is unavailable")
    return op(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_weight,
    )


def shared_down_covered(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    if not enabled():
        return False
    if gate_up.shape[0] != 1:
        return False
    _, _, _, supports = _ops()
    return bool(supports is not None and supports(gate_up, weight, scale))


def run_shared_down(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float,
    out: torch.Tensor,
) -> torch.Tensor:
    _, op, _, _ = _ops()
    if op is None:
        raise RuntimeError("AITER Kimi-K3 shared-down is unavailable")
    return op(
        gate_up,
        weight,
        scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        out=out,
    )


def warmup(
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
    router_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    shared_interleaved_weight: torch.Tensor | None = None,
    shared_interleaved_scale: torch.Tensor | None = None,
    *,
    situ_beta: float,
    situ_linear_beta: float,
) -> None:
    if not enabled():
        return
    token_buckets = [1]
    if cooperative_preactivated_enabled():
        token_buckets.append(2)
        token_buckets.append(4)
    for num_tokens in token_buckets:
        hidden = torch.zeros(
            (num_tokens, 7168),
            dtype=torch.bfloat16,
            device=routed_weight.device,
        )
        if (
            num_tokens in (2, 4)
            and shared_interleaved_weight is not None
            and shared_interleaved_scale is not None
            and cooperative_preactivated_tri_covered(
                hidden,
                routed_weight,
                routed_scale,
                shared_interleaved_weight,
                shared_interleaved_scale,
                router_weight,
            )
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
            continue
        if not tri_covered(
            hidden,
            routed_weight,
            routed_scale,
            shared_weight,
            shared_scale,
            router_weight,
        ):
            continue
        _, gate_up, _ = run_tri(
            hidden,
            routed_weight,
            routed_scale,
            shared_weight,
            shared_scale,
            router_weight,
        )
        out = hidden.new_empty((num_tokens, 7168))
        if shared_down_covered(gate_up, shared_down_weight, shared_down_scale):
            run_shared_down(
                gate_up,
                shared_down_weight,
                shared_down_scale,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                out=out,
            )
    torch.cuda.synchronize(hidden.device)
