"""Attempt-and-verify handoff for the fused K3 MoE-front prep launch.

At decode batch sizes the chain between the K3 fused-front GEMM and the
trtllm-gen SiTU MoE op is three tiny back-to-back kernels on the critical
path — route_radix (top-16 of 896), the triton ``(id << 16) | bf16(weight)``
pack, and per_token_group_quant (mxfp8, ``[T, 3584]``) — about 7.5 us busy
plus two extra launches per MoE layer, each leaving the SMs near idle. The
fused kernel (kernels/ops/moe/moe_route_quant_fused.py) runs all three in one
launch: routing CTAs and one quant CTA per token, concurrently.

The inputs live in different modules (router logits reach the router through
TopK, activations through the MoE runner), so the fusion is wired as a
consume-once stash instead of new signatures:

    KimiK3MoE._forward_routed*      stage(x) before self.topk, clear() after
                                    the experts call
    biased_grouped_topk_gpu         try_route_quant_fused() replaces the
                                    moe_fused_gate call on a hit
    Mxfp4MoEMethod.apply (situ)     take(x) skips the quant and the pack

Every step is fallback-safe: if the routing dispatch never consumes the staged
request (different model, uncovered shape, triton fallback) or the runner's
``take`` misses (activations re-viewed or copied), the unfused chain runs as
before. The stage/clear bracket in the model layer guarantees a published
entry can never leak into another layer whose allocator reused the same
activation address.
"""

from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch


class _Handoff(msgspec.Struct):
    # staged by the model layer: the activation rows the runner will quantize
    request_x: Optional[torch.Tensor] = None
    # published by the routing dispatch, keyed by the staged activations
    produced_x: Optional[torch.Tensor] = None
    packed: Optional[torch.Tensor] = None
    x_q: Optional[torch.Tensor] = None
    x_s: Optional[torch.Tensor] = None


_handoff = _Handoff()


def stage(x: torch.Tensor) -> None:
    """Publish the routed activations for the upcoming topk call. Caller pairs
    this with clear() after the experts call (try/finally)."""
    _handoff.request_x = x
    _handoff.produced_x = None


def clear() -> None:
    _handoff.request_x = None
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None


def try_route_quant_fused(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused replacement for the ungrouped-sigmoid moe_fused_gate call when a
    staged request covers it. Returns (weights, ids) on a hit, None otherwise
    (caller falls through to the unfused router)."""
    x = _handoff.request_x
    if x is None or num_fused_shared_experts != 0:
        return None

    from sglang.kernels.ops.moe import moe_route_quant_fused

    if (
        not moe_route_quant_fused.covered(gating_output, correction_bias, topk, x)
        or not moe_route_quant_fused.available()
    ):
        return None

    weights, ids, packed, x_q, x_s = moe_route_quant_fused.route_quant_fused(
        gating_output,
        correction_bias,
        x,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=(
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        ),
        apply_scale=apply_routed_scaling_factor_on_output,
    )
    _handoff.request_x = None
    _handoff.produced_x = x
    _handoff.packed = packed
    _handoff.x_q = x_q
    _handoff.x_s = x_s
    return weights, ids


def take(
    x: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Consume the published (packed_topk, x_q, x_s int32) for these exact
    activation rows, or None. Storage identity is verified so a re-viewed or
    copied tensor simply misses."""
    produced = _handoff.produced_x
    if produced is None:
        return None
    if (
        produced.data_ptr() != x.data_ptr()
        or produced.shape != x.shape
        or produced.dtype != x.dtype
        or produced.stride() != x.stride()
    ):
        return None
    out = (_handoff.packed, _handoff.x_q, _handoff.x_s)
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None
    return out
