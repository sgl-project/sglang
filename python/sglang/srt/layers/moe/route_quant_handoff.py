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
    NVFP4 trtllm runner             stage/take with the w13 global scale
                                    select the NVFP4 quantizer instead

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

from sglang.kernels.ops.moe import moe_route_quant_fused
from sglang.kernels.ops.moe.trtllm_prepared_fp4 import prepared_routing_sm_supported


class _Handoff(msgspec.Struct):
    # staged by the model layer: the activation rows the runner will quantize
    request_x: Optional[torch.Tensor] = None
    # NVFP4 global scale; None stages the MXFP8 quantizer
    request_scale: Optional[torch.Tensor] = None
    # published by the routing dispatch, keyed by the staged activations
    produced_x: Optional[torch.Tensor] = None
    produced_scale: Optional[torch.Tensor] = None
    packed: Optional[torch.Tensor] = None
    x_q: Optional[torch.Tensor] = None
    x_s: Optional[torch.Tensor] = None
    metadata: Optional[list[torch.Tensor]] = None
    metadata_routing: tuple = ()


_handoff = _Handoff()


def stage(x: torch.Tensor, global_scale: Optional[torch.Tensor] = None) -> None:
    """Publish the routed activations for the upcoming topk call. Caller pairs
    this with clear() after the experts call (try/finally)."""
    clear()
    _handoff.request_x = x
    _handoff.request_scale = global_scale


def clear() -> None:
    global _handoff
    _handoff = _Handoff()


def _same(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    return a is b or (
        a is not None
        and b is not None
        and a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.dtype == b.dtype
        and a.stride() == b.stride()
    )


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
    x, scale = _handoff.request_x, _handoff.request_scale
    if x is None or num_fused_shared_experts != 0:
        return None

    if scale is not None:
        if not moe_route_quant_fused.nvfp4_covered(
            gating_output, correction_bias, topk, x, scale
        ):
            return None
        metadata = (
            moe_route_quant_fused.allocate_metadata(x.device, x.shape[0], torch.float32)
            if x.shape[0] <= 8
            else None
        )
        weights, ids, x_q, x_s = moe_route_quant_fused.route_quant_nvfp4(
            gating_output,
            correction_bias,
            x,
            scale,
            topk,
            renormalize,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            apply_routed_scaling_factor_on_output,
            metadata,
        )
        return _publish(
            x, scale, weights, ids, None, x_q, x_s, metadata, (ids, weights)
        )

    if (
        not moe_route_quant_fused.covered(gating_output, correction_bias, topk, x)
        or not moe_route_quant_fused.available()
    ):
        return None

    metadata = None
    if (
        x.shape[0] <= 8
        and prepared_routing_sm_supported()
        and moe_route_quant_fused.metadata_available()
    ):
        metadata = moe_route_quant_fused.allocate_metadata(x.device, x.shape[0])
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
        metadata=metadata,
    )
    return _publish(x, None, weights, ids, packed, x_q, x_s, metadata, (packed,))


def _publish(x, scale, weights, ids, packed, x_q, x_s, metadata, routing):
    _handoff.request_x = None
    _handoff.produced_x = x
    _handoff.produced_scale = scale
    _handoff.metadata = metadata
    _handoff.metadata_routing = routing
    _handoff.packed = packed
    _handoff.x_q = x_q
    _handoff.x_s = x_s
    return weights, ids


def ready(x: torch.Tensor, global_scale: Optional[torch.Tensor] = None) -> bool:
    return (
        _handoff.produced_x is not None
        and _same(_handoff.produced_x, x)
        and _same(_handoff.produced_scale, global_scale)
    )


def take(
    x: torch.Tensor, global_scale: Optional[torch.Tensor] = None
) -> Optional[Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]]:
    """Consume the published (packed_topk, x_q, x_s) for these exact
    activation rows and quantizer, or None. MXFP8 gives int32 UE8M0 scales;
    NVFP4 gives no packed ids and fp8 scales. Storage identity is verified so
    a re-viewed or copied tensor simply misses."""
    if not ready(x, global_scale):
        return None
    out = (_handoff.packed, _handoff.x_q, _handoff.x_s)
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None
    return out


def take_metadata(routing):
    """Consume metadata only for the exact routing tensors that made it: the
    packed ids, or the (ids, weights) pair."""
    routing = routing if isinstance(routing, tuple) else (routing,)
    made = _handoff.metadata_routing
    hit = len(made) == len(routing) and all(a is b for a, b in zip(made, routing))
    metadata = _handoff.metadata if hit else None
    _handoff.metadata = None
    _handoff.metadata_routing = ()
    return metadata
