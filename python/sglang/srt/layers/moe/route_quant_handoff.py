# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

K3.1 fused routing (SGLANG_K31_FUSED_ROUTING): when the model layer also
stages its routed-experts layer, the routing dispatch plans that layer's
trtllm-gen MoE call first (its autotuned tactic fixes the tile the routing
metadata is padded to) and runs kernels/ops/moe/moe_k31_fused_routing.py
instead: route, quant, the padded-row mask (select_experts then skips its own)
and trtllm-gen's routing metadata in one launch, for 1-512 rows and any expert
range (EP). ``take`` then hands the runner the plan
(flashinfer_trtllm.K31FusedRoutingPlan) in place of the packed ids, and the
routed MoE runs its GEMMs without its routing kernel. Where it does not apply
the M <= 8 prepared routing of route_quant_fused (sgl-project/sglang #46475 /
#46484) runs as before, and only for a layer that holds all the experts
(SGLANG_K31_PF_PREPARED_ROUTING, which also gates the NVFP4 quantizer). With
either switch set, this dispatch replaces the kimi_k3_fused_routing one below.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs


class _Handoff(msgspec.Struct):
    # staged by the model layer: the activation rows the runner will quantize
    request_x: Optional[torch.Tensor] = None
    # NVFP4 global scale; None stages the MXFP8 quantizer
    request_scale: Optional[torch.Tensor] = None
    # staged by the model layer: the routed-experts layer (FusedMoE) those rows
    # go to, which opts them into the K3.1 fused routing
    request_layer: Optional[Any] = None
    # bound by select_experts: the padded-row bound, and whether its
    # post-processing leaves the router's ids as they are (no EPLB remap, no
    # routing override), which the fused routing's metadata assumes
    num_token_non_padded: Optional[torch.Tensor] = None
    fused_routing_ok: bool = False
    # set by the fused routing: it masked the padded rows itself
    padded_rows_masked: bool = False
    # published by the routing dispatch, keyed by the staged activations; the
    # packed ids, or the K3.1 fused routing's K31FusedRoutingPlan
    produced_x: Optional[torch.Tensor] = None
    produced_scale: Optional[torch.Tensor] = None
    packed: Optional[Any] = None
    prepared_ids: Optional[torch.Tensor] = None
    routing_metadata: Optional[Any] = None
    da_body: Optional[Any] = None
    x_q: Optional[torch.Tensor] = None
    x_s: Optional[torch.Tensor] = None
    metadata: Optional[list[torch.Tensor]] = None
    metadata_routing: tuple = ()


_handoff = _Handoff()


def stage(
    x: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    layer: Optional[Any] = None,
) -> None:
    """Publish the routed activations for the upcoming topk call. Caller pairs
    this with clear() after the experts call (try/finally). ``layer``, the
    routed-experts layer, opts the rows into the K3.1 fused routing."""
    clear()
    _handoff.request_x = x
    _handoff.request_scale = global_scale
    _handoff.request_layer = layer


def clear() -> None:
    global _handoff
    _handoff = _Handoff()


def bind_select_experts(
    num_token_non_padded: Optional[torch.Tensor], fused_routing_ok: bool
) -> None:
    """Called by select_experts before the routing dispatch of a staged request."""
    if _handoff.request_x is None:
        return
    _handoff.num_token_non_padded = num_token_non_padded
    _handoff.fused_routing_ok = fused_routing_ok


def take_padded_rows_masked() -> bool:
    """Whether the routing dispatch just masked the padded rows' ids itself."""
    masked = _handoff.padded_rows_masked
    _handoff.padded_rows_masked = False
    return masked


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
    x = _handoff.request_x
    if x is None or num_fused_shared_experts != 0:
        return None
    if envs.SGLANG_K31_FUSED_ROUTING.get() or envs.SGLANG_K31_PF_PREPARED_ROUTING.get():
        return _try_route_quant_fused_k31(
            gating_output,
            correction_bias,
            topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    from sglang.kernels.ops.moe import kimi_k3_fused_routing, moe_route_quant_fused

    layer = _handoff.request_layer
    if (
        layer is not None
        and _handoff.fused_routing_ok
        and kimi_k3_fused_routing.covered(gating_output, correction_bias, topk, x)
        and kimi_k3_fused_routing.available()
    ):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            select_trtllm_routed_body,
        )

        weights_dtype = kimi_k3_fused_routing.expert_weights_dtype(x.shape[0])
        body = select_trtllm_routed_body(layer, x, topk, weights_dtype)
        if body is not None:
            routed = kimi_k3_fused_routing.fused_routing(
                gating_output,
                correction_bias,
                x,
                topk,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_scale=apply_routed_scaling_factor_on_output,
                num_token_non_padded=_handoff.num_token_non_padded,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                num_local_experts=layer.num_local_experts,
                tile_tokens=body.tile_n,
                weights_dtype=weights_dtype,
            )
            _handoff.padded_rows_masked = True
            _handoff.prepared_ids = routed.ids
            _handoff.routing_metadata = routed.metadata
            _handoff.da_body = body
            return _publish(
                x,
                None,
                routed.weights,
                routed.ids,
                routed.ids,
                routed.x_q,
                routed.x_s,
                None,
                (),
            )

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
        routed_scaling_factor=routed_scaling_factor,
        apply_scale=apply_routed_scaling_factor_on_output,
    )
    return _publish(x, None, weights, ids, packed, x_q, x_s, None, ())


def _try_route_quant_fused_k31(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The K3.1 dispatch (SGLANG_K31_FUSED_ROUTING / SGLANG_K31_PF_PREPARED_ROUTING):
    the NVFP4 quantizer, the K3.1 fused routing, then route_quant_fused with the
    M <= 8 prepared routing metadata."""
    from sglang.kernels.ops.moe import moe_route_quant_fused
    from sglang.kernels.ops.moe.trtllm_prepared_fp4 import (
        prepared_routing_sm_supported,
    )

    x, scale = _handoff.request_x, _handoff.request_scale
    prepared_routing = envs.SGLANG_K31_PF_PREPARED_ROUTING.get()

    if scale is not None and prepared_routing:
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

    layer = _handoff.request_layer
    if layer is not None and _handoff.fused_routing_ok:
        fused = _try_k31_fused_routing(
            x,
            gating_output,
            correction_bias,
            topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )
        if fused is not None:
            return fused

    if (
        not moe_route_quant_fused.covered(gating_output, correction_bias, topk, x)
        or not moe_route_quant_fused.available()
    ):
        return None

    metadata = None
    # The prepared body takes this metadata only for a layer that holds all the
    # experts (try_prepared_k3_mxfp4); on an expert range it would go unused.
    if (
        prepared_routing
        and x.shape[0] <= 8
        and (layer is None or layer.num_local_experts == layer.num_experts)
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


def _try_k31_fused_routing(
    x: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not envs.SGLANG_K31_FUSED_ROUTING.get():
        return None

    from sglang.kernels.ops.moe import moe_k31_fused_routing

    if (
        x.shape[0] < moe_k31_fused_routing.SERVING_MIN_TOKENS
        or not moe_k31_fused_routing.covered(gating_output, correction_bias, topk, x)
        or not moe_k31_fused_routing.available()
    ):
        return None

    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        plan_k31_fused_routing,
    )

    plan = plan_k31_fused_routing(_handoff.request_layer, x, topk)
    if plan is None:
        return None
    routed = moe_k31_fused_routing.k31_fused_routing(
        gating_output,
        correction_bias,
        x,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=(
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        ),
        apply_scale=apply_routed_scaling_factor_on_output,
        num_token_non_padded=_handoff.num_token_non_padded,
        local_expert_offset=plan.local_expert_offset,
        num_local_experts=plan.local_num_experts,
        tile_tokens=plan.tile_tokens,
        weights_dtype=plan.weights_dtype,
    )
    plan.metadata = routed.metadata
    plan.topk_ids = routed.ids
    _handoff.padded_rows_masked = True
    return _publish(
        x, None, routed.weights, routed.ids, plan, routed.x_q, routed.x_s, None, ()
    )


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


def take_prepared_routing(ids: torch.Tensor) -> Optional[Tuple[Any, Any]]:
    if _handoff.prepared_ids is not ids:
        return None
    out = (_handoff.routing_metadata, _handoff.da_body)
    _handoff.prepared_ids = None
    _handoff.routing_metadata = None
    _handoff.da_body = None
    return out


def ready(x: torch.Tensor, global_scale: Optional[torch.Tensor] = None) -> bool:
    return (
        _handoff.produced_x is not None
        and _same(_handoff.produced_x, x)
        and _same(_handoff.produced_scale, global_scale)
    )


def take(
    x: torch.Tensor, global_scale: Optional[torch.Tensor] = None
) -> Optional[Tuple[Optional[Any], torch.Tensor, torch.Tensor]]:
    """Consume the published (packed_topk or K31FusedRoutingPlan, x_q, x_s)
    for these exact activation rows and quantizer, or None. MXFP8 gives int32
    UE8M0 scales; NVFP4 gives no packed ids and fp8 scales. Storage identity is
    verified so a re-viewed or copied tensor simply misses."""
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
