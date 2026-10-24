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

Rank-local packed ids (expert parallelism): a rank's routed MoE only computes
its own experts, but FlashInfer picks its routing-kernel tier from
``num_experts``, and K3's 896 experts take the 896-expert tier, which runs as a
cluster kernel from 5 tokens on. When the model stages the rank's expert range,
the fused launch writes those experts as ``0..count-1`` and any other expert in
top-k slot ``k`` as ``count + k``, and the runner passes
``num_experts=rank_local_num_experts(...)`` with offset 0 (128 at EP8: the
256-expert tier, a single-block kernel up to 16 tokens). The same (token,
local expert, weight) entries reach the GEMMs in the same order, so the output
is unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch


class _Handoff(msgspec.Struct):
    # staged by the model layer: the activation rows the runner will quantize
    request_x: Optional[torch.Tensor] = None
    # staged with them: this rank's (first expert, count) under EP, or None
    request_local_experts: Optional[Tuple[int, int]] = None
    # published by the routing dispatch, keyed by the staged activations
    produced_x: Optional[torch.Tensor] = None
    packed: Optional[torch.Tensor] = None
    x_q: Optional[torch.Tensor] = None
    x_s: Optional[torch.Tensor] = None
    # the expert range `packed` was made rank-local for, or None (global ids)
    packed_local_experts: Optional[Tuple[int, int]] = None
    # after take(): the packed ids handed to the runner and their range
    taken_packed: Optional[torch.Tensor] = None
    taken_local_experts: Optional[Tuple[int, int]] = None


_handoff = _Handoff()


def rank_local_num_experts(
    local_expert_offset: int, num_local_experts: int, num_experts: int, topk: int
) -> Optional[int]:
    """``num_experts`` for rank-local packed ids (local experts plus one id per
    top-k slot, rounded up to FlashInfer's multiple of 4), or None when the range
    is not a proper sub-range of ``num_experts`` or nothing would shrink."""
    if (
        local_expert_offset < 0
        or num_local_experts <= 0
        or local_expert_offset + num_local_experts > num_experts
    ):
        return None
    compact = -(-(num_local_experts + topk) // 4) * 4
    return compact if compact < num_experts else None


def stage(x: torch.Tensor, local_experts: Optional[Tuple[int, int]] = None) -> None:
    """Publish the routed activations for the upcoming topk call. Caller pairs
    this with clear() after the experts call (try/finally).

    ``local_experts`` is this rank's ``(first expert, count)`` when the experts
    are sharded over EP ranks; the fused launch then writes rank-local packed
    ids (see the module docstring)."""
    _handoff.request_x = x
    _handoff.request_local_experts = local_experts
    _handoff.produced_x = None


def clear() -> None:
    _handoff.request_x = None
    _handoff.request_local_experts = None
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None
    _handoff.packed_local_experts = None
    _handoff.taken_packed = None
    _handoff.taken_local_experts = None


def _rank_local_range(num_experts: int, topk: int) -> Optional[Tuple[int, int]]:
    local = _handoff.request_local_experts
    if local is None or rank_local_num_experts(*local, num_experts, topk) is None:
        return None
    return local


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

    local = _rank_local_range(gating_output.shape[1], topk)
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
        local_expert_offset=local[0] if local is not None else 0,
        num_local_experts=local[1] if local is not None else 0,
    )
    _handoff.request_x = None
    _handoff.request_local_experts = None
    _handoff.produced_x = x
    _handoff.packed = packed
    _handoff.x_q = x_q
    _handoff.x_s = x_s
    _handoff.packed_local_experts = local
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
    _handoff.taken_packed = _handoff.packed
    _handoff.taken_local_experts = _handoff.packed_local_experts
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None
    _handoff.packed_local_experts = None
    return out


def take_local_experts(packed: torch.Tensor) -> Optional[Tuple[int, int]]:
    """The ``(first expert, count)`` the packed ids from take() were made
    rank-local for, or None when they carry global ids. Keyed by the exact
    tensor take() returned and consumed once."""
    local = _handoff.taken_local_experts if packed is _handoff.taken_packed else None
    _handoff.taken_packed = None
    _handoff.taken_local_experts = None
    return local
