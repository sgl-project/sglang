"""Shared-outer gate/up-A forms: repeated-pair control vs token dedup.

The §41.1(3) keystone: with a shared-outer factor, gate/up-A is K-fold
redundant per token — the same adapter, the same A, the same input row for
every one of a token's K pairs.  Today's runner executes the REPEATED-PAIR
control form (a pair-domain adapter-keyed route, K identical A products per
token).  The proper form computes each (token, adapter) product ONCE.

This module builds the dedup form from EXISTING machinery — no new kernels,
per the §41.3 rule:

* **T-domain adapter plan**: `build_virtual_expert_routing` over a dummy
  single-expert route (``topk_ids = zeros [T, 1]``) with
  ``lora_experts_per_adapter=1`` — the fused key degenerates to the adapter slot
  (§41.1(3a): the ID pass is an identity here), so the aligned plan groups
  TOKENS by adapter.  V = L_cap, always the JIT path, the "tiny adapter
  align" of §47.3.
* **Dedup A**: the stock `grouped_lora_a` over that plan — TOP_K = 1 makes
  its token-major input row exactly the token id, and the output lands
  token-major ``[T, slices*R]``: a K-times smaller bridge than the
  pair-major control's ``[T*K, slices*R]``.
* **B consumes tokens directly**: `stock_grouped_lora_b` with
  ``intermediate_top_k = K`` reads row ``pair // K`` — every pair of a
  token reads the single deduplicated A row.  No broadcast/materialize
  step exists in this form at all (the archive candidate paid one).

Numerically the two forms are BITWISE equal through B: the control's K
copies of a token's A row are computed from identical inputs and weights,
so B consumes identical BF16 values either way — pinned by the registered
test, which makes the dedup decision purely a performance question.
"""

from __future__ import annotations

import torch

from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a, stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_ALIGNED,
    RouteView,
    build_virtual_expert_routing,
)


def masked_token_slots_for_plan(
    token_slots: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_local_experts: int,
) -> torch.Tensor:
    """Slot ``-1`` for tokens with ZERO valid local pairs.

    Fourth S3 review: under sparse-local EP routing some tokens route to no
    local expert at all; the repeated-pair control skips their (invalid)
    pairs, so an unmasked T-plan would schedule A work the control never
    does. Masking is output-invisible — B never reads a no-local token's
    bridge row, because every one of its pairs is a sentinel in the
    per-expert route.
    """
    has_local_pair = ((topk_ids >= 0) & (topk_ids < num_local_experts)).any(dim=1)
    return torch.where(has_local_pair, token_slots, torch.full_like(token_slots, -1))


def build_token_adapter_plan(
    token_slots: torch.Tensor,
    *,
    max_loras: int,
    block_size: int,
) -> RouteView:
    """Aligned plan grouping TOKENS by adapter slot (T-domain, V = L_cap)."""
    dummy_single_expert = torch.zeros(
        (token_slots.shape[0], 1), dtype=torch.int32, device=token_slots.device
    )
    return build_virtual_expert_routing(
        dummy_single_expert,
        token_slots,
        lora_experts_per_adapter=1,
        max_loras=max_loras,
        block_size=block_size,
        view=ROUTE_ALIGNED,
    )


def shared_gate_up_a_token_dedup(
    hidden_states: torch.Tensor,
    gate_up_a: torch.Tensor,
    token_plan: RouteView,
    rank_out_tokens: torch.Tensor,
    *,
    config,
) -> None:
    """Dedup A: one ``[slices*R]`` product per (token, adapter) row.

    ``gate_up_a`` is the flattened shared factor ``[L_cap * 1, slices*R, H]``;
    ``rank_out_tokens`` is the token-major bridge ``[T, slices*R]``.  Rows of
    base tokens (adapter -1) are sentinel-skipped and stay undefined — the
    downstream B zero-overwrites their pair destinations, same contract as
    the pair-major path.
    """
    grouped_lora_a(
        hidden_states,
        gate_up_a,
        rank_out_tokens,
        token_plan,
        config=config,
    )


def shared_gate_up_delta_from_token_bridge(
    rank_out_tokens: torch.Tensor,
    gate_up_b: torch.Tensor,
    gate_up_delta: torch.Tensor,
    per_expert_route: RouteView,
    *,
    intermediate_size: int,
    config,
) -> None:
    """Gate/up-B over the token-major bridge (``intermediate_top_k = K``)."""
    stock_grouped_lora_b(
        rank_out_tokens,
        gate_up_b,
        gate_up_delta,
        per_expert_route,
        destination_offsets=(0, intermediate_size),
        config=config,
        intermediate_top_k=per_expert_route.topk_ids.shape[1],
    )


def run_shared_gate_up(
    spec: LoraAExecutionSpec,
    *,
    hidden_states: torch.Tensor,
    gate_up_a: torch.Tensor,
    gate_up_b: torch.Tensor,
    rank_out: torch.Tensor,
    gate_up_delta: torch.Tensor,
    a_route: RouteView,
    per_expert_route: RouteView,
    intermediate_size: int,
    config_a,
    config_b,
    segment_info=None,
) -> None:
    """Execute one shared-outer gate/up form FROM its spec.

    Fourth S3 review: spec keys previously only labeled records while the
    arm methods were mapped by hand, so a label could drift from what ran.
    ``a_route`` is the form's own A-side route (pair-domain outer for the
    repeated-pair control, the T-plan for token dedup) and ``rank_out``
    must be that form's bridge shape ([P, 2R] vs [T, 2R]).

    Segmented (SGMV) specs execute their DECLARED kernel variant from
    ``segment_info`` (a ``LoRABatchInfo``) — sixth S3 review: the SGMV
    thunks were hand-mapped, so identity could drift from the kernel run.
    ``rank_out`` and ``a_route`` are ignored for segmented forms (the
    kernels allocate the token-major bridge and take their route from the
    segments); B consumes ``per_expert_route`` as in every form.
    """
    if spec.site != "gate_up" or spec.reduction != "whole_rank":
        raise NotImplementedError(f"no shared-outer executor for {spec.key()!r}")
    if spec.ownership == "segmented":
        if spec.shared_handling != "token_dedup" or spec.implementation != "triton":
            raise NotImplementedError(f"no shared-outer executor for {spec.key()!r}")
        if segment_info is None:
            raise ValueError(
                f"spec {spec.key()!r} declares segmented ownership but no "
                "segment_info (LoRABatchInfo) was supplied"
            )
        from sglang.kernels.ops.gemm.chunked_sgmv_shrink import (
            chunked_sgmv_lora_shrink_forward,
        )
        from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd

        a_weights_3d = gate_up_a.view(
            per_expert_route.max_loras, -1, hidden_states.shape[1]
        )
        if spec.variant == "chunked":
            bridge = chunked_sgmv_lora_shrink_forward(
                hidden_states, a_weights_3d, segment_info, 2
            )
        else:
            bridge = sgemm_lora_a_fwd(hidden_states, a_weights_3d, segment_info, 2)
        shared_gate_up_delta_from_token_bridge(
            bridge,
            gate_up_b,
            gate_up_delta,
            per_expert_route,
            intermediate_size=intermediate_size,
            config=config_b,
        )
        return
    if spec.ownership != "grouped" or spec.implementation != "triton":
        raise NotImplementedError(f"no shared-outer executor for {spec.key()!r}")
    if spec.shared_handling == "repeated_pairs":
        grouped_lora_a(hidden_states, gate_up_a, rank_out, a_route, config=config_a)
        stock_grouped_lora_b(
            rank_out,
            gate_up_b,
            gate_up_delta,
            per_expert_route,
            destination_offsets=(0, intermediate_size),
            config=config_b,
        )
    elif spec.shared_handling == "token_dedup":
        shared_gate_up_a_token_dedup(
            hidden_states, gate_up_a, a_route, rank_out, config=config_a
        )
        shared_gate_up_delta_from_token_bridge(
            rank_out,
            gate_up_b,
            gate_up_delta,
            per_expert_route,
            intermediate_size=intermediate_size,
            config=config_b,
        )
    else:
        raise NotImplementedError(f"no shared-outer executor for {spec.key()!r}")
