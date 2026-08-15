"""Down-B "scatter-into-base" GEMM for the BF16 MoE row domains (the
``down_b_scatter`` experiment).

Sibling of ``lora_b._one_launch_sliced_lora_b_kernel`` — the shipping kernel
is untouched — with only the OUTPUT ADDRESSING changed.  The whole point
versus a per-token down-B-in-finalize fusion (measured, rejected: without a
route-block-tiled GEMM it degrades to scalar FMA) is that the tiled
GEMM is preserved: the M schedule (aligned-route ``sorted_pair_ids`` blocks,
``GROUP_SIZE_M`` swizzle over a host-static ``NUM_M_BLOCKS`` grid), the N/K
tiling, the FP32 ``tl.dot`` accumulation, and the invalid-pair masking are
the one-launch kernel's, with the down site's single output slice constant
folded (``NUM_SLICES == 1``, so the per-slice tile arithmetic disappears).
The two deltas are in the epilogue:

* the destination row is indirect — ``down_rows[src2dst[pair]]`` instead of
  the dense pair-major ``delta[pair]`` — targeting the base down GEMM's
  output AFTER it has run, so the ``[T, K, H]`` pair-major delta buffer is
  never allocated; and
* the store is a read-modify-write ADD of the unweighted delta.  No atomics:
  each provider row corresponds to exactly one canonical routed pair in both
  row domains (masked ``e * m_max + slot`` and contiguous
  ``seg_offsets[e] + slot`` are both injective over valid pairs), each pair
  appears exactly once in ``sorted_pair_ids``, and different N tiles of one
  row touch disjoint columns.  Per-row H-vector accesses stay coalesced.

Sentinel semantics DIFFER from the pair-major twin by necessity: there the
kernel owns every delta cell and must zero ``-1``-group blocks so stale graph
memory never becomes a false delta; here the base down GEMM owns
``down_rows`` and a zero-ADD is a no-op, so ``-1`` blocks return without any
memory traffic.  That is also a correctness requirement — sentinel pairs'
``src2dst`` entries are never written by either row domain's dispatch and
must not be dereferenced.  A valid virtual-expert group implies routed pairs
(the canonical key is ``-1`` whenever ``topk_id < 0``), so every unmasked
lane's ``src2dst`` entry was written.

``src2dst`` is only READ here, so the documented hazard barring kernels that
combine an in-place ``src2dst`` STORE with bulk row copies (see
``_contig_fill_rows_kernel``) does not apply.

NUMERICS: the FP32-accumulated delta is added to the BF16 base row and the
sum is rounded to BF16 once, before the finalize's FP32 weighted top-k
reduction — whereas the shipped tail rounds the delta to BF16 separately
(pair-major) and keeps base and delta as two BF16 operands of that FP32 sum.
Output equality versus the shipped tail is therefore judged by the
established allclose discipline, not bitwise.

Row-domain agnostic BY CONSTRUCTION, like ``post_reorder_deepgemm``: every
physical row access goes through ``src2dst`` over a flat 2-D row view.  There is no PDL variant: the eligible plan shape is
fully serial with the base down GEMM between down-A and this launch, so
neither the down-A -> down-B nor the base-down -> finalize edge exists.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.routing import ROUTE_ALIGNED, RouteView

DOWN_B_SCATTER_TRITON = "triton"


@triton.jit
def _down_b_scatter_kernel(
    bridge_ptr,
    weight_ptr,
    down_rows_ptr,
    src2dst_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    stride_bm,
    stride_bk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_dm,
    stride_dn,
    N_HIDDEN: tl.constexpr,
    RANK: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """One-launch down-B tiling; epilogue scatter-adds into base rows."""
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    # Verbatim one-launch M/N schedule with the down site's single slice
    # constant folded (NUM_SLICES == 1 makes num_pid_n the plain N tiling).
    num_pid_n: tl.constexpr = (N_HIDDEN + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    programs_per_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return

    group = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)
    if group == -1:
        # Unlike the pair-major twin, B does NOT own these cells — the base
        # down GEMM does, and a zero-add is a no-op.  Sentinel pairs'
        # src2dst entries were never written by either row domain's
        # dispatch, so this early return is also what keeps them from being
        # dereferenced.
        return

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N_HIDDEN
    # A valid group implies routed pairs (the canonical key is -1 whenever
    # topk_id < 0), so every unmasked src2dst entry was written.
    dest_rows = tl.load(src2dst_ptr + pair_ids, mask=pair_mask, other=0).to(tl.int64)
    destination_ptrs = (
        down_rows_ptr + dest_rows[:, None] * stride_dm + n_offsets[None, :] * stride_dn
    )
    store_mask = pair_mask[:, None] & n_mask[None, :]

    # The down bridge is inherently pair-major (INTERMEDIATE_TOP_K == 1 in
    # the one-launch kernel's terms): bridge rows ARE pair ids.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_begin in range(0, RANK, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < RANK
        lhs = tl.load(
            bridge_ptr + pair_ids[:, None] * stride_bm + k_offsets[None, :] * stride_bk,
            mask=pair_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + group * stride_wg
            + n_offsets[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.float32)

    # Read-modify-write add; each (row, N tile) cell is owned by exactly one
    # program, so no atomics.  This is the one joint BF16 rounding of
    # base + delta the module docstring's numerics note describes.
    base = tl.load(destination_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    tl.store(
        destination_ptrs,
        (base + accumulator).to(down_rows_ptr.dtype.element_ty),
        mask=store_mask,
    )


def invoke_down_b_scatter(
    *,
    down_rows: torch.Tensor,
    src2dst: torch.Tensor,
    bridge: torch.Tensor,
    b_down: torch.Tensor,
    routing: RouteView,
    config: Mapping[str, int],
) -> None:
    """Scatter-add each routed pair's unweighted down-B delta into base rows.

    ``down_rows`` is the provider's S4 output flattened to ``[rows, H]`` —
    masked slab and contiguous compact buffer alike — indexed only through
    ``src2dst``.  ``bridge`` is the canonical pair-major down-A output and
    ``b_down`` the flattened ``[V, H, rank]`` down-B groups, exactly the
    operands of the standalone one-launch down-B whose tiling this launch
    keeps; ``config`` is the down-B site's launch config with the same field
    semantics (``BLOCK_SIZE_M``, when present, must equal the aligned
    route's block size, as in the standalone launcher).
    """
    if routing.view != ROUTE_ALIGNED:
        raise ValueError(
            f"down-B scatter needs route view {ROUTE_ALIGNED!r}, got "
            f"{routing.view!r}"
        )
    num_tokens, top_k = routing.topk_ids.shape
    pairs = num_tokens * top_k
    if down_rows.ndim != 2 or down_rows.shape[1] < 1:
        raise ValueError("down_rows must be a flat [rows, hidden] view")
    hidden = down_rows.shape[1]
    if src2dst.dtype != torch.int32 or src2dst.numel() != pairs:
        raise ValueError(f"src2dst must be int32 with {pairs} entries")
    if not src2dst.is_contiguous():
        raise ValueError("src2dst must be contiguous")
    if bridge.ndim != 2 or bridge.shape[0] != pairs:
        raise ValueError(f"bridge must have {pairs} pair-major rows")
    rank = bridge.shape[1]
    if rank < 1:
        raise ValueError("the down bridge rank must be positive")
    num_groups = routing.max_loras * routing.lora_experts_per_adapter
    if b_down.shape != (num_groups, hidden, rank):
        raise ValueError(
            f"b_down must be {(num_groups, hidden, rank)}, got "
            f"{tuple(b_down.shape)}"
        )
    if (
        down_rows.dtype != torch.bfloat16
        or bridge.dtype != torch.bfloat16
        or b_down.dtype != torch.bfloat16
    ):
        raise TypeError("down-B scatter requires BF16 base rows, bridge, and down-B")
    if "BLOCK_SIZE_M" in config:
        configured_block = int(config["BLOCK_SIZE_M"])
        if configured_block != routing.block_size:
            raise ValueError(
                "down-B scatter consumes the aligned route's exact "
                f"BLOCK_SIZE_M: config declares {configured_block}, route "
                f"uses {routing.block_size}"
            )
    if pairs == 0:
        return
    tensors = (
        down_rows,
        src2dst,
        bridge,
        b_down,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
    )
    if len({item.device for item in tensors}) != 1:
        raise ValueError("down-B scatter tensors must share one device")

    block_size_n = int(config["BLOCK_SIZE_N"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_pid_n = triton.cdiv(hidden, block_size_n)
    _down_b_scatter_kernel[(num_m_blocks * num_pid_n,)](
        bridge,
        b_down,
        down_rows,
        src2dst,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        pairs,
        bridge.stride(0),
        bridge.stride(1),
        b_down.stride(0),
        b_down.stride(1),
        b_down.stride(2),
        down_rows.stride(0),
        down_rows.stride(1),
        N_HIDDEN=hidden,
        RANK=rank,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        GROUP_SIZE_M=int(config["GROUP_SIZE_M"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
