"""Down-B GEMM that adds its result into the base down output.

This kernel is the one-launch down-B kernel with one change: the output
address. It writes to ``down_rows[src2dst[pair]]``, not to a pair-major
``delta`` buffer, so no code allocates that buffer. The store is a
read-modify-write add. A benchmark measured a per-token fusion into finalize.
This file does not use that fusion, because the fusion loses the block tiling
and does scalar work.

The add needs no atomics. Each row maps to one routed pair. Each pair occurs
once in ``sorted_pair_ids``. Different N tiles write to different columns.

A block with group ``-1`` returns immediately. The base GEMM writes those rows,
and an add of zero changes nothing. The early return is also a safety rule. The
dispatch does not write ``src2dst`` for a sentinel pair, so the kernel must not
read it.

NUMERICS: this path adds the FP32 delta to the BF16 base row, then rounds once.
The shipped path rounds the delta first. Thus you must compare the two paths
with allclose, not with an exact test.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.route_view import RouteView, RouteViewKind


@triton.jit
def _down_b_into_base_kernel(
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
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    # This is the M/N schedule of the one-launch kernel. The down site has one
    # slice, so num_pid_n folds down to the plain N tiling.
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
        return

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N_HIDDEN
    # The fused key is -1 for every pair with a topk_id below 0. A valid group
    # therefore means the dispatch wrote each unmasked src2dst entry.
    dest_rows = tl.load(src2dst_ptr + pair_ids, mask=pair_mask, other=0).to(tl.int64)
    destination_ptrs = (
        down_rows_ptr + dest_rows[:, None] * stride_dm + n_offsets[None, :] * stride_dn
    )
    store_mask = pair_mask[:, None] & n_mask[None, :]

    # The down bridge is pair-major. A bridge row index is a pair id.
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

    base = tl.load(destination_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    tl.store(
        destination_ptrs,
        (base + accumulator).to(down_rows_ptr.dtype.element_ty),
        mask=store_mask,
    )


def invoke_down_b_into_base(
    *,
    down_rows: torch.Tensor,
    src2dst: torch.Tensor,
    bridge: torch.Tensor,
    b_down: torch.Tensor,
    routing: RouteView,
    config: Mapping[str, int],
) -> None:
    if routing.view is not RouteViewKind.ALIGNED:
        raise ValueError(
            f"down-B into-base needs route view {RouteViewKind.ALIGNED.value!r}, got "
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
        raise TypeError("down-B into-base requires BF16 base rows, bridge, and down-B")
    if "BLOCK_SIZE_M" in config:
        configured_block = int(config["BLOCK_SIZE_M"])
        if configured_block != routing.block_size:
            raise ValueError(
                "down-B into-base consumes the aligned route's exact "
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
        raise ValueError("down-B into-base tensors must share one device")

    block_size_n = int(config["BLOCK_SIZE_N"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_pid_n = triton.cdiv(hidden, block_size_n)
    _down_b_into_base_kernel[(num_m_blocks * num_pid_n,)](
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
