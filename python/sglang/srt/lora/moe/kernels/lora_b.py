"""LoRA-B projections over aligned or raw pairs, including in-place down updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.kernels.routing import (
    grouped_tile_coords,
    virtual_expert_ids_inline,
)
from sglang.srt.lora.moe.route_view import RouteView

if TYPE_CHECKING:
    from sglang.srt.lora.moe.execution_plan import LoraBSpec


@triton.jit
def _grouped_lora_b_kernel(
    bridge_ptr,
    weight_ptr,
    destination_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    dest_offset_0,
    dest_offset_1,
    stride_bm,
    stride_bk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_dm,
    stride_dn,
    INTERMEDIATE_TOP_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    N_PER_SLICE: tl.constexpr,
    RANK: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    CONSUME_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    tiles_per_slice: tl.constexpr = (N_PER_SLICE + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n: tl.constexpr = NUM_SLICES * tiles_per_slice
    pid_m, pid_n = grouped_tile_coords(pid, num_pid_n, NUM_M_BLOCKS, GROUP_SIZE_M)
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return

    slice_id = pid_n // tiles_per_slice
    n_tile = pid_n % tiles_per_slice
    destination_offset = tl.where(slice_id == 0, dest_offset_0, dest_offset_1).to(
        tl.int64
    )
    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    group = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)
    n_offsets = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N_PER_SLICE
    destination_ptrs = (
        destination_ptr
        + pair_ids[:, None] * stride_dm
        + (destination_offset + n_offsets)[None, :] * stride_dn
    )
    store_mask = pair_mask[:, None] & n_mask[None, :]

    if group == -1:
        # Zero sentinel destinations so graph replay cannot reuse stale deltas.
        zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tl.store(
            destination_ptrs,
            zeros.to(destination_ptr.dtype.element_ty),
            mask=store_mask,
        )
        return

    if CONSUME_PDL:
        tl.extra.cuda.gdc_wait()

    bridge_rows = pair_ids // INTERMEDIATE_TOP_K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_begin in range(0, RANK, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < RANK
        lhs = tl.load(
            bridge_ptr
            + bridge_rows[:, None] * stride_bm
            + (slice_id * RANK + k_offsets)[None, :] * stride_bk,
            mask=pair_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + group * stride_wg
            + (slice_id * N_PER_SLICE + n_offsets)[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.float32)

    tl.store(
        destination_ptrs,
        accumulator.to(destination_ptr.dtype.element_ty),
        mask=store_mask,
    )


def grouped_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
    consume_pdl: bool = False,
) -> None:
    _, weight_rows, rank = weight.shape
    num_slices = len(destination_offsets)
    slice_width = weight_rows // num_slices
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    offsets = tuple(int(offset) for offset in destination_offsets)
    block_size_n = int(config["BLOCK_SIZE_N"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_pid_n = num_slices * triton.cdiv(slice_width, block_size_n)
    _grouped_lora_b_kernel[(num_m_blocks * num_pid_n,)](
        bridge,
        weight,
        destination,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        num_pairs,
        offsets[0],
        offsets[1] if num_slices == 2 else offsets[0],
        bridge.stride(0),
        bridge.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        destination.stride(0),
        destination.stride(1),
        INTERMEDIATE_TOP_K=intermediate_top_k,
        NUM_SLICES=num_slices,
        N_PER_SLICE=slice_width,
        RANK=rank,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        GROUP_SIZE_M=int(config["GROUP_SIZE_M"]),
        CONSUME_PDL=consume_pdl,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
        **({"launch_pdl": True} if consume_pdl else {}),
    )


@triton.jit
def _per_pair_lora_b_kernel(
    bridge_ptr,
    weight_ptr,
    destination_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    num_pairs,
    routed_expert_id_bound,
    dest_offset_0,
    dest_offset_1,
    stride_bs,
    stride_bm,
    stride_bk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_dm,
    stride_dn,
    INTERMEDIATE_TOP_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    N_PER_SLICE: tl.constexpr,
    RANK: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # stride_bs selects an adapter plane when shared A used a batched GEMM.
    pair_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    tiles_per_slice: tl.constexpr = (N_PER_SLICE + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    slice_id = pid_n // tiles_per_slice
    n_tile = pid_n % tiles_per_slice

    key = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        pair_id,
        pair_id < num_pairs,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        SHARED_OUTER=SHARED_OUTER,
    )
    pair64 = pair_id.to(tl.int64)
    destination_offset = tl.where(slice_id == 0, dest_offset_0, dest_offset_1).to(
        tl.int64
    )
    n_offsets = n_tile.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(
        tl.int64
    )
    n_mask = n_offsets < N_PER_SLICE
    destination_ptrs = (
        destination_ptr
        + pair64 * stride_dm
        + (destination_offset + n_offsets) * stride_dn
    )

    if key == -1:
        # Zero invalid destinations without reading uninitialized bridge rows.
        zeros = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        tl.store(
            destination_ptrs,
            zeros.to(destination_ptr.dtype.element_ty),
            mask=n_mask,
        )
        return

    group = key.to(tl.int64)
    slot = group // LORA_EXPERTS_PER_ADAPTER
    bridge_row = pair64 // INTERMEDIATE_TOP_K
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_begin in range(0, RANK, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < RANK
        lhs = tl.load(
            bridge_ptr
            + slot * stride_bs
            + bridge_row * stride_bm
            + (slice_id * RANK + k_offsets) * stride_bk,
            mask=k_mask,
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + group * stride_wg
            + (slice_id * N_PER_SLICE + n_offsets)[:, None] * stride_wn
            + k_offsets[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator += tl.sum(rhs.to(tl.float32) * lhs[None, :].to(tl.float32), axis=1)

    tl.store(
        destination_ptrs,
        accumulator.to(destination_ptr.dtype.element_ty),
        mask=n_mask,
    )


def per_pair_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
) -> None:
    _, weight_rows, rank = weight.shape
    num_slices = len(destination_offsets)
    slice_width = weight_rows // num_slices
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    offsets = tuple(int(offset) for offset in destination_offsets)
    shared_outer = routing.is_shared_outer
    routed_bound = routing.num_local_experts
    block_size_n = int(config["BLOCK_SIZE_N"])
    stride_bs = bridge.stride(0) if bridge.dim() == 3 else 0
    _per_pair_lora_b_kernel[
        (num_pairs, num_slices * triton.cdiv(slice_width, block_size_n))
    ](
        bridge,
        weight,
        destination,
        routing.topk_ids,
        routing.token_lora_mapping,
        num_pairs,
        routed_bound,
        offsets[0],
        offsets[1] if num_slices == 2 else offsets[0],
        stride_bs,
        bridge.stride(-2),
        bridge.stride(-1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        destination.stride(0),
        destination.stride(1),
        INTERMEDIATE_TOP_K=intermediate_top_k,
        NUM_SLICES=num_slices,
        N_PER_SLICE=slice_width,
        RANK=rank,
        LORA_EXPERTS_PER_ADAPTER=routing.lora_experts_per_adapter,
        MAX_LORAS=routing.max_loras,
        TOP_K=routing.topk_ids.shape[1],
        SHARED_OUTER=shared_outer,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def run_lora_b(
    spec: LoraBSpec,
    *,
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
    consume_pdl: bool = False,
) -> None:
    family = spec.family.value
    match family:
        case "grouped":
            grouped_lora_b(
                bridge,
                weight,
                destination,
                routing,
                destination_offsets=destination_offsets,
                config=config,
                intermediate_top_k=intermediate_top_k,
                consume_pdl=consume_pdl,
            )
        case "per_pair":
            if consume_pdl:
                raise ValueError(
                    f"{family} B has no qualified programmatic-dependent-launch "
                    "consumer"
                )
            num_tokens = routing.topk_ids.shape[0]
            if (
                intermediate_top_k > 1
                and routing.max_loras > 1
                and bridge.shape[0] == routing.max_loras * num_tokens
            ):
                # A token-major bridge with one plane per slot, as the dense
                # shared A writes it (a pair-major bridge has M * top_k rows).
                bridge = bridge.view(routing.max_loras, num_tokens, -1)
            per_pair_lora_b(
                bridge,
                weight,
                destination,
                routing,
                destination_offsets=destination_offsets,
                config=config,
                intermediate_top_k=intermediate_top_k,
            )
        case _:
            raise NotImplementedError(f"no production LoRA-B executor for {family!r}")


@triton.jit
def _down_b_into_base_kernel(
    bridge_ptr,
    weight_ptr,
    down_rows_ptr,
    pair_to_row_ptr,
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
    num_pid_n: tl.constexpr = (N_HIDDEN + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m, pid_n = grouped_tile_coords(pid, num_pid_n, NUM_M_BLOCKS, GROUP_SIZE_M)
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
    # Valid groups exclude sentinel pairs whose pair_to_row was never written.
    dest_rows = tl.load(pair_to_row_ptr + pair_ids, mask=pair_mask, other=0).to(
        tl.int64
    )
    destination_ptrs = (
        down_rows_ptr + dest_rows[:, None] * stride_dm + n_offsets[None, :] * stride_dn
    )
    store_mask = pair_mask[:, None] & n_mask[None, :]

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
    pair_to_row: torch.Tensor,
    bridge: torch.Tensor,
    b_down: torch.Tensor,
    routing: RouteView,
    config: Mapping[str, int],
) -> None:
    """Add down-B into base rows addressed by pair_to_row."""
    num_tokens, top_k = routing.topk_ids.shape
    pairs = num_tokens * top_k
    hidden = down_rows.shape[1]
    rank = bridge.shape[1]
    if pairs == 0:
        return
    block_size_n = int(config["BLOCK_SIZE_N"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_pid_n = triton.cdiv(hidden, block_size_n)
    _down_b_into_base_kernel[(num_m_blocks * num_pid_n,)](
        bridge,
        b_down,
        down_rows,
        pair_to_row,
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
