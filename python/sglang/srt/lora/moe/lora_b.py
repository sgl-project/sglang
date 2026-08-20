"""The two LoRA-B families that an execution plan can name.

One-launch sliced B reads the aligned route. It is the general choice.
Pair-indexed sliced B reads the raw route. Its grid covers the occupied pairs,
not whole expert blocks. A sparse decode route can hold hundreds of experts with
one pair each. This family then does no padded work.

A plan can also ask one-launch sliced B to add its result into the base down
output. ``invoke_down_b_into_base`` does that. The tiling is the same. Only the
output address changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.route_kernels import virtual_expert_ids_inline
from sglang.srt.lora.moe.route_view import RouteView

if TYPE_CHECKING:
    from sglang.srt.lora.moe.execution_plan import LoraBSpec


@triton.jit
def _one_launch_sliced_lora_b_kernel(
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
    """Compute one or two output slices in a single launch over the aligned route."""
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    tiles_per_slice: tl.constexpr = (N_PER_SLICE + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n: tl.constexpr = NUM_SLICES * tiles_per_slice
    programs_per_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m
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
        # This kernel writes every cell that it can reach. It must zero a
        # sentinel block, or stale CUDA-graph memory becomes a false LoRA
        # delta. This path never reads the A bridge, so it must not wait on
        # the producer.
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


def one_launch_sliced_lora_b(
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
    _one_launch_sliced_lora_b_kernel[(num_m_blocks * num_pid_n,)](
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
def _indexed_pairs_lora_b_kernel(
    bridge_ptr,
    weight_ptr,
    destination_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    num_pairs,
    routed_expert_id_bound,
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
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One program handles one raw-route pair and one N tile.

    This family sorts nothing and pads nothing. It sums a tile in another order
    than one-launch B. Thus you must compare the two with allclose, not with an
    exact test.
    """
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
        # This kernel writes every cell that it can reach. It must zero an
        # invalid pair, or stale CUDA-graph memory becomes a false LoRA delta.
        zeros = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        tl.store(
            destination_ptrs,
            zeros.to(destination_ptr.dtype.element_ty),
            mask=n_mask,
        )
        return

    group = key.to(tl.int64)
    bridge_row = pair64 // INTERMEDIATE_TOP_K
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_begin in range(0, RANK, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < RANK
        lhs = tl.load(
            bridge_ptr
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


def indexed_pairs_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    intermediate_top_k: int = 1,
) -> None:
    """Run B over the raw route.

    This family builds no aligned route. Its grid is a constant for each
    CUDA-graph capture size. It cannot consume a programmatic dependent launch
    (PDL) signal.
    """
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
    _indexed_pairs_lora_b_kernel[
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
    """Run the family that the spec names. There is no fallback and no selector."""
    family = spec.family.value
    match family:
        case "one_launch_sliced":
            one_launch_sliced_lora_b(
                bridge,
                weight,
                destination,
                routing,
                destination_offsets=destination_offsets,
                config=config,
                intermediate_top_k=intermediate_top_k,
                consume_pdl=consume_pdl,
            )
        case "indexed_pairs":
            if consume_pdl:
                raise ValueError(
                    f"{family} B has no qualified programmatic-dependent-launch "
                    "consumer"
                )
            indexed_pairs_lora_b(
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
    """Add each routed pair's down-B result into the base down rows.

    ``down_rows`` is the provider output as a flat ``[rows, hidden]`` view. The
    kernel reaches it only through ``src2dst``, so both row domains work.
    """
    num_tokens, top_k = routing.topk_ids.shape
    pairs = num_tokens * top_k
    hidden = down_rows.shape[1]
    rank = bridge.shape[1]
    if "BLOCK_SIZE_M" in config:
        # The tile tables do not pin this to the route, and a mismatch reads
        # the wrong stride of sorted_pair_ids without an error.
        configured_block = int(config["BLOCK_SIZE_M"])
        if configured_block != routing.block_size:
            raise ValueError(
                "down-B into-base consumes the aligned route's exact "
                f"BLOCK_SIZE_M: config declares {configured_block}, route "
                f"uses {routing.block_size}"
            )
    if pairs == 0:
        return
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
