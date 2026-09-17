"""MoE LoRA-B dispatch over shared grouped/per-row expand kernels.

MoE slots are pre-scaled; every reachable delta cell is written. In-place
down-B uses the grouped kernel with provider row mappings.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.lora.kernels import lora_b as shared
from sglang.srt.lora.kernels.routing import grouped_tile_coords
from sglang.srt.lora.route_view import RouteView

if TYPE_CHECKING:
    from sglang.srt.lora.moe.plan import BSpec


@functools.lru_cache(maxsize=None)
def _rows(weight_rows: int, num_slices: int) -> tuple[int, ...]:
    """The equal weight-row slices of a MoE B weight."""
    width = weight_rows // num_slices
    return tuple(s * width for s in range(num_slices + 1))


def _geometry(
    weight: torch.Tensor, destination_offsets: Sequence[int], config: Mapping[str, int]
) -> shared.SliceGeometry:
    """A MoE B weight holds equal slices of its rows, written at the given
    destination columns."""
    return shared.slice_geometry(
        _rows(weight.shape[1], len(destination_offsets)),
        config["BLOCK_SIZE_N"],
        weight.device,
        out_offsets=destination_offsets,
    )


def grouped_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    pair_bridge: bool = True,
) -> None:
    shared.grouped_lora_b(
        bridge,
        weight,
        destination,
        routing,
        geometry=_geometry(weight, destination_offsets, config),
        config=config,
        add_inplace=False,
        zero_sentinel=True,
        pair_bridge=pair_bridge,
    )


def _per_row_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    pair_bridge: bool = True,
    slot_planes: bool = False,
) -> None:
    shared.per_row_lora_b(
        bridge,
        weight,
        destination,
        routing,
        geometry=_geometry(weight, destination_offsets, config),
        config=config,
        add_inplace=False,
        zero_sentinel=True,
        pair_bridge=pair_bridge,
        slot_planes=slot_planes,
    )


def run_lora_b(
    spec: BSpec,
    *,
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    destination_offsets: Sequence[int],
    config: Mapping[str, int],
    pair_bridge: bool = True,
    slot_planes: bool = False,
) -> None:
    """``slot_planes``: ``bridge`` is [slots, tokens, N], one plane per
    adapter slot (the token_dense shrink); only the per_row family reads it."""
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
                pair_bridge=pair_bridge,
            )
        case "per_row":
            _per_row_lora_b(
                bridge,
                weight,
                destination,
                routing,
                destination_offsets=destination_offsets,
                config=config,
                pair_bridge=pair_bridge,
                slot_planes=slot_planes,
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
    block_bucket_ids_ptr,
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

    bucket_id = tl.load(block_bucket_ids_ptr + pid_m).to(tl.int64)
    if bucket_id == -1:
        return

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N_HIDDEN
    # Valid buckets exclude sentinel pairs whose pair_to_row was never written.
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
            + bucket_id * stride_wg
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
    num_tokens, top_k = routing.num_tokens, routing.width
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
        routing.block_bucket_ids,
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
