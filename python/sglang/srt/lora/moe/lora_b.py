"""Selector-free BF16 LoRA-B candidate kernels for the SGL MoE backend.

Only the production shortlist lives here:

* one-launch sliced grouped B, the general promoted family; and
* pair-indexed sliced B, the decode experiment behind
  the shipped per-expert decode B family: its grid covers
  occupied routed pairs instead of aligned expert-major M blocks, so a
  sparse decode route (hundreds of distinct experts at one pair each) pays
  no block padding.

Rejected deterministic rank-split B and B-SGMV/CSGMV families are not copied
into production.  The caller owns route and launch-config selection.

Port provenance:

* ``one_launch_sliced_lora_b`` mirrors function
  ``invoke_one_launch_sliced_lora_b`` in
  ``benchmark.kernels.lora_moe.lora_b_candidates``; and
* ``indexed_pairs_lora_b`` models its raw-route indexing on the H200
  ``_indexed_lora_a_kernel`` and keeps ``one_launch_sliced``'s destination,
  slice-folding, and K-loop arithmetic.

The production bodies preserve those launchers' tile geometry, config field
semantics, route use, and invalid-pair zero-store contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.routing import RouteView, virtual_expert_ids_inline

if TYPE_CHECKING:
    from sglang.srt.lora.moe.execution_plan import LoraBSpec

MAX_SLICES = 2


def _spec_value(spec: object, field: str) -> str:
    value = getattr(spec, field, None)
    if value is None:
        raise ValueError(f"LoRA-B execution spec is missing {field!r}")
    return str(getattr(value, "value", value))


def _validate_b_call(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    destination_offsets: Sequence[int],
    intermediate_top_k: int,
) -> tuple[int, int, int]:
    """Fail-closed common contract; return slices, width, and rank."""
    num_slices = len(destination_offsets)
    if not 1 <= num_slices <= MAX_SLICES:
        raise ValueError(f"LoRA-B supports 1..{MAX_SLICES} slices")
    if weight.ndim != 3:
        raise ValueError(f"weight must be 3D, got shape {tuple(weight.shape)}")
    num_groups, weight_rows, rank = weight.shape
    if weight_rows % num_slices:
        raise ValueError(
            f"weight rows {weight_rows} not divisible by {num_slices} slices"
        )
    slice_width = weight_rows // num_slices

    offsets = tuple(int(offset) for offset in destination_offsets)
    ordered = sorted(offsets)
    if ordered[0] < 0:
        raise ValueError(f"destination offsets must be non-negative: {ordered}")
    for low, high in zip(ordered, ordered[1:]):
        if high - low < slice_width:
            raise ValueError(
                f"destination offsets {ordered} overlap at width {slice_width}"
            )

    expected_groups = routing.max_loras * routing.lora_experts_per_adapter
    if num_groups != expected_groups:
        raise ValueError(
            f"weight groups {num_groups} != max_loras * "
            f"lora_experts_per_adapter {expected_groups}"
        )
    num_tokens, top_k = routing.topk_ids.shape
    num_pairs = routing.topk_ids.numel()
    if intermediate_top_k == 1:
        expected_rows = num_pairs
    elif intermediate_top_k == top_k:
        expected_rows = num_tokens
    else:
        raise ValueError(
            f"intermediate_top_k must be 1 or route top-k {top_k}, "
            f"got {intermediate_top_k}"
        )
    if bridge.ndim != 2 or bridge.shape != (
        expected_rows,
        num_slices * rank,
    ):
        raise ValueError(
            f"bridge must have shape {(expected_rows, num_slices * rank)}, "
            f"got {tuple(bridge.shape)}"
        )
    if destination.ndim != 2 or destination.shape[0] != num_pairs:
        raise ValueError(f"destination must have {num_pairs} rows")
    for offset in offsets:
        if offset + slice_width > destination.shape[1]:
            raise ValueError(
                f"destination offset {offset} + width {slice_width} exceeds "
                f"{destination.shape[1]} columns"
            )
    devices = {
        bridge.device,
        weight.device,
        destination.device,
        routing.topk_ids.device,
    }
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {sorted(map(str, devices))}")
    return num_slices, slice_width, rank


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
    """All one/two output slices in one aligned-plan launch."""
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
        # B owns these cells: zeroing sentinels prevents stale graph-buffer
        # contents from becoming a false LoRA delta. This path never reads A's
        # bridge and therefore must not wait on the producer.
        zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tl.store(
            destination_ptrs,
            zeros.to(destination_ptr.dtype.element_ty),
            mask=store_mask,
        )
        return

    if CONSUME_PDL:
        # Route decoding, destination-address calculation, and sentinel stores
        # are independent of A. Wait only immediately before the first bridge
        # load in a CTA that actually consumes A's stores.
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
    num_slices, slice_width, rank = _validate_b_call(
        bridge,
        weight,
        destination,
        routing,
        destination_offsets=destination_offsets,
        intermediate_top_k=intermediate_top_k,
    )
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
    token_slots_ptr,
    lora_expert_map_ptr,
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
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One raw-route pair and one sliced N tile per program.

    The aligned one-launch kernel pays a full ``BLOCK_SIZE_M`` tensor-core
    tile per occupied virtual expert, which at sparse decode routes is
    almost entirely block padding.  This variant derives the pair's virtual
    expert key inline (the one canonical definition), so the M axis is
    exactly the occupied pairs: no sort, no capacity padding, no
    ``num_pairs_post_padded`` device scalar.

    Numerics: the K loop steps ``BLOCK_SIZE_K`` over ``RANK`` with one FP32
    accumulator per destination cell — the same per-pair k-tile order as
    ``_one_launch_sliced_lora_b_kernel``.  BF16 products are exact in FP32
    in both kernels, so results differ from one-launch only by the
    within-tile summation order (``tl.sum`` here, the ``tl.dot`` MMA tree
    there); oracles compare with the established allclose discipline.
    """
    pair_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    tiles_per_slice: tl.constexpr = (N_PER_SLICE + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    slice_id = pid_n // tiles_per_slice
    n_tile = pid_n % tiles_per_slice

    key = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        lora_expert_map_ptr,
        pair_id,
        pair_id < num_pairs,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=USE_LORA_EXPERT_MAP,
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
        # B owns these cells: zeroing invalid pairs prevents stale
        # graph-buffer contents from becoming a false LoRA delta (the same
        # contract as the one-launch sentinel-block store).
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
    """Execute raw-route pair-indexed sliced B.

    Consumes only the raw source tensors carried on every ``RouteView``
    (``topk_ids``/``token_slots``); an execution plan choosing this family
    requests ``ROUTE_RAW`` so no aligned pair plan is built merely for this
    stage.  The ``(num_pairs, n_tiles)`` grid is static per CUDA-graph
    capture bucket (``num_pairs == T * top_k``), matching the graph-captured
    indexed down-A precedent.  This launcher has no PDL operations.
    """
    num_slices, slice_width, rank = _validate_b_call(
        bridge,
        weight,
        destination,
        routing,
        destination_offsets=destination_offsets,
        intermediate_top_k=intermediate_top_k,
    )
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    offsets = tuple(int(offset) for offset in destination_offsets)
    use_map = routing.lora_expert_map is not None
    shared_outer = routing.shared_outer_local_expert_count is not None
    routed_bound = (
        routing.shared_outer_local_expert_count
        if shared_outer
        else (routing.lora_expert_map.numel() if use_map else 0)
    )
    map_arg = routing.lora_expert_map if use_map else routing.topk_ids
    block_size_n = int(config["BLOCK_SIZE_N"])
    _indexed_pairs_lora_b_kernel[
        (num_pairs, num_slices * triton.cdiv(slice_width, block_size_n))
    ](
        bridge,
        weight,
        destination,
        routing.topk_ids,
        routing.token_slots,
        map_arg,
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
        USE_LORA_EXPERT_MAP=use_map,
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
    """Execute exactly the B family named by an execution-plan spec."""
    family = _spec_value(spec, "family")
    if consume_pdl and family != "one_launch_sliced":
        raise ValueError(
            f"{family} B has no qualified programmatic-dependent-launch consumer"
        )
    if family == "one_launch_sliced":
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
        return
    if family == "indexed_pairs":
        indexed_pairs_lora_b(
            bridge,
            weight,
            destination,
            routing,
            destination_offsets=destination_offsets,
            config=config,
            intermediate_top_k=intermediate_top_k,
        )
        return
    raise NotImplementedError(f"no production LoRA-B executor for {family!r}")
