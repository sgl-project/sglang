"""LoRA-B (expand) kernels shared by the dense and MoE engines: grouped over
an aligned route, per pair over a raw route. Slice ``s`` of a stacked site is
weight rows [offsets[s], offsets[s+1]) and bridge columns [s*r, (s+1)*r), read
from the offset tables or, for equal slices, from constants. Dense slots
carry a rank and a scaling (``lora_ranks`` / ``scalings``); MoE slots are
pre-scaled and zero-fill unrouted rows (ZERO_SENTINEL).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.lora.common.route_view import RouteView
from sglang.kernels.ops.lora.common.routing import (
    grouped_tile_coords,
    route_bucket_ids,
)


@triton.jit
def _slice_of_tile(
    slice_offsets_ptr, pid_n, NUM_SLICES: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Map a column tile to (slice, tile in slice, slice start, width);
    -1 for a tile past the last slice."""
    start = tl.zeros((), dtype=tl.int32)
    slice_id = tl.full((), -1, tl.int32)
    n_tile = tl.zeros((), dtype=tl.int32)
    n_start = tl.zeros((), dtype=tl.int64)
    width = tl.zeros((), dtype=tl.int64)
    for s in tl.static_range(NUM_SLICES):
        lo = tl.load(slice_offsets_ptr + s).to(tl.int64)
        hi = tl.load(slice_offsets_ptr + s + 1).to(tl.int64)
        tiles = ((hi - lo + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N).to(tl.int32)
        hit = (pid_n >= start) & (pid_n < start + tiles)
        slice_id = tl.where(hit, s, slice_id)
        n_tile = tl.where(hit, pid_n - start, n_tile)
        n_start = tl.where(hit, lo, n_start)
        width = tl.where(hit, hi - lo, width)
        start += tiles
    return slice_id, n_tile, n_start, width


class SliceGeometry(NamedTuple):
    """Map expand tiles to weight slices and destination columns.

    Zero uniform_width/out_stride denotes irregular geometry. full_tiles means
    every slice width is a multiple of the column tile.
    """

    slice_offsets: torch.Tensor  # [S + 1] int32, the weight rows of each slice
    out_offsets: torch.Tensor  # [S] int32, the destination column of each slice
    num_slices: int
    num_column_tiles: int
    uniform_width: int
    out_stride: int
    full_tiles: bool
    block_size_n: int


_GEOMETRY: dict[tuple, SliceGeometry] = {}


def _geometry_inputs(offsets, block_size_n, out_offsets):
    if (
        type(block_size_n) is not int
        or block_size_n <= 0
        or block_size_n & (block_size_n - 1)
    ):
        raise ValueError("block_size_n must be a positive power of two")
    try:
        rows = tuple(offsets)
        columns = rows[:-1] if out_offsets is None else tuple(out_offsets)
    except TypeError:
        raise ValueError("slice offsets must be integer sequences") from None
    if len(rows) < 2 or any(type(o) is not int for o in rows) or rows[0] != 0:
        raise ValueError("offsets must be increasing int32 prefixes starting at zero")
    if out_offsets is not None and (
        len(columns) != len(rows) - 1 or any(type(o) is not int for o in columns)
    ):
        raise ValueError("out_offsets needs one nonnegative int32 entry per slice")
    return rows, columns


def slice_geometry(
    offsets: Sequence[int],
    block_size_n: int,
    device: torch.device,
    out_offsets: Sequence[int] | None = None,
) -> SliceGeometry:
    """Cache weight slices ``offsets`` [S+1] and destination columns [S].

    None out_offsets uses weight-row offsets. Entries are never evicted because
    graphs retain their device tables. Blocking host-to-device initialization
    must precede capture; a cache miss during capture raises.
    """
    offsets, out_offsets = _geometry_inputs(offsets, block_size_n, out_offsets)
    if device.type != "cuda":
        raise ValueError(_NOT_CUDA)
    # An index-less device may name a different current device on each call.
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (offsets, out_offsets, block_size_n, device)
    geometry = _GEOMETRY.get(key)
    if geometry is None:
        geometry = _build_geometry(offsets, block_size_n, device, out_offsets)
        _GEOMETRY[key] = geometry
    return geometry


_NOT_CUDA = "the expand kernels read their offset tables from a CUDA device"


def _build_geometry(
    rows: tuple[int, ...],
    block_size_n: int,
    device: torch.device,
    columns: tuple[int, ...],
) -> SliceGeometry:
    if any(o < 0 or o >= 2**31 for o in rows) or any(
        lo >= hi for lo, hi in zip(rows, rows[1:])
    ):
        raise ValueError("offsets must be increasing int32 prefixes starting at zero")
    if any(o < 0 or o >= 2**31 for o in columns):
        raise ValueError("out_offsets needs one nonnegative int32 entry per slice")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"slice geometry {(rows, columns, block_size_n, device)} was first "
                "needed inside a CUDA graph capture; run the site eagerly once "
                "before capturing"
            )
    widths = [hi - lo for lo, hi in zip(rows, rows[1:])]
    uniform = widths[0] if len(set(widths)) == 1 else 0
    stride = columns[1] if len(columns) > 1 else uniform
    regular = stride > 0 and all(c == s * stride for s, c in enumerate(columns))
    # Share an allocation for cache locality; align the columns to 16 bytes.
    head = -(-len(rows) // 4) * 4
    table = torch.tensor(
        list(rows) + [0] * (head - len(rows)) + list(columns),
        dtype=torch.int32,
        device=device,
    )
    return SliceGeometry(
        slice_offsets=table[: len(rows)],
        out_offsets=table[head:],
        num_slices=len(widths),
        num_column_tiles=sum(triton.cdiv(w, block_size_n) for w in widths),
        uniform_width=uniform,
        out_stride=stride if regular else 0,
        full_tiles=all(w % block_size_n == 0 for w in widths),
        block_size_n=block_size_n,
    )


@triton.jit
def _grouped_lora_b_kernel(
    bridge_ptr,
    weight_ptr,
    destination_ptr,
    sorted_pair_ids_ptr,
    block_bucket_ids_ptr,
    num_pairs_post_padded_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    slice_offsets_ptr,
    out_offsets_ptr,
    num_pairs,
    num_pid_n: tl.constexpr,
    stride_bp,
    stride_bm,
    stride_bk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_dn,
    stride_dt,
    stride_dh,
    WIDTH: tl.constexpr,
    PAIR_BRIDGE: tl.constexpr,
    RANK: tl.constexpr,
    ADD_INPLACE: tl.constexpr,
    ZERO_SENTINEL: tl.constexpr,
    GROUPS_PER_SLOT: tl.constexpr,
    WEIGHT_DIV: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    UNIFORM_WIDTH: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    FULL_TILES: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    SORTED_CAPACITY: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BRIDGE_SLICES: tl.constexpr,
    PLANES: tl.constexpr = 1,
    PAIR_HEADS: tl.constexpr = 1,
):
    # lora_ranks_ptr / scalings_ptr are None for pre-scaled full-rank slots.
    # PAIR_BRIDGE: the bridge has one row per pair; otherwise one per token
    # (row = pair // WIDTH). The slot of a bucket is bucket // GROUPS_PER_SLOT,
    # its weight plane bucket // WEIGHT_DIV. BRIDGE_SLICES: slice s reads bridge
    # block s % BRIDGE_SLICES (NUM_SLICES = one block per slice; fewer = a
    # shared shrink expanded per expert). PLANES > 1: the bridge is the split-K
    # shrink's fp32 planes [PLANES, rows, N]; they are summed here in plane
    # order, so no reduce kernel runs between shrink and expand. Pair row p
    # lands at row p // PAIR_HEADS, head p % PAIR_HEADS of the destination:
    # PAIR_HEADS = 1 is a plain [rows, width] destination (stride_dh unused),
    # more is a [tokens, heads, width] destination with its own token and head
    # strides (a transposed bmm result), accumulated in place.
    pid = tl.program_id(0)
    if UNIFORM_WIDTH > 0:
        tiles_per_slice: tl.constexpr = (
            UNIFORM_WIDTH + BLOCK_SIZE_N - 1
        ) // BLOCK_SIZE_N
        pid_m, pid_n = grouped_tile_coords(
            pid, NUM_SLICES * tiles_per_slice, NUM_M_BLOCKS, GROUP_SIZE_M
        )
    else:
        pid_m, pid_n = grouped_tile_coords(pid, num_pid_n, NUM_M_BLOCKS, GROUP_SIZE_M)
    # Issue route loads together, but the JIT route's final block can be short.
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    bucket_id = tl.load(block_bucket_ids_ptr + pid_m)
    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(
        sorted_pair_ids_ptr + pair_slots,
        mask=pair_slots < SORTED_CAPACITY,
        other=num_pairs,
    ).to(tl.int64)
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return
    if not ZERO_SENTINEL:
        if bucket_id == -1:
            return
    slot = bucket_id // GROUPS_PER_SLOT
    if UNIFORM_WIDTH > 0:
        slice_id = pid_n // tiles_per_slice
        n_tile = pid_n % tiles_per_slice
        n_start = (slice_id * UNIFORM_WIDTH).to(tl.int64)
        width = UNIFORM_WIDTH
    else:
        slice_id, n_tile, n_start, width = _slice_of_tile(
            slice_offsets_ptr, pid_n, NUM_SLICES, BLOCK_SIZE_N
        )
        if slice_id < 0:
            return
    if OUT_STRIDE > 0:
        out_start = (slice_id * OUT_STRIDE).to(tl.int64)
    else:
        out_start = tl.load(out_offsets_ptr + slice_id).to(tl.int64)

    pair_mask = pair_ids < num_pairs
    n_offsets = n_tile.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(
        tl.int64
    )
    if FULL_TILES:
        # Every slice is whole tiles wide: no column mask to carry.
        n_mask = tl.full((BLOCK_SIZE_N,), 1, tl.int1)
    else:
        n_mask = n_offsets < width
    row_offsets = (pair_ids // PAIR_HEADS) * stride_dt + (
        pair_ids % PAIR_HEADS
    ) * stride_dh
    destination_ptrs = (
        destination_ptr
        + row_offsets[:, None]
        + (out_start + n_offsets)[None, :] * stride_dn
    )
    store_mask = pair_mask[:, None] & n_mask[None, :]
    if ZERO_SENTINEL:
        if bucket_id == -1:
            # A fresh delta buffer must not keep stale graph memory.
            zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            tl.store(
                destination_ptrs,
                zeros.to(destination_ptr.dtype.element_ty),
                mask=store_mask,
            )
            return

    if lora_ranks_ptr is not None:
        rank = tl.load(lora_ranks_ptr + slot).to(tl.int32)
    else:
        rank = RANK
    if scalings_ptr is not None:
        scaling = tl.load(scalings_ptr + slot).to(tl.float32)
    plane = (bucket_id // WEIGHT_DIV).to(tl.int64)
    if PAIR_BRIDGE:
        bridge_rows = pair_ids
    else:
        bridge_rows = pair_ids // WIDTH
    # Read the prior output before the dot so the load overlaps the operands.
    if ADD_INPLACE:
        prior = tl.load(destination_ptrs, mask=store_mask, other=0.0).to(tl.float32)

    if lora_ranks_ptr is not None:
        if rank == 0:
            return
    # A single slice starts at column 0; shared bridge blocks need modulo.
    if NUM_SLICES == 1:
        bridge_col0 = tl.zeros((), dtype=tl.int64)
    elif BRIDGE_SLICES < NUM_SLICES:
        bridge_col0 = (slice_id % BRIDGE_SLICES).to(tl.int64) * rank
    else:
        bridge_col0 = slice_id.to(tl.int64) * rank
    # accumulator[rows, BLOCK_N] = bridge[rows, col0:col0+rank] @ weight[plane,
    # n_start+n, :rank].T. The first rank block is loaded under the static
    # mask (the pool's rank), the slot's own rank applied afterwards with a
    # select, so the loads do not wait for it.
    K0: tl.constexpr = min(BLOCK_SIZE_K, RANK)
    k_offsets = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    k_static = k_offsets < K0
    weight_tile = (
        weight_ptr
        + plane * stride_wg
        + (n_start + n_offsets)[None, :] * stride_wn
        + k_offsets[:, None] * stride_wk
    )
    rhs = tl.load(weight_tile, mask=n_mask[None, :] & k_static[:, None], other=0.0)
    lhs_ptrs = (
        bridge_ptr
        + bridge_rows[:, None] * stride_bm
        + (bridge_col0 + k_offsets)[None, :] * stride_bk
    )
    lhs_mask = pair_mask[:, None] & k_static[None, :]
    lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
    if PLANES > 1:
        lhs = lhs.to(tl.float32)
        for p in tl.static_range(1, PLANES):
            lhs += tl.load(lhs_ptrs + p * stride_bp, mask=lhs_mask, other=0.0).to(
                tl.float32
            )
    if lora_ranks_ptr is not None:
        k_mask = k_offsets < rank
        # Select, not multiply: the unselected lanes may hold anything.
        lhs = tl.where(k_mask[None, :], lhs, 0.0)
        rhs = tl.where(k_mask[:, None], rhs, 0.0)
    # Match a locally materialized bridge's rounding before B; an intervening
    # TP reduction can change that rounding order.
    accumulator = tl.dot(lhs.to(weight_ptr.dtype.element_ty), rhs, out_dtype=tl.float32)
    for k_begin in range(BLOCK_SIZE_K, RANK, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < rank
        lhs_ptrs = (
            bridge_ptr
            + bridge_rows[:, None] * stride_bm
            + (bridge_col0 + k_offsets)[None, :] * stride_bk
        )
        lhs_mask = pair_mask[:, None] & k_mask[None, :]
        lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
        if PLANES > 1:
            lhs = lhs.to(tl.float32)
            for p in tl.static_range(1, PLANES):
                lhs += tl.load(lhs_ptrs + p * stride_bp, mask=lhs_mask, other=0.0).to(
                    tl.float32
                )
        rhs = tl.load(
            weight_ptr
            + plane * stride_wg
            + (n_start + n_offsets)[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(
            lhs.to(weight_ptr.dtype.element_ty), rhs, out_dtype=tl.float32
        )

    if scalings_ptr is not None:
        accumulator *= scaling
    if ADD_INPLACE:
        accumulator += prior
    tl.store(
        destination_ptrs,
        accumulator.to(destination_ptr.dtype.element_ty),
        mask=store_mask,
    )


def _k_tile(block_k: int, rank: int) -> int:
    """Limit the configured tile to rounded-up rank, with MMA's minimum of 16."""
    return max(16, min(block_k, triton.next_power_of_2(rank)))


def grouped_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    geometry: SliceGeometry,
    config: Mapping[str, int],
    add_inplace: bool,
    zero_sentinel: bool,
    pair_bridge: bool = True,
    lora_ranks: torch.Tensor | None = None,
    scalings: torch.Tensor | None = None,
    weight_div: int = 1,
    planes: int = 1,
    pair_heads: int = 0,
    bridge_slices: int = 0,
) -> None:
    """destination[pair, o_s:o_s+w_s] (+)= [scaling *] bridge[row, s*rank:(s+1)*rank] @ weight[plane, off_s:off_s+w_s, :rank].T
    ``pair_bridge``: the bridge has one row per pair; False = one row per token
    (row = pair // ``routing.width``). ``pair_heads`` > 0: ``destination`` is
    ``[tokens, heads, width]`` with any strides and pair row ``p`` lands at
    ``[p // heads, p % heads]``; otherwise ``destination`` is ``[rows, width]``.
    ``planes`` > 1: ``bridge`` is the split-K shrink's fp32 [planes, rows, rank] and is summed on load.

    ``geometry`` (see ``slice_geometry``) maps the column tiles to weight-row
    slices and destination columns.
    ``lora_ranks`` / ``scalings`` None = full rank, pre-scaled weights; a
    slot is its bucket // ``routing.groups_per_slot``, its weight plane the
    bucket // ``weight_div``. ``bridge_slices`` > 0: slice s reads bridge block
    s % bridge_slices (0 = one block per slice)."""
    if geometry.block_size_n != config["BLOCK_SIZE_N"]:
        raise ValueError("expand BLOCK_SIZE_N must match its slice geometry")
    num_pairs = routing.num_rows
    if num_pairs == 0:
        return
    if bridge.dim() != 2 + (planes > 1):
        raise ValueError(
            "the grouped expand reads a [rows, rank] bridge (or [planes, rows, rank] "
            "fp32 planes); an all_slots bridge (one plane per slot) needs the "
            "per-row expand"
        )
    if pair_heads:
        if destination.dim() != 3 or destination.shape[1] != pair_heads:
            raise ValueError(
                "a pair-addressed destination is [tokens, heads, width] with heads == pair_heads"
            )
        stride_dt, stride_dh, stride_dn = destination.stride()
    else:
        stride_dt, stride_dh, stride_dn = (
            destination.stride(0),
            0,
            destination.stride(1),
        )
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    _grouped_lora_b_kernel[(num_m_blocks * geometry.num_column_tiles,)](
        bridge,
        weight,
        destination,
        routing.sorted_pair_ids,
        routing.block_bucket_ids,
        routing.num_pairs_post_padded,
        lora_ranks,
        scalings,
        geometry.slice_offsets,
        geometry.out_offsets,
        num_pairs,
        geometry.num_column_tiles,
        bridge.stride(0) if planes > 1 else 0,
        bridge.stride(-2),
        bridge.stride(-1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        stride_dn,
        stride_dt,
        stride_dh,
        WIDTH=routing.width,
        PAIR_BRIDGE=pair_bridge,
        RANK=weight.shape[2],
        ADD_INPLACE=add_inplace,
        ZERO_SENTINEL=zero_sentinel,
        GROUPS_PER_SLOT=routing.groups_per_slot,
        WEIGHT_DIV=weight_div,
        NUM_SLICES=geometry.num_slices,
        UNIFORM_WIDTH=geometry.uniform_width,
        OUT_STRIDE=geometry.out_stride,
        FULL_TILES=geometry.full_tiles,
        NUM_M_BLOCKS=num_m_blocks,
        SORTED_CAPACITY=routing.sorted_pair_ids.numel(),
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=int(config["BLOCK_SIZE_N"]),
        BLOCK_SIZE_K=_k_tile(int(config["BLOCK_SIZE_K"]), weight.shape[2]),
        GROUP_SIZE_M=int(config["GROUP_SIZE_M"]),
        BRIDGE_SLICES=int(bridge_slices) or geometry.num_slices,
        PLANES=int(planes),
        PAIR_HEADS=int(pair_heads) or 1,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


@triton.jit
def _per_row_lora_b_kernel(
    bridge_ptr,
    weight_ptr,
    destination_ptr,
    group_ids_ptr,
    token_slots_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    slice_offsets_ptr,
    out_offsets_ptr,
    # Keep the weight-stride pair 8-byte aligned for a single sm103 constant load.
    stride_wg,
    stride_wn,
    stride_wk,
    stride_bs,
    stride_bp,
    stride_bm,
    stride_bk,
    stride_dm,
    stride_dn,
    PAIR_BRIDGE: tl.constexpr,
    RANK: tl.constexpr,
    ADD_INPLACE: tl.constexpr,
    ZERO_SENTINEL: tl.constexpr,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    UNIFORM_WIDTH: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    FULL_TILES: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BRIDGE_SLICES: tl.constexpr,
    PLANES: tl.constexpr = 1,
):
    # A GEMV per (pair, column tile): sums in another order than the grouped
    # kernel (compare with allclose). The grid is exactly the row count, so
    # every program's row is in range. stride_bs > 0 = one bridge plane per
    # slot; PLANES > 1 = the split-K shrink's fp32 planes, summed here on load.
    # lora_ranks_ptr / scalings_ptr, PAIR_BRIDGE, BRIDGE_SLICES: see the
    # grouped kernel.
    pair_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    pair64 = pair_id.to(tl.int64)
    bucket_id = route_bucket_ids(
        group_ids_ptr,
        token_slots_ptr,
        pair_id,
        pair_id >= 0,
        GROUPS_PER_SLOT=GROUPS_PER_SLOT,
        MAX_LORAS=MAX_LORAS,
        WIDTH=WIDTH,
        HAS_GROUPS=HAS_GROUPS,
    )
    if not ZERO_SENTINEL:
        if bucket_id == -1:
            return
    safe_bucket_id = tl.maximum(bucket_id, 0).to(tl.int64)
    slot = safe_bucket_id // GROUPS_PER_SLOT
    if lora_ranks_ptr is not None:
        rank = tl.load(lora_ranks_ptr + slot).to(tl.int32)
        if rank == 0:
            return
    else:
        rank = RANK
    if UNIFORM_WIDTH > 0:
        tiles_per_slice: tl.constexpr = (
            UNIFORM_WIDTH + BLOCK_SIZE_N - 1
        ) // BLOCK_SIZE_N
        slice_id = pid_n // tiles_per_slice
        n_tile = pid_n % tiles_per_slice
        n_start = (slice_id * UNIFORM_WIDTH).to(tl.int64)
        width = UNIFORM_WIDTH
    else:
        slice_id, n_tile, n_start, width = _slice_of_tile(
            slice_offsets_ptr, pid_n, NUM_SLICES, BLOCK_SIZE_N
        )
        if slice_id < 0:
            return
    if OUT_STRIDE > 0:
        out_start = (slice_id * OUT_STRIDE).to(tl.int64)
    else:
        out_start = tl.load(out_offsets_ptr + slice_id).to(tl.int64)
    n_offsets = n_tile.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(
        tl.int64
    )
    if FULL_TILES:
        # Every slice is whole tiles wide: no column mask to carry.
        n_mask = tl.full((BLOCK_SIZE_N,), 1, tl.int1)
    else:
        n_mask = n_offsets < width
    destination_ptrs = (
        destination_ptr + pair64 * stride_dm + (out_start + n_offsets) * stride_dn
    )
    if ZERO_SENTINEL:
        if bucket_id == -1:
            # A fresh delta buffer must not keep stale graph memory.
            zeros = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            tl.store(
                destination_ptrs,
                zeros.to(destination_ptr.dtype.element_ty),
                mask=n_mask,
            )
            return

    if PAIR_BRIDGE:
        bridge_row = bridge_ptr + slot * stride_bs + pair64 * stride_bm
    else:
        bridge_row = bridge_ptr + slot * stride_bs + (pair64 // WIDTH) * stride_bm
    if ADD_INPLACE:
        prior = tl.load(destination_ptrs, mask=n_mask, other=0.0).to(tl.float32)
    if BRIDGE_SLICES < NUM_SLICES:
        bridge_col0 = (slice_id % BRIDGE_SLICES).to(tl.int64) * rank
    else:
        bridge_col0 = slice_id.to(tl.int64) * rank
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_begin in range(0, RANK, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < rank
        lhs_ptrs = bridge_row + (bridge_col0 + k_offsets) * stride_bk
        lhs = tl.load(lhs_ptrs, mask=k_mask, other=0.0)
        if PLANES > 1:
            lhs = lhs.to(tl.float32)
            for p in tl.static_range(1, PLANES):
                lhs += tl.load(lhs_ptrs + p * stride_bp, mask=k_mask, other=0.0).to(
                    tl.float32
                )
        rhs = tl.load(
            weight_ptr
            + safe_bucket_id * stride_wg
            + (n_start + n_offsets)[:, None] * stride_wn
            + k_offsets[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # Match grouped B's bridge cast before the FP32 multiply.
        lhs = lhs.to(weight_ptr.dtype.element_ty).to(tl.float32)
        accumulator += tl.sum(rhs.to(tl.float32) * lhs[None, :], axis=1)

    if scalings_ptr is not None:
        accumulator *= tl.load(scalings_ptr + slot).to(tl.float32)
    if ADD_INPLACE:
        accumulator += prior
    tl.store(
        destination_ptrs,
        accumulator.to(destination_ptr.dtype.element_ty),
        mask=n_mask,
    )


def per_row_lora_b(
    bridge: torch.Tensor,
    weight: torch.Tensor,
    destination: torch.Tensor,
    routing: RouteView,
    *,
    geometry: SliceGeometry,
    config: Mapping[str, int],
    add_inplace: bool,
    zero_sentinel: bool,
    pair_bridge: bool = True,
    lora_ranks: torch.Tensor | None = None,
    scalings: torch.Tensor | None = None,
    planes: int = 1,
    slot_planes: bool = False,
    bridge_slices: int = 0,
) -> None:
    """The grouped expand's contract over the raw route, one program per
    (pair, column tile), for a [rows, width] destination whose weight has one
    plane per bucket. ``slot_planes``: ``bridge`` is [slots, rows, rank], one
    plane per slot (the all_slots and token_dense shrinks); ``planes`` > 1: it
    is the split-K shrink's fp32 [planes, rows, rank], summed on load."""
    if geometry.block_size_n != config["BLOCK_SIZE_N"]:
        raise ValueError("expand BLOCK_SIZE_N must match its slice geometry")
    num_pairs = routing.num_rows
    if num_pairs == 0:
        return
    if (slot_planes and planes > 1) or bridge.dim() != 2 + (slot_planes or planes > 1):
        raise ValueError(
            "the per-row expand reads a [rows, rank] bridge, [slots, rows, rank] "
            "slot planes or [planes, rows, rank] fp32 split-K planes"
        )
    _per_row_lora_b_kernel[(num_pairs, geometry.num_column_tiles)](
        bridge,
        weight,
        destination,
        routing.kernel_groups,
        routing.token_slots,
        lora_ranks,
        scalings,
        geometry.slice_offsets,
        geometry.out_offsets,
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        bridge.stride(0) if slot_planes else 0,
        bridge.stride(0) if planes > 1 else 0,
        bridge.stride(-2),
        bridge.stride(-1),
        destination.stride(0),
        destination.stride(1),
        PAIR_BRIDGE=pair_bridge,
        RANK=weight.shape[2],
        ADD_INPLACE=add_inplace,
        ZERO_SENTINEL=zero_sentinel,
        GROUPS_PER_SLOT=routing.groups_per_slot,
        MAX_LORAS=routing.max_loras,
        WIDTH=routing.width,
        HAS_GROUPS=routing.group_ids is not None,
        NUM_SLICES=geometry.num_slices,
        UNIFORM_WIDTH=geometry.uniform_width,
        OUT_STRIDE=geometry.out_stride,
        FULL_TILES=geometry.full_tiles,
        BLOCK_SIZE_N=int(config["BLOCK_SIZE_N"]),
        BLOCK_SIZE_K=_k_tile(int(config["BLOCK_SIZE_K"]), weight.shape[2]),
        BRIDGE_SLICES=int(bridge_slices) or geometry.num_slices,
        PLANES=int(planes),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
