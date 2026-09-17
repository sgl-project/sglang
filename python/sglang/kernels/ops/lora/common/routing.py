"""Build and index (adapter slot, group) routes for dense and MoE LoRA.

Rows represent tokens, (token, head) pairs for MLA, or (token, expert) pairs
for MoE. Shared experts fold every live group into the slot's bucket.
RAW routes carry metadata only; ALIGNED routes sort and pad each bucket to
whole blocks. Capacity is rows + buckets * (block - 1), not request-dependent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.lora.common.route_view import RouteView, RouteViewKind
from sglang.kernels.ops.lora.moe.virtual_experts import (
    _align_block_size_jit,
)

if TYPE_CHECKING:
    from sglang.srt.lora.workspace import LoraWorkspace

# Route-builder dispatch limits; CUDA alignment caps at 8192 buckets.
_LARGE_ROUTE_MIN_BUCKETS = 8192
_LARGE_ROUTE_MIN_PAIRS = 16384
_SMALL_ROUTE_MAX_PAIRS = 512
_SMALL_ROUTE_LOW_BUCKET_MAX_BUCKETS = 4
_SMALL_ROUTE_LOW_BUCKET_MAX_PAIRS = 768

_HIST_BLOCK = 512
_HIST_WARPS = 8
_EXPAND_BLOCK = 128
_EXPAND_WARPS = 4
_SCAN_CHUNK = 2048
_SCAN_WARPS = 4

# Use in-block histograms to reduce global atomics within these limits.
_COUNT_MAX_BINS = 512
_COUNT_MIN_PAIRS = 16384
_CLAIM_MIN_PAIRS_PER_BUCKET = 12288


@triton.jit
def route_bucket_ids(
    group_ids_ptr,
    token_slots_ptr,
    pair_ids,
    pair_mask,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
):
    # Invalid slots/groups map to -1. With one group per slot, any nonnegative
    # group folds into that slot; otherwise bound it to the bucket table.
    token_ids = pair_ids // WIDTH
    slots = tl.load(token_slots_ptr + token_ids, mask=pair_mask, other=-1).to(tl.int32)
    live = (slots >= 0) & (slots < MAX_LORAS)
    if HAS_GROUPS:
        groups = tl.load(group_ids_ptr + pair_ids, mask=pair_mask, other=-1).to(
            tl.int32
        )
        live = live & (groups >= 0)
        if GROUPS_PER_SLOT == 1:
            buckets = slots
        else:
            live = live & (groups < GROUPS_PER_SLOT)
            buckets = slots * GROUPS_PER_SLOT + groups
    else:
        buckets = slots
    return tl.where(live, buckets, -1)


@triton.jit
def grouped_tile_coords(
    pid,
    num_pid_n,
    NUM_M_BLOCKS: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Group M blocks to reuse weight tiles in cache.
    programs_per_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _add_counts(
    counts_ptr,
    bucket_ids,
    mask,
    NUM_BUCKETS: tl.constexpr,
    BINS: tl.constexpr,
):
    buckets = tl.where(bucket_ids < 0, NUM_BUCKETS - 1, bucket_ids)
    if BINS == 0:
        tl.atomic_add(counts_ptr + buckets, 1, mask=mask)
    else:
        mine = tl.histogram(tl.where(mask, buckets, BINS - 1), BINS)
        bins = tl.arange(0, BINS)
        tl.atomic_add(counts_ptr + bins, mine, mask=(bins < NUM_BUCKETS) & (mine > 0))


@triton.jit
def _route_histogram_kernel(
    group_ids_ptr,
    token_slots_ptr,
    counts_ptr,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # The previous scan (or first allocation) cleared counts.
    pair_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    _add_counts(
        counts_ptr,
        route_bucket_ids(
            group_ids_ptr,
            token_slots_ptr,
            pair_ids,
            pair_mask,
            GROUPS_PER_SLOT=GROUPS_PER_SLOT,
            MAX_LORAS=MAX_LORAS,
            WIDTH=WIDTH,
            HAS_GROUPS=HAS_GROUPS,
        ),
        pair_mask,
        NUM_BUCKETS=NUM_BUCKETS,
        BINS=BINS,
    )
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _scan_one(
    counts_ptr,
    block_cumulative_ptr,
    cursor_ptr,
    bucket_end_ptr,
    padded_pairs_ptr,
    num_buckets,
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Clear counts during the scan for the next replay.
    running = 0
    for base in range(0, num_buckets, CHUNK):
        offsets = base + tl.arange(0, CHUNK)
        mask = offsets < num_buckets
        counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
        tl.store(counts_ptr + offsets, 0, mask=mask)
        blocks = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        block_start = running + tl.cumsum(blocks) - blocks
        tl.store(block_cumulative_ptr + offsets, block_start, mask=mask)
        slot_start = block_start * BLOCK_SIZE_M
        tl.store(cursor_ptr + offsets, slot_start, mask=mask)
        tl.store(bucket_end_ptr + offsets, slot_start + counts, mask=mask)
        running += tl.sum(blocks)
    tl.store(block_cumulative_ptr + num_buckets, running)
    tl.store(padded_pairs_ptr, running * BLOCK_SIZE_M)


@triton.jit
def _route_scan_kernel(
    counts_ptr,
    block_cumulative_ptr,
    cursor_ptr,
    bucket_end_ptr,
    padded_pairs_ptr,
    num_buckets,
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        # Both place paths wait before reading scan outputs.
        tl.extra.cuda.gdc_launch_dependents()
    _scan_one(
        counts_ptr,
        block_cumulative_ptr,
        cursor_ptr,
        bucket_end_ptr,
        padded_pairs_ptr,
        num_buckets,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        CHUNK=CHUNK,
    )


@triton.jit
def _label_blocks(
    pid,
    block_cumulative_ptr,
    bucket_end_ptr,
    sorted_pair_ids_ptr,
    block_bucket_ids_ptr,
    num_blocks,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    NUM_ROUTE_BUCKETS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    block_ids = pid * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_ids < num_blocks
    low = tl.zeros(block_ids.shape, dtype=tl.int32)
    high = tl.full(block_ids.shape, NUM_BUCKETS, dtype=tl.int32)
    for _ in range(SEARCH_STEPS):
        midpoint = (low + high) // 2
        bound = tl.load(
            block_cumulative_ptr + tl.minimum(midpoint + 1, NUM_BUCKETS),
            mask=block_mask,
            other=0,
        )
        take_upper = block_ids >= bound
        low = tl.where(take_upper & (low < high), midpoint + 1, low)
        high = tl.where(take_upper | (low >= high), high, midpoint)
    owner = tl.minimum(low, NUM_BUCKETS - 1)
    total_blocks = tl.load(block_cumulative_ptr + NUM_BUCKETS)
    in_plan = block_mask & (block_ids < total_blocks)
    tl.store(
        block_bucket_ids_ptr + block_ids,
        tl.where(in_plan & (owner < NUM_ROUTE_BUCKETS), owner, -1),
        mask=block_mask,
    )
    # B reads sentinel blocks too, so initialize their padding slots.
    real_end = tl.load(bucket_end_ptr + owner, mask=in_plan, other=0)
    slots = block_ids[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]
    tl.store(
        sorted_pair_ids_ptr + slots,
        num_pairs,
        mask=in_plan[:, None] & (slots >= real_end[:, None]),
    )


@triton.jit
def _claim_slots(
    cursor_ptr,
    bucket_ids,
    mask,
    NUM_BUCKETS: tl.constexpr,
    PER_BLOCK: tl.constexpr,
):
    # Per-block claims may reorder pairs within each bucket.
    buckets = tl.where(bucket_ids < 0, NUM_BUCKETS - 1, bucket_ids)
    if not PER_BLOCK:
        return tl.atomic_add(cursor_ptr + buckets, 1, mask=mask)
    slots = tl.zeros(buckets.shape, dtype=tl.int32)
    for bucket in tl.static_range(NUM_BUCKETS):
        mine = tl.where(mask & (buckets == bucket), 1, 0).to(tl.int32)
        start = tl.atomic_add(cursor_ptr + bucket, tl.sum(mine))
        slots = tl.where(mine == 1, start + tl.cumsum(mine) - mine, slots)
    return slots


@triton.jit
def _route_place_kernel(
    group_ids_ptr,
    token_slots_ptr,
    cursor_ptr,
    bucket_end_ptr,
    block_cumulative_ptr,
    sorted_ptr,
    block_bucket_ids_ptr,
    num_blocks,
    label_programs,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    NUM_ROUTE_BUCKETS: tl.constexpr,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    CLAIM_PER_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # Low program IDs label blocks; the rest place pairs.
    pid = tl.program_id(0)
    if pid < label_programs:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        _label_blocks(
            pid,
            block_cumulative_ptr,
            bucket_end_ptr,
            sorted_ptr,
            block_bucket_ids_ptr,
            num_blocks,
            num_pairs,
            NUM_BUCKETS=NUM_BUCKETS,
            NUM_ROUTE_BUCKETS=NUM_ROUTE_BUCKETS,
            BLOCK=BLOCK,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            SEARCH_STEPS=SEARCH_STEPS,
        )
        return

    # Recompute keys while the scan runs; this needs no scan output.
    pair_ids = (pid - label_programs) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    bucket_ids = route_bucket_ids(
        group_ids_ptr,
        token_slots_ptr,
        pair_ids,
        pair_mask,
        GROUPS_PER_SLOT=GROUPS_PER_SLOT,
        MAX_LORAS=MAX_LORAS,
        WIDTH=WIDTH,
        HAS_GROUPS=HAS_GROUPS,
    )
    if USE_PDL:
        # Cursors are this path's first scan dependency.
        tl.extra.cuda.gdc_wait()
    slots = _claim_slots(
        cursor_ptr,
        bucket_ids,
        pair_mask,
        NUM_BUCKETS=NUM_BUCKETS,
        PER_BLOCK=CLAIM_PER_BLOCK,
    )
    tl.store(sorted_ptr + slots, pair_ids, mask=pair_mask)


@triton.jit
def _build_route_bucket_ids_kernel(
    group_ids_ptr,
    token_slots_ptr,
    buckets_ptr,
    num_pairs,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair_ids < num_pairs
    buckets = route_bucket_ids(
        group_ids_ptr,
        token_slots_ptr,
        pair_ids,
        pair_mask,
        GROUPS_PER_SLOT=GROUPS_PER_SLOT,
        MAX_LORAS=MAX_LORAS,
        WIDTH=WIDTH,
        HAS_GROUPS=HAS_GROUPS,
    )
    tl.store(buckets_ptr + pair_ids, buckets, mask=pair_mask)


def _histogram_bins(num_buckets: int, num_pairs: int) -> int:
    # 0 = one atomic per pair; otherwise in-block counting over 2^k bins.
    if num_buckets >= _COUNT_MAX_BINS or num_pairs < _COUNT_MIN_PAIRS:
        return 0
    return 1 << num_buckets.bit_length()  # the extra bin holds the masked-off lanes


def _aligned_route_capacity(
    num_pairs: int,
    block_size: int,
    num_route_buckets: int,
) -> int:
    """Bound padded pairs, including the invalid-ID bucket."""
    if num_pairs == 0:
        return 0
    max_nonempty_buckets = min(num_pairs, num_route_buckets + 1)
    upper_bound = num_pairs + max_nonempty_buckets * (block_size - 1)
    return triton.cdiv(triton.cdiv(upper_bound, block_size) * block_size, 4) * 4


def _aligned_route_scratch(
    workspace: LoraWorkspace,
    *,
    prefix: str,
    num_buckets: int,
    capacity: int,
    block_size: int,
    device: torch.device,
) -> dict[str, object]:
    # The scan clears counts for the next replay.
    scratch: dict[str, object] = {
        "num_buckets": num_buckets,
        "capacity": capacity,
        "counts": workspace.tensor(
            f"{prefix}:counts",
            (num_buckets,),
            dtype=torch.int32,
            device=device,
            zero_on_first_allocation=True,
        ),
        "block_cumulative": workspace.tensor(
            f"{prefix}:block_cumulative",
            (num_buckets + 1,),
            dtype=torch.int32,
            device=device,
        ),
        "cursor": workspace.tensor(
            f"{prefix}:cursor", (num_buckets,), dtype=torch.int32, device=device
        ),
        "bucket_end": workspace.tensor(
            f"{prefix}:bucket_end", (num_buckets,), dtype=torch.int32, device=device
        ),
        "padded_pairs": workspace.tensor(
            f"{prefix}:padded_pairs", (1,), dtype=torch.int32, device=device
        ),
    }
    scratch["sorted"] = workspace.tensor(
        f"{prefix}:sorted", (capacity,), dtype=torch.int32, device=device
    )
    scratch["block_bucket_ids"] = workspace.tensor(
        f"{prefix}:block_bucket_ids",
        (capacity // block_size,),
        dtype=torch.int32,
        device=device,
    )
    return scratch


def _build_large_route(
    token_slots: torch.Tensor,
    group_ids: torch.Tensor | None,
    *,
    groups_per_slot: int,
    max_loras: int,
    block_size: int,
    workspace: LoraWorkspace,
    tensor_prefix: str,
    capacity: int | None = None,
) -> RouteView:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    groups = token_slots if group_ids is None else group_ids  # placeholder pointer
    num_pairs = groups.numel()
    width = 1 if group_ids is None else group_ids.shape[1]
    num_route_buckets = groups_per_slot * max_loras
    # Shared buffers may reserve more rows than this builder needs.
    capacity = max(
        _aligned_route_capacity(num_pairs, block_size, num_route_buckets), capacity or 0
    )
    capacity = triton.cdiv(capacity, block_size) * block_size
    if num_route_buckets + 1 >= 2**31 or capacity >= 2**31:
        raise ValueError(
            f"aligned routes use int32 plan math: {tensor_prefix} needs "
            f"{num_route_buckets + 1} buckets and {capacity} slots, both must be < 2**31"
        )
    # Scratch per bucket layout: one prefix may carry routes of different widths.
    own = _aligned_route_scratch(
        workspace,
        prefix=f"{tensor_prefix}:groups{groups_per_slot}",
        num_buckets=num_route_buckets + 1,
        capacity=capacity,
        block_size=block_size,
        device=token_slots.device,
    )
    num_buckets = own["num_buckets"]

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _route_histogram_kernel[(triton.cdiv(max(num_pairs, 1), _HIST_BLOCK),)](
        groups,
        token_slots,
        own["counts"],
        num_pairs,
        NUM_BUCKETS=num_buckets,
        GROUPS_PER_SLOT=groups_per_slot,
        MAX_LORAS=max_loras,
        WIDTH=width,
        HAS_GROUPS=group_ids is not None,
        BLOCK=_HIST_BLOCK,
        BINS=_histogram_bins(num_buckets, num_pairs),
        USE_PDL=use_pdl,
        num_warps=_HIST_WARPS,
    )
    _route_scan_kernel[(1,)](
        own["counts"],
        own["block_cumulative"],
        own["cursor"],
        own["bucket_end"],
        own["padded_pairs"],
        num_buckets,
        BLOCK_SIZE_M=block_size,
        CHUNK=_SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=_SCAN_WARPS,
        **pdl_kwargs,
    )
    num_blocks = own["capacity"] // block_size
    label_programs = triton.cdiv(max(num_blocks, 1), _EXPAND_BLOCK)
    _route_place_kernel[
        (label_programs + triton.cdiv(max(num_pairs, 1), _EXPAND_BLOCK),)
    ](
        groups,
        token_slots,
        own["cursor"],
        own["bucket_end"],
        own["block_cumulative"],
        own["sorted"],
        own["block_bucket_ids"],
        num_blocks,
        label_programs,
        num_pairs,
        NUM_BUCKETS=num_buckets,
        NUM_ROUTE_BUCKETS=num_buckets - 1,
        GROUPS_PER_SLOT=groups_per_slot,
        MAX_LORAS=max_loras,
        WIDTH=width,
        HAS_GROUPS=group_ids is not None,
        BLOCK=_EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        # Include the sentinel in the binary-search depth.
        SEARCH_STEPS=num_buckets.bit_length(),
        CLAIM_PER_BLOCK=num_pairs >= _CLAIM_MIN_PAIRS_PER_BUCKET * num_buckets,
        USE_PDL=use_pdl,
        num_warps=_EXPAND_WARPS,
        **pdl_kwargs,
    )
    return RouteView(
        view=RouteViewKind.ALIGNED,
        block_size=block_size,
        token_slots=token_slots,
        group_ids=group_ids,
        groups_per_slot=groups_per_slot,
        max_loras=max_loras,
        maybe_sorted_pair_ids=own["sorted"],
        maybe_block_bucket_ids=own["block_bucket_ids"],
        maybe_num_pairs_post_padded=own["padded_pairs"],
    )


def _build_route_bucket_ids(
    token_slots: torch.Tensor,
    group_ids: torch.Tensor | None,
    *,
    groups_per_slot: int,
    max_loras: int,
    out: torch.Tensor,
) -> torch.Tensor:
    groups = token_slots if group_ids is None else group_ids  # placeholder pointer
    num_pairs = out.numel()
    if num_pairs == 0:
        return out

    block_size = 1024
    _build_route_bucket_ids_kernel[(triton.cdiv(num_pairs, block_size),)](
        groups,
        token_slots,
        out,
        num_pairs,
        GROUPS_PER_SLOT=groups_per_slot,
        MAX_LORAS=max_loras,
        WIDTH=1 if group_ids is None else group_ids.shape[1],
        HAS_GROUPS=group_ids is not None,
        BLOCK_SIZE=block_size,
    )
    return out


# One launch builds a small aligned route. Atomic claims give arbitrary order
# within each bucket; like CUDA alignment, sentinel -1 comes first and counts
# toward the padded total, followed by live buckets in ascending order.


@triton.jit
def _build_small_route_kernel(
    group_ids_ptr,
    token_slots_ptr,
    sorted_pair_ids_ptr,
    block_bucket_ids_ptr,
    num_pairs_post_padded_ptr,
    scratch_ptr,  # int32 [2 * NUM_BINS]: bucket counters, then padded bucket starts
    num_pairs,
    num_route_buckets,
    capacity,
    num_blocks,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LANES: tl.constexpr,
    NUM_BINS: tl.constexpr,
    FILL: tl.constexpr,
):
    lane = tl.arange(0, LANES)
    pair_mask = lane < num_pairs
    bucket = route_bucket_ids(
        group_ids_ptr,
        token_slots_ptr,
        lane,
        pair_mask,
        GROUPS_PER_SLOT=GROUPS_PER_SLOT,
        MAX_LORAS=MAX_LORAS,
        WIDTH=WIDTH,
        HAS_GROUPS=HAS_GROUPS,
    )
    # Bin 0 is the sentinel bucket of the dead rows; bucket b is bin b + 1.
    bin_id = tl.where(bucket >= 0, bucket + 1, 0)
    # The counters are cleared here, so a graph replay is self-contained.
    for off in range(0, NUM_BINS, FILL):
        idx = off + tl.arange(0, FILL)
        tl.store(
            scratch_ptr + idx, tl.zeros([FILL], dtype=tl.int32), mask=idx < NUM_BINS
        )
    tl.debug_barrier()
    rank = tl.atomic_add(scratch_ptr + bin_id, 1, mask=pair_mask)
    tl.debug_barrier()
    bins = tl.arange(0, NUM_BINS)
    counts = tl.load(
        scratch_ptr + bins,
        mask=bins < num_route_buckets + 1,
        other=0,
        cache_modifier=".cg",
    )
    padded = (counts + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    starts = tl.cumsum(padded, axis=0) - padded
    tl.store(scratch_ptr + NUM_BINS + bins, starts)
    total = tl.sum(padded, axis=0)
    for off in range(0, capacity, FILL):
        idx = off + tl.arange(0, FILL)
        tl.store(
            sorted_pair_ids_ptr + idx,
            tl.zeros([FILL], dtype=tl.int32) + num_pairs,
            mask=idx < capacity,
        )
    for off in range(0, num_blocks, FILL):
        idx = off + tl.arange(0, FILL)
        tl.store(
            block_bucket_ids_ptr + idx,
            tl.zeros([FILL], dtype=tl.int32) - 1,
            mask=idx < num_blocks,
        )
    tl.debug_barrier()
    pos = tl.load(scratch_ptr + NUM_BINS + bin_id, cache_modifier=".cg") + rank
    tl.store(sorted_pair_ids_ptr + pos, lane, mask=pair_mask)
    tl.store(block_bucket_ids_ptr + pos // BLOCK_SIZE, bin_id - 1, mask=pair_mask)
    tl.store(num_pairs_post_padded_ptr, total)


def _build_small_route(
    token_slots: torch.Tensor,
    group_ids: torch.Tensor | None,
    *,
    groups_per_slot: int,
    max_loras: int,
    block_size: int,
    workspace: LoraWorkspace | None,
    tensor_prefix: str | None,
) -> RouteView:
    groups = token_slots if group_ids is None else group_ids  # placeholder pointer
    shape = (token_slots.numel(), 1) if group_ids is None else tuple(group_ids.shape)
    num_pairs = shape[0] * shape[1]
    num_route_buckets = groups_per_slot * max_loras
    # The CUDA builder's capacity, so consumers size their grids the same way.
    if num_pairs < num_route_buckets + 1:
        capacity = num_pairs * block_size
    else:
        capacity = num_pairs + (num_route_buckets + 1) * (block_size - 1)
    capacity = (capacity + 3) & ~3
    num_blocks = triton.cdiv(capacity, block_size)
    lanes = max(16, triton.next_power_of_2(num_pairs))
    num_bins = max(16, triton.next_power_of_2(num_route_buckets + 1))
    device = token_slots.device
    if workspace is not None:
        prefix = f"{tensor_prefix}:small"

        def alloc(name: str, numel: int) -> torch.Tensor:
            return workspace.tensor(
                f"{prefix}:{name}", (numel,), dtype=torch.int32, device=device
            )

    else:

        def alloc(name: str, numel: int) -> torch.Tensor:
            return torch.empty((numel,), dtype=torch.int32, device=device)

    sorted_pair_ids = alloc("sorted", capacity)
    block_bucket_ids = alloc("block_bucket_ids", num_blocks)
    num_pairs_post_padded = alloc("num_post", 1)
    scratch = alloc("scratch", 2 * num_bins)
    _build_small_route_kernel[(1,)](
        groups,
        token_slots,
        sorted_pair_ids,
        block_bucket_ids,
        num_pairs_post_padded,
        scratch,
        num_pairs,
        num_route_buckets,
        capacity,
        num_blocks,
        GROUPS_PER_SLOT=groups_per_slot,
        MAX_LORAS=max_loras,
        WIDTH=shape[1],
        HAS_GROUPS=group_ids is not None,
        BLOCK_SIZE=block_size,
        LANES=lanes,
        NUM_BINS=num_bins,
        FILL=1024,
        num_warps=8 if max(lanes, num_bins) >= 1024 else 4,
    )
    return RouteView(
        view=RouteViewKind.ALIGNED,
        block_size=block_size,
        token_slots=token_slots,
        group_ids=group_ids,
        groups_per_slot=groups_per_slot,
        max_loras=max_loras,
        maybe_sorted_pair_ids=sorted_pair_ids,
        maybe_block_bucket_ids=block_bucket_ids,
        maybe_num_pairs_post_padded=num_pairs_post_padded,
    )


def build_route(
    token_slots: torch.Tensor,
    *,
    group_ids: torch.Tensor | None = None,
    groups_per_slot: int = 1,
    max_loras: int,
    block_size: int,
    view: RouteViewKind | str = RouteViewKind.ALIGNED,
    workspace: LoraWorkspace | None = None,
    tensor_prefix: str | None = None,
    capacity: int | None = None,
) -> RouteView:
    """Build a RAW or ALIGNED route: one row per token, or per (token, column)
    of ``group_ids`` [tokens, width] when given.

    ``groups_per_slot`` buckets per adapter slot (1 folds every live row into
    the slot's bucket). For ALIGNED routes, ``capacity`` forces workspace-backed
    sorting and sets the minimum reserved rows, rounded up to a whole block.
    """
    try:
        view = RouteViewKind(view)
    except (TypeError, ValueError):
        raise ValueError(
            f"unknown route view {view!r}; expected one of "
            f"{tuple(kind.value for kind in RouteViewKind)}"
        ) from None
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if capacity is not None and (type(capacity) is not int or capacity < 0):
        raise ValueError("capacity must be a non-negative integer")
    if groups_per_slot > 1 and group_ids is None:
        raise ValueError("a route with more than one group per slot needs group_ids")
    common = {
        "view": view,
        "block_size": block_size,
        "token_slots": token_slots,
        "group_ids": group_ids,
        "groups_per_slot": groups_per_slot,
        "max_loras": max_loras,
    }
    if view is RouteViewKind.RAW:
        route = RouteView(**common)
        return route

    shape = (token_slots.numel(), 1) if group_ids is None else tuple(group_ids.shape)
    rows = shape[0] * shape[1]
    num_route_buckets = groups_per_slot * max_loras
    if (
        capacity is not None
        or num_route_buckets >= _LARGE_ROUTE_MIN_BUCKETS
        or rows >= _LARGE_ROUTE_MIN_PAIRS
    ):
        route = _build_large_route(
            token_slots,
            group_ids,
            groups_per_slot=groups_per_slot,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            tensor_prefix=tensor_prefix,
            capacity=capacity,
        )
        return route

    small_max_pairs = _SMALL_ROUTE_MAX_PAIRS
    if 0 < num_route_buckets <= _SMALL_ROUTE_LOW_BUCKET_MAX_BUCKETS:
        small_max_pairs = _SMALL_ROUTE_LOW_BUCKET_MAX_PAIRS
    if 0 < rows <= small_max_pairs:
        return _build_small_route(
            token_slots,
            group_ids,
            groups_per_slot=groups_per_slot,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            tensor_prefix=tensor_prefix,
        )

    # Use workspace storage to keep scratch out of the CUDA graph pool.
    scratch = None
    device = token_slots.device
    if workspace is not None:
        prefix = f"{tensor_prefix}:jit"
        buckets_out = workspace.tensor(
            f"{prefix}:bucket_ids", shape, dtype=torch.int32, device=device
        )
        scratch = lambda numel: workspace.tensor(  # noqa: E731
            f"{prefix}:scratch", (numel,), dtype=torch.int32, device=device
        )
    else:
        buckets_out = torch.empty(shape, dtype=torch.int32, device=device)
    buckets = _build_route_bucket_ids(
        token_slots,
        group_ids,
        groups_per_slot=groups_per_slot,
        max_loras=max_loras,
        out=buckets_out,
    )
    sorted_pair_ids, block_bucket_ids, num_pairs_post_padded = _align_block_size_jit(
        buckets, block_size, num_route_buckets, scratch=scratch
    )
    route = RouteView(
        **common,
        maybe_sorted_pair_ids=sorted_pair_ids,
        maybe_block_bucket_ids=block_bucket_ids,
        maybe_num_pairs_post_padded=num_pairs_post_padded,
    )
    return route
