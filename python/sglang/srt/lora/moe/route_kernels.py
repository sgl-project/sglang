"""Triton bodies behind the route builders; their hosts live in routing.py.

Kept apart so they move to the kernels folder as one unit, and so the host layer
can import the plan layer without dragging triton in behind it.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def virtual_expert_ids_inline(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    pair_ids,
    pair_mask,
    routed_expert_id_bound,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
):
    token_ids = pair_ids // TOP_K
    adapter_ids = tl.load(
        token_lora_mapping_ptr + token_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int32)
    routed_expert_ids = tl.load(
        topk_ids_ptr + pair_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int32)
    if SHARED_OUTER:
        in_range = (routed_expert_ids >= 0) & (
            routed_expert_ids < routed_expert_id_bound
        )
        lora_expert_ids = tl.where(in_range, 0, -1)
    else:
        lora_expert_ids = routed_expert_ids

    valid = (
        (adapter_ids >= 0)
        & (adapter_ids < MAX_LORAS)
        & (lora_expert_ids >= 0)
        & (lora_expert_ids < LORA_EXPERTS_PER_ADAPTER)
    )
    return tl.where(valid, adapter_ids * LORA_EXPERTS_PER_ADAPTER + lora_expert_ids, -1)


@triton.jit
def add_counts_inline(
    counts_ptr,
    virtual_ids,
    mask,
    NUM_BUCKETS: tl.constexpr,
    BINS: tl.constexpr,
):
    """Count these pairs, one atomic per bucket when BINS is nonzero."""
    buckets = tl.where(virtual_ids < 0, NUM_BUCKETS - 1, virtual_ids)
    if BINS == 0:
        tl.atomic_add(counts_ptr + buckets, 1, mask=mask)
    else:
        mine = tl.histogram(tl.where(mask, buckets, BINS - 1), BINS)
        bins = tl.arange(0, BINS)
        tl.atomic_add(counts_ptr + bins, mine, mask=(bins < NUM_BUCKETS) & (mine > 0))


@triton.jit
def claim_slots_inline(
    cursor_ptr,
    virtual_ids,
    mask,
    NUM_BUCKETS: tl.constexpr,
    PER_BLOCK: tl.constexpr,
):
    """A slot per pair; PER_BLOCK claims each bucket's run at once, reordering it."""
    buckets = tl.where(virtual_ids < 0, NUM_BUCKETS - 1, virtual_ids)
    if not PER_BLOCK:
        return tl.atomic_add(cursor_ptr + buckets, 1, mask=mask)
    slots = tl.zeros(buckets.shape, dtype=tl.int32)
    for bucket in tl.static_range(NUM_BUCKETS):
        mine = tl.where(mask & (buckets == bucket), 1, 0).to(tl.int32)
        start = tl.atomic_add(cursor_ptr + bucket, tl.sum(mine))
        slots = tl.where(mine == 1, start + tl.cumsum(mine) - mine, slots)
    return slots


@triton.jit
def _build_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    virtual_topk_ids_ptr,
    num_pairs,
    routed_expert_id_bound,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        pair_ids,
        pair_mask,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        SHARED_OUTER=SHARED_OUTER,
    )
    tl.store(virtual_topk_ids_ptr + pair_ids, virtual_ids, mask=pair_mask)


@triton.jit
def _hist_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    per_expert_counts_ptr,
    shared_counts_ptr,
    num_pairs,
    num_local_experts,
    NEED_PER_EXPERT: tl.constexpr,
    NEED_SHARED: tl.constexpr,
    NUM_PER_EXPERT_BUCKETS: tl.constexpr,
    NUM_SHARED_BUCKETS: tl.constexpr,
    E_LOCAL: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    PER_EXPERT_BINS: tl.constexpr,
    SHARED_BINS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Count each group's pairs; counts arrive zeroed, so there is no memset."""
    pair_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    if NEED_PER_EXPERT:
        add_counts_inline(
            per_expert_counts_ptr,
            virtual_expert_ids_inline(
                topk_ids_ptr,
                token_lora_mapping_ptr,
                pair_ids,
                pair_mask,
                0,
                LORA_EXPERTS_PER_ADAPTER=E_LOCAL,
                MAX_LORAS=MAX_LORAS,
                TOP_K=TOP_K,
                SHARED_OUTER=False,
            ),
            pair_mask,
            NUM_BUCKETS=NUM_PER_EXPERT_BUCKETS,
            BINS=PER_EXPERT_BINS,
        )
    if NEED_SHARED:
        add_counts_inline(
            shared_counts_ptr,
            virtual_expert_ids_inline(
                topk_ids_ptr,
                token_lora_mapping_ptr,
                pair_ids,
                pair_mask,
                num_local_experts,
                LORA_EXPERTS_PER_ADAPTER=1,
                MAX_LORAS=MAX_LORAS,
                TOP_K=TOP_K,
                SHARED_OUTER=True,
            ),
            pair_mask,
            NUM_BUCKETS=NUM_SHARED_BUCKETS,
            BINS=SHARED_BINS,
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
    """Exclusive scan of block-padded bucket sizes; re-zeroes counts as it reads."""
    running = 0
    for base in range(0, num_buckets, CHUNK):
        offsets = base + tl.arange(0, CHUNK)
        mask = offsets < num_buckets
        counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
        # This restores the zero-count invariant for the next replay.
        tl.store(counts_ptr + offsets, 0, mask=mask)
        blocks = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        block_start = running + tl.cumsum(blocks) - blocks
        tl.store(block_cumulative_ptr + offsets, block_start, mask=mask)
        slot_start = block_start * BLOCK_SIZE_M
        tl.store(cursor_ptr + offsets, slot_start, mask=mask)
        # bucket_end must be materialized here; counts are gone by the fill.
        tl.store(bucket_end_ptr + offsets, slot_start + counts, mask=mask)
        running += tl.sum(blocks)
    tl.store(block_cumulative_ptr + num_buckets, running)
    tl.store(padded_pairs_ptr, running * BLOCK_SIZE_M)


@triton.jit
def _scan_kernel(
    per_expert_counts_ptr,
    per_expert_block_cumulative_ptr,
    per_expert_cursor_ptr,
    per_expert_bucket_end_ptr,
    per_expert_padded_pairs_ptr,
    num_per_expert_buckets,
    shared_counts_ptr,
    shared_block_cumulative_ptr,
    shared_cursor_ptr,
    shared_bucket_end_ptr,
    shared_padded_pairs_ptr,
    num_shared_buckets,
    NEED_PER_EXPERT: tl.constexpr,
    NEED_SHARED: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Scan every requested group, one program each."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        # The place kernel can launch now: its pair path recomputes keys with no
        # scan output, then waits before its first cursor load, and its label
        # paths wait before their first access.
        tl.extra.cuda.gdc_launch_dependents()
    scan_per_expert = (
        tl.program_id(0) == 0 if NEED_PER_EXPERT and NEED_SHARED else NEED_PER_EXPERT
    )
    if scan_per_expert:
        _scan_one(
            per_expert_counts_ptr,
            per_expert_block_cumulative_ptr,
            per_expert_cursor_ptr,
            per_expert_bucket_end_ptr,
            per_expert_padded_pairs_ptr,
            num_per_expert_buckets,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            CHUNK=CHUNK,
        )
    else:
        _scan_one(
            shared_counts_ptr,
            shared_block_cumulative_ptr,
            shared_cursor_ptr,
            shared_bucket_end_ptr,
            shared_padded_pairs_ptr,
            num_shared_buckets,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            CHUNK=CHUNK,
        )


@triton.jit
def _label_blocks(
    pid,
    block_cumulative_ptr,
    bucket_end_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_blocks,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    NUM_VIRTUAL_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Name each block's owning group and fill its padding tail."""
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
        block_virtual_expert_ids_ptr + block_ids,
        tl.where(in_plan & (owner < NUM_VIRTUAL_EXPERTS), owner, -1),
        mask=block_mask,
    )
    # Coalesced 2D fill of every in-plan block's padding tail, sentinel bucket
    # included (a -1 block's slots ARE read by the aligned B kernels).
    real_end = tl.load(bucket_end_ptr + owner, mask=in_plan, other=0)
    slots = block_ids[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]
    tl.store(
        sorted_pair_ids_ptr + slots,
        num_pairs,
        mask=in_plan[:, None] & (slots >= real_end[:, None]),
    )


@triton.jit
def _place_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    per_expert_cursor_ptr,
    per_expert_bucket_end_ptr,
    per_expert_block_cumulative_ptr,
    per_expert_sorted_ptr,
    per_expert_block_ids_ptr,
    num_per_expert_blocks,
    per_expert_label_programs,
    shared_cursor_ptr,
    shared_bucket_end_ptr,
    shared_block_cumulative_ptr,
    shared_sorted_ptr,
    shared_block_ids_ptr,
    num_shared_blocks,
    shared_label_programs,
    num_pairs,
    num_local_experts,
    NEED_PER_EXPERT: tl.constexpr,
    NEED_SHARED: tl.constexpr,
    NUM_PER_EXPERT_BUCKETS: tl.constexpr,
    NUM_PER_EXPERT_VIRTUAL: tl.constexpr,
    NUM_SHARED_BUCKETS: tl.constexpr,
    NUM_SHARED_VIRTUAL: tl.constexpr,
    E_LOCAL: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    PER_EXPERT_SEARCH_STEPS: tl.constexpr,
    SHARED_SEARCH_STEPS: tl.constexpr,
    PER_EXPERT_CLAIM_PER_BLOCK: tl.constexpr,
    SHARED_CLAIM_PER_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Every scan consumer on one grid, split by program id: labels, then pairs."""
    pid = tl.program_id(0)
    if NEED_PER_EXPERT:
        if pid < per_expert_label_programs:
            if USE_PDL:
                # The label path immediately consumes scan outputs.
                tl.extra.cuda.gdc_wait()
            _label_blocks(
                pid,
                per_expert_block_cumulative_ptr,
                per_expert_bucket_end_ptr,
                per_expert_sorted_ptr,
                per_expert_block_ids_ptr,
                num_per_expert_blocks,
                num_pairs,
                NUM_BUCKETS=NUM_PER_EXPERT_BUCKETS,
                NUM_VIRTUAL_EXPERTS=NUM_PER_EXPERT_VIRTUAL,
                BLOCK=BLOCK,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                SEARCH_STEPS=PER_EXPERT_SEARCH_STEPS,
            )
            return
    if NEED_SHARED:
        if pid < per_expert_label_programs + shared_label_programs:
            if USE_PDL:
                tl.extra.cuda.gdc_wait()
            _label_blocks(
                pid - per_expert_label_programs,
                shared_block_cumulative_ptr,
                shared_bucket_end_ptr,
                shared_sorted_ptr,
                shared_block_ids_ptr,
                num_shared_blocks,
                num_pairs,
                NUM_BUCKETS=NUM_SHARED_BUCKETS,
                NUM_VIRTUAL_EXPERTS=NUM_SHARED_VIRTUAL,
                BLOCK=BLOCK,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                SEARCH_STEPS=SHARED_SEARCH_STEPS,
            )
            return

    # Recomputing the key beats a [T, K] round trip, and needs no scan -- so it
    # runs before the wait below.
    pair_ids = (
        pid - per_expert_label_programs - shared_label_programs
    ) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    if NEED_PER_EXPERT:
        per_expert_ids = virtual_expert_ids_inline(
            topk_ids_ptr,
            token_lora_mapping_ptr,
            pair_ids,
            pair_mask,
            0,
            LORA_EXPERTS_PER_ADAPTER=E_LOCAL,
            MAX_LORAS=MAX_LORAS,
            TOP_K=TOP_K,
            SHARED_OUTER=False,
        )
    if NEED_SHARED:
        shared_ids = virtual_expert_ids_inline(
            topk_ids_ptr,
            token_lora_mapping_ptr,
            pair_ids,
            pair_mask,
            num_local_experts,
            LORA_EXPERTS_PER_ADAPTER=1,
            MAX_LORAS=MAX_LORAS,
            TOP_K=TOP_K,
            SHARED_OUTER=True,
        )
    if USE_PDL:
        # Cursors below are the first scan-produced values this path consumes.
        tl.extra.cuda.gdc_wait()
    if NEED_PER_EXPERT:
        per_expert_slots = claim_slots_inline(
            per_expert_cursor_ptr,
            per_expert_ids,
            pair_mask,
            NUM_BUCKETS=NUM_PER_EXPERT_BUCKETS,
            PER_BLOCK=PER_EXPERT_CLAIM_PER_BLOCK,
        )
        tl.store(per_expert_sorted_ptr + per_expert_slots, pair_ids, mask=pair_mask)
    if NEED_SHARED:
        shared_slots = claim_slots_inline(
            shared_cursor_ptr,
            shared_ids,
            pair_mask,
            NUM_BUCKETS=NUM_SHARED_BUCKETS,
            PER_BLOCK=SHARED_CLAIM_PER_BLOCK,
        )
        tl.store(shared_sorted_ptr + shared_slots, pair_ids, mask=pair_mask)
