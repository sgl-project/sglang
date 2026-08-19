"""Sort token/expert pairs into whole route blocks, one group per virtual expert.

One kernel counts the pairs per group, one scans those counts into block-aligned
runs, one labels each block with its group and scatters the pairs into slots. The
group key is never materialized: both kernels that need it recompute it, which is
the [T,K] round trip the JIT path pays along with a ladder that stops at 32767
groups. routing.py switches here above FUSED_ALIGN_MIN_PAIRS and
FUSED_ALIGN_MIN_VIRTUAL_EXPERTS.

A shared-factor plan wants the same pairs grouped two ways at once, by virtual
expert for its inner factors and by adapter alone for its shared outer ones, so
every kernel carries both groupings behind NEED_PER_EXPERT and NEED_SHARED and
runs whichever the caller asked for. Doing the two as separate builds would read
the pair sources twice and run two three-kernel chains.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.routing import (
    RouteView,
    RouteViewKind,
    virtual_expert_ids_inline,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# Launch tiles; see configs/README.md before changing them.
HIST_BLOCK = 512
HIST_WARPS = 8
EXPAND_BLOCK = 128
EXPAND_WARPS = 4
SCAN_CHUNK = 2048
SCAN_WARPS = 4

# Bin ceiling and smallest pair counts at which counting a block's pairs beats
# one atomic per pair; outside them the helpers below keep the per-pair path.
COUNT_MAX_BINS = 512
COUNT_MIN_PAIRS = 16384
CLAIM_MIN_PAIRS_PER_BUCKET = 12288


def count_bins(num_buckets: int, num_pairs: int) -> int:
    """Bins for counting inside a block, or 0 to add one pair at a time."""
    if num_buckets >= COUNT_MAX_BINS or num_pairs < COUNT_MIN_PAIRS:
        return 0
    return 1 << num_buckets.bit_length()  # one spare bin, for masked-off lanes


def _routing_capacity(
    num_pairs: int,
    block_size: int,
    num_virtual_experts: int,
) -> int:
    if num_pairs == 0:
        return 0
    max_nonempty_buckets = min(num_pairs, num_virtual_experts + 1)
    upper_bound = num_pairs + max_nonempty_buckets * (block_size - 1)
    return triton.cdiv(triton.cdiv(upper_bound, block_size) * block_size, 4) * 4


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


def _plan_scratch(
    workspace: MoeLoraWorkspace,
    *,
    prefix: str,
    num_buckets: int,
    capacity: int,
    block_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Route-owned scratch; counts are zeroed once because the scan restores that."""
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
    scratch["block_ids"] = workspace.tensor(
        f"{prefix}:block_ids",
        (capacity // block_size,),
        dtype=torch.int32,
        device=device,
    )
    return scratch


def _run(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    per_expert: dict[str, object],
    shared: dict[str, object],
    need_per_expert: bool,
    need_shared: bool,
) -> None:
    """Launch count, scan and place over whichever groupings were asked for."""
    from sglang.kernels.jit.utils import is_arch_support_pdl

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    pe_buckets = per_expert["num_buckets"]
    sh_buckets = shared["num_buckets"]
    shape = dict(
        NEED_PER_EXPERT=need_per_expert,
        NEED_SHARED=need_shared,
        NUM_PER_EXPERT_BUCKETS=pe_buckets,
        NUM_SHARED_BUCKETS=sh_buckets,
        E_LOCAL=num_local_experts,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        USE_PDL=use_pdl,
    )

    _hist_kernel[(triton.cdiv(max(num_pairs, 1), HIST_BLOCK),)](
        topk_ids,
        token_lora_mapping,
        per_expert["counts"],
        shared["counts"],
        num_pairs,
        num_local_experts,
        BLOCK=HIST_BLOCK,
        PER_EXPERT_BINS=count_bins(pe_buckets, num_pairs),
        SHARED_BINS=count_bins(sh_buckets, num_pairs),
        num_warps=HIST_WARPS,
        **shape,
    )
    _scan_kernel[(int(need_per_expert) + int(need_shared),)](
        per_expert["counts"],
        per_expert["block_cumulative"],
        per_expert["cursor"],
        per_expert["bucket_end"],
        per_expert["padded_pairs"],
        pe_buckets,
        shared["counts"],
        shared["block_cumulative"],
        shared["cursor"],
        shared["bucket_end"],
        shared["padded_pairs"],
        sh_buckets,
        NEED_PER_EXPERT=need_per_expert,
        NEED_SHARED=need_shared,
        BLOCK_SIZE_M=block_size,
        CHUNK=SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=SCAN_WARPS,
        **pdl_kwargs,
    )
    pe_blocks = per_expert["capacity"] // block_size
    sh_blocks = shared["capacity"] // block_size
    pe_labels = triton.cdiv(max(pe_blocks, 1), EXPAND_BLOCK) if need_per_expert else 0
    sh_labels = triton.cdiv(max(sh_blocks, 1), EXPAND_BLOCK) if need_shared else 0
    pair_programs = triton.cdiv(max(num_pairs, 1), EXPAND_BLOCK)
    _place_kernel[(pe_labels + sh_labels + pair_programs,)](
        topk_ids,
        token_lora_mapping,
        per_expert["cursor"],
        per_expert["bucket_end"],
        per_expert["block_cumulative"],
        per_expert["sorted"],
        per_expert["block_ids"],
        pe_blocks,
        pe_labels,
        shared["cursor"],
        shared["bucket_end"],
        shared["block_cumulative"],
        shared["sorted"],
        shared["block_ids"],
        sh_blocks,
        sh_labels,
        num_pairs,
        num_local_experts,
        NUM_PER_EXPERT_VIRTUAL=pe_buckets - 1,
        NUM_SHARED_VIRTUAL=sh_buckets - 1,
        BLOCK=EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        # The search picks one of NUM_BUCKETS + 1 answers, so it needs
        # num_buckets.bit_length() steps -- one fewer and a sentinel reads as 0.
        PER_EXPERT_SEARCH_STEPS=pe_buckets.bit_length(),
        SHARED_SEARCH_STEPS=sh_buckets.bit_length(),
        PER_EXPERT_CLAIM_PER_BLOCK=num_pairs >= CLAIM_MIN_PAIRS_PER_BUCKET * pe_buckets,
        SHARED_CLAIM_PER_BLOCK=num_pairs >= CLAIM_MIN_PAIRS_PER_BUCKET * sh_buckets,
        num_warps=EXPAND_WARPS,
        **pdl_kwargs,
        **shape,
    )


def build(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: MoeLoraWorkspace,
    tensor_prefix: str,
    need_per_expert: bool,
    need_shared: bool,
) -> tuple[RouteView | None, RouteView | None]:
    """Return the per-expert and shared-outer views; either is None if unasked."""
    if topk_ids.ndim != 2 or token_lora_mapping.shape != (topk_ids.shape[0],):
        raise ValueError("expected topk_ids [T,K] and token_lora_mapping [T]")
    if num_local_experts < 1 or max_loras < 1 or block_size < 1:
        raise ValueError("expert, adapter, and route block counts must be positive")
    num_pairs = topk_ids.numel()
    scratch: dict[str, dict[str, object]] = {}
    for name, virtual in (
        ("per_expert", num_local_experts * max_loras),
        ("shared", max_loras),
    ):
        capacity = _routing_capacity(num_pairs, block_size, virtual)
        if virtual + 1 >= 2**31 or capacity >= 2**31:
            raise ValueError(
                f"aligned routes use int32 plan math: {name} needs {virtual + 1} "
                f"buckets and {capacity} slots, both must be < 2**31"
            )
        scratch[name] = _plan_scratch(
            workspace,
            prefix=f"{tensor_prefix}:{name}",
            num_buckets=virtual + 1,
            capacity=capacity,
            block_size=block_size,
            device=topk_ids.device,
        )
    _run(
        topk_ids,
        token_lora_mapping,
        num_local_experts=num_local_experts,
        max_loras=max_loras,
        block_size=block_size,
        per_expert=scratch["per_expert"],
        shared=scratch["shared"],
        need_per_expert=need_per_expert,
        need_shared=need_shared,
    )

    def route(name: str, *, is_shared_outer: bool) -> RouteView:
        own = scratch[name]
        return RouteView(
            view=RouteViewKind.ALIGNED,
            block_size=block_size,
            topk_ids=topk_ids,
            token_lora_mapping=token_lora_mapping,
            num_local_experts=num_local_experts,
            is_shared_outer=is_shared_outer,
            max_loras=max_loras,
            maybe_sorted_pair_ids=own["sorted"],
            maybe_block_virtual_expert_ids=own["block_ids"],
            maybe_num_pairs_post_padded=own["padded_pairs"],
        )

    return (
        route("per_expert", is_shared_outer=False) if need_per_expert else None,
        route("shared", is_shared_outer=True) if need_shared else None,
    )
