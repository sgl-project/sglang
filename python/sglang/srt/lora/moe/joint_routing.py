"""Joint aligned-route construction for shared-factor MoE-LoRA plans.

Shared-factor execution consumes two aligned views over the same token/expert
pairs: a per-expert view for the inner factors and a per-adapter view for the
shared outer factors.  This implementation reads the pair sources once and
builds both views in one three-launch chain.  It is an explicit candidate;
the execution plan, never this module, decides whether it is used.

The kernel bodies and three-stage protocol are the qualified R10 candidate
from ``benchmark/kernels/lora_moe/r10_joint_route.py``.  The production port's
only structural change is replacing that benchmark's process-global scratch
cache with the layer-owned :class:`MoeLoraWorkspace`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.routing import (
    ROUTE_ALIGNED,
    RouteView,
    _routing_capacity,
    virtual_expert_ids_inline,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

_HIST_BLOCK = 512
_EXPAND_BLOCK = 128
_SCAN_CHUNK = 2048


@triton.jit
def _joint_hist_kernel(
    topk_ids_ptr,
    token_slots_ptr,
    per_expert_counts_ptr,
    shared_counts_ptr,
    num_pairs,
    num_local_experts,
    NUM_PER_EXPERT_BUCKETS: tl.constexpr,
    E_LOCAL: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_SHARED_BUCKETS: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    per_expert_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        0,
        LORA_EXPERTS_PER_ADAPTER=E_LOCAL,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=False,
    )
    shared_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        num_local_experts,
        LORA_EXPERTS_PER_ADAPTER=1,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=True,
    )
    tl.atomic_add(
        per_expert_counts_ptr
        + tl.where(
            per_expert_ids < 0,
            NUM_PER_EXPERT_BUCKETS - 1,
            per_expert_ids,
        ),
        1,
        mask=pair_mask,
    )
    tl.atomic_add(
        shared_counts_ptr
        + tl.where(shared_ids < 0, NUM_SHARED_BUCKETS - 1, shared_ids),
        1,
        mask=pair_mask,
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
        tl.store(bucket_end_ptr + offsets, slot_start + counts, mask=mask)
        running += tl.sum(blocks)
    tl.store(block_cumulative_ptr + num_buckets, running)
    tl.store(padded_pairs_ptr, running * BLOCK_SIZE_M)


@triton.jit
def _dual_scan_kernel(
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
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        # Both scan CTAs consume histogram output immediately.
        tl.extra.cuda.gdc_wait()
        # The expand/scatter kernel can launch now. Its pair path recomputes
        # virtual keys without scan output, then waits immediately before the
        # first cursor load; its label paths wait before their first access.
        tl.extra.cuda.gdc_launch_dependents()
    if tl.program_id(0) == 0:
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
    real_end = tl.load(bucket_end_ptr + owner, mask=in_plan, other=0)
    slots = block_ids[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]
    tl.store(
        sorted_pair_ids_ptr + slots,
        num_pairs,
        mask=in_plan[:, None] & (slots >= real_end[:, None]),
    )


@triton.jit
def _joint_expand_scatter_kernel(
    topk_ids_ptr,
    token_slots_ptr,
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
    NUM_PER_EXPERT_BUCKETS: tl.constexpr,
    NUM_PER_EXPERT_VIRTUAL: tl.constexpr,
    E_LOCAL: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_SHARED_BUCKETS: tl.constexpr,
    NUM_SHARED_VIRTUAL: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    PER_EXPERT_SEARCH_STEPS: tl.constexpr,
    SHARED_SEARCH_STEPS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
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
    if pid < per_expert_label_programs + shared_label_programs:
        if USE_PDL:
            # The label path immediately consumes scan outputs.
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

    pair_ids = (
        pid - per_expert_label_programs - shared_label_programs
    ) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    per_expert_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        0,
        LORA_EXPERTS_PER_ADAPTER=E_LOCAL,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=False,
    )
    shared_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        num_local_experts,
        LORA_EXPERTS_PER_ADAPTER=1,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=True,
    )
    per_expert_buckets = tl.where(
        per_expert_ids < 0,
        NUM_PER_EXPERT_BUCKETS - 1,
        per_expert_ids,
    )
    shared_buckets = tl.where(shared_ids < 0, NUM_SHARED_BUCKETS - 1, shared_ids)
    if USE_PDL:
        # Key recomputation above is independent of the scan. Cursors below
        # are the first scan-produced values this path consumes.
        tl.extra.cuda.gdc_wait()
    per_expert_slots = tl.atomic_add(
        per_expert_cursor_ptr + per_expert_buckets, 1, mask=pair_mask
    )
    tl.store(
        per_expert_sorted_ptr + per_expert_slots,
        pair_ids,
        mask=pair_mask,
    )
    shared_slots = tl.atomic_add(shared_cursor_ptr + shared_buckets, 1, mask=pair_mask)
    tl.store(shared_sorted_ptr + shared_slots, pair_ids, mask=pair_mask)


def _plan_scratch(
    workspace: MoeLoraWorkspace,
    *,
    prefix: str,
    num_buckets: int,
    capacity: int,
    block_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return route-owned scratch with a self-restoring count invariant.

    ``_dual_scan_kernel`` clears each count immediately after its final read,
    so counts need initialization only when their workspace storage is first
    allocated.  Re-zeroing them on every lookup would add two redundant
    memset launches to every joint-route construction.
    """
    return {
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
            f"{prefix}:cursor",
            (num_buckets,),
            dtype=torch.int32,
            device=device,
        ),
        "bucket_end": workspace.tensor(
            f"{prefix}:bucket_end",
            (num_buckets,),
            dtype=torch.int32,
            device=device,
        ),
        "padded_pairs": workspace.tensor(
            f"{prefix}:padded_pairs",
            (1,),
            dtype=torch.int32,
            device=device,
        ),
        "sorted": workspace.tensor(
            f"{prefix}:sorted",
            (capacity,),
            dtype=torch.int32,
            device=device,
        ),
        "block_ids": workspace.tensor(
            f"{prefix}:block_ids",
            (capacity // block_size,),
            dtype=torch.int32,
            device=device,
        ),
    }


def build_joint_shared_routes(
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: MoeLoraWorkspace,
    use_pdl: bool | None = None,
) -> tuple[RouteView, RouteView]:
    """Return ``(per_expert, shared_outer)`` aligned views from one pair pass.

    ``use_pdl=None`` selects PDL on architectures that support it. The three
    kernels form a real histogram -> scan -> scatter dependency chain; the
    consumer launches carry ``launch_pdl=True`` and wait only immediately
    before their first predecessor-produced load.
    """
    if topk_ids.ndim != 2 or token_slots.shape != (topk_ids.shape[0],):
        raise ValueError("joint routing expects topk_ids [T,K] and token_slots [T]")
    if num_local_experts < 1 or max_loras < 1 or block_size < 1:
        raise ValueError("expert, adapter, and route block counts must be positive")
    if use_pdl is None:
        from sglang.kernels.jit.utils import is_arch_support_pdl

        use_pdl = is_arch_support_pdl()
    else:
        use_pdl = bool(use_pdl)
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    device = topk_ids.device
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    num_per_expert_virtual = num_local_experts * max_loras
    num_shared_virtual = max_loras
    num_per_expert_buckets = num_per_expert_virtual + 1
    num_shared_buckets = num_shared_virtual + 1
    per_expert_capacity = _routing_capacity(
        num_pairs, block_size, num_per_expert_virtual
    )
    shared_capacity = _routing_capacity(num_pairs, block_size, num_shared_virtual)
    per_expert = _plan_scratch(
        workspace,
        prefix="joint_route:per_expert",
        num_buckets=num_per_expert_buckets,
        capacity=per_expert_capacity,
        block_size=block_size,
        device=device,
    )
    shared = _plan_scratch(
        workspace,
        prefix="joint_route:shared",
        num_buckets=num_shared_buckets,
        capacity=shared_capacity,
        block_size=block_size,
        device=device,
    )

    _joint_hist_kernel[(triton.cdiv(max(num_pairs, 1), _HIST_BLOCK),)](
        topk_ids,
        token_slots,
        per_expert["counts"],
        shared["counts"],
        num_pairs,
        num_local_experts,
        NUM_PER_EXPERT_BUCKETS=num_per_expert_buckets,
        E_LOCAL=num_local_experts,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        NUM_SHARED_BUCKETS=num_shared_buckets,
        BLOCK=_HIST_BLOCK,
        USE_PDL=use_pdl,
        num_warps=8,
    )
    _dual_scan_kernel[(2,)](
        per_expert["counts"],
        per_expert["block_cumulative"],
        per_expert["cursor"],
        per_expert["bucket_end"],
        per_expert["padded_pairs"],
        num_per_expert_buckets,
        shared["counts"],
        shared["block_cumulative"],
        shared["cursor"],
        shared["bucket_end"],
        shared["padded_pairs"],
        num_shared_buckets,
        BLOCK_SIZE_M=block_size,
        CHUNK=_SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=4,
        **pdl_kwargs,
    )
    per_expert_label_programs = triton.cdiv(
        max(per_expert_capacity // block_size, 1), _EXPAND_BLOCK
    )
    shared_label_programs = triton.cdiv(
        max(shared_capacity // block_size, 1), _EXPAND_BLOCK
    )
    pair_programs = triton.cdiv(max(num_pairs, 1), _EXPAND_BLOCK)
    _joint_expand_scatter_kernel[
        (per_expert_label_programs + shared_label_programs + pair_programs,)
    ](
        topk_ids,
        token_slots,
        per_expert["cursor"],
        per_expert["bucket_end"],
        per_expert["block_cumulative"],
        per_expert["sorted"],
        per_expert["block_ids"],
        per_expert_capacity // block_size,
        per_expert_label_programs,
        shared["cursor"],
        shared["bucket_end"],
        shared["block_cumulative"],
        shared["sorted"],
        shared["block_ids"],
        shared_capacity // block_size,
        shared_label_programs,
        num_pairs,
        num_local_experts,
        NUM_PER_EXPERT_BUCKETS=num_per_expert_buckets,
        NUM_PER_EXPERT_VIRTUAL=num_per_expert_virtual,
        E_LOCAL=num_local_experts,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        NUM_SHARED_BUCKETS=num_shared_buckets,
        NUM_SHARED_VIRTUAL=num_shared_virtual,
        BLOCK=_EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        # Each upper-bound search starts with high=NUM_*_BUCKETS, so its
        # interval has NUM_*_BUCKETS + 1 states.  A power-of-two bucket count
        # needs one more iteration than (count - 1).bit_length(); otherwise
        # the final (sentinel) bucket can be mislabelled as a real expert.
        PER_EXPERT_SEARCH_STEPS=max(1, num_per_expert_buckets.bit_length()),
        SHARED_SEARCH_STEPS=max(1, num_shared_buckets.bit_length()),
        USE_PDL=use_pdl,
        num_warps=4,
        **pdl_kwargs,
    )

    def route(
        *,
        num_virtual_experts: int,
        lora_experts_per_adapter: int,
        shared_outer_local_expert_count: int | None,
        scratch: dict[str, torch.Tensor],
    ) -> RouteView:
        return RouteView(
            view=ROUTE_ALIGNED,
            num_virtual_experts=num_virtual_experts,
            block_size=block_size,
            topk_ids=topk_ids,
            token_slots=token_slots,
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            shared_outer_local_expert_count=shared_outer_local_expert_count,
            maybe_sorted_pair_ids=scratch["sorted"],
            maybe_block_virtual_expert_ids=scratch["block_ids"],
            maybe_num_pairs_post_padded=scratch["padded_pairs"],
        )

    return (
        route(
            num_virtual_experts=num_per_expert_virtual,
            lora_experts_per_adapter=num_local_experts,
            shared_outer_local_expert_count=None,
            scratch=per_expert,
        ),
        route(
            num_virtual_experts=num_shared_virtual,
            lora_experts_per_adapter=1,
            shared_outer_local_expert_count=num_local_experts,
            scratch=shared,
        ),
    )
