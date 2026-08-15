"""Fused ID + histogram + scan + scatter route plan (plan section 7.1 candidate).

Derives the `(adapter, LoRA expert)` key inline through the SAME device
helper the ``fused_ids`` view uses — so the plan-free and plan-based paths
cannot disagree — and has no bucket ceiling, unlike the JIT CUDA align whose
EPT ladder tops out at 32767 virtual experts.

Steady-state launch structure (3 kernels, no memset, no per-call metadata
allocation):

    K1  fused key + histogram        multi-CTA over pairs
    K2  padded exclusive scan        single CTA over V+1 buckets
    K3  block labels + padding fill, multi-CTA over blocks and over pairs,
        and pair scatter             split by program id

The kernel-audit findings this design answers (execution plan section 40):

* **No counts memset.** Per-bucket metadata (counts / block_cum / cursor /
  bucket_end / num_pairs_post_padded) is cached per ``(device, num_buckets)``
  with counts zeroed once at creation; K2 re-zeroes counts AFTER its last read,
  so the invariant "counts is zero between calls" self-restores. K2 cannot
  simply be "the last reader" by accident: K3 needs each block's data end, so
  K2 materializes ``bucket_end = slot_start + count`` explicitly — the audit
  caught that zeroing counts while K3 still re-read them would collapse every
  ``real_end`` to ``slot_start`` and overwrite scattered pairs with padding.
* **Coalesced padding fill.** The fill is a 2D store over
  ``block_ids[:, None] * BM + arange(0, BM)[None, :]`` — contiguous per lane
  row. The previous per-lane unrolled loop issued BLOCK_SIZE_M stores whose
  warps strided 64 B apart: zero coalescing on every one of them.
* **PDL that actually forms edges.** This repo's Triton kernels enable PDL
  with a launch-side ``launch_pdl=True`` kwarg (see decode_attention.py); the
  constexpr alone compiles the intrinsics but never creates an edge. Edges are
  consecutive-launch only: K1->K2 and K2->K3. The waits sit immediately before
  the first predecessor-dependent access of each consumer half, so K3's
  scatter half runs its whole key recompute (scan-independent) before waiting.
* **int32 index math** end to end; the key domain is bounded by V+1 < 2^31.
* **Graph capture with production PDL-off execution** uses stable
  runner-workspace addresses, and every replay's K2 re-establishes the counts
  invariant. Direct callers retain a serial module-cache fallback. PDL is an
  explicit, default-off plan choice: after the sentinel-search fix, both PDL
  states pass the exact full-history replay, while bounded composed off/on
  twins decide whether it is worth enabling. An eager exception BETWEEN K1
  and K2 would leave scratch dirty; all allocation, lookup, and contract
  validation is therefore hoisted above the first launch.

Determinism: K3 claims slots with an atomic cursor, so intra-bucket pair ORDER
varies run to run. That is permitted — intra-bucket order carries no semantics
(the incumbent CUDA and torch paths already disagree on it) and every consumer
writes ``out[pair_id]``, indexed by pair, with fixed-order accumulation. The
registered test pins both properties: plans vary, consumer output is bitwise
stable.

Blocks past ``num_pairs_post_padded`` keep garbage labels/slots: every consumer
early-returns on that device scalar before loading either array. Blocks INSIDE
the plan are fully initialized, including the sentinel bucket's — the aligned
B kernels read a ``-1`` block's slots to zero their output rows, which
is how uninitialized sentinel tails caused an illegal memory access.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.routing import (
    FusedAlignScratch,
    validate_shared_outer,
    virtual_expert_ids_inline,
)

# Launch tiles, selected by a 64-point sweep over 4 representative cells on
# GB300 (tune_fused_align.py, 2026-07-25: best 83.45us vs 88-100us untuned).
# Module constants rather than runtime autotune: graph capture wants one
# deterministic launch shape per call site.
HIST_BLOCK = 512
HIST_WARPS = 8
EXPAND_BLOCK = 128
EXPAND_WARPS = 4
SCAN_CHUNK = 2048
SCAN_WARPS = 4


@triton.jit
def _fused_hist_kernel(
    topk_ids_ptr,
    token_slots_ptr,
    lora_expert_map_ptr,
    counts_ptr,
    num_pairs,
    routed_expert_id_bound,
    NUM_BUCKETS: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Histogram the fused key over pairs; the key array is never materialized.

    Relies on counts arriving ZEROED (the cache-invariant maintained by the
    scan kernel), so there is no zeroing pass and no predecessor edge.
    """
    pair_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        lora_expert_map_ptr,
        pair_ids,
        pair_mask,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=USE_LORA_EXPERT_MAP,
        SHARED_OUTER=SHARED_OUTER,
    )
    # Invalid pairs land in the sentinel bucket at NUM_BUCKETS - 1; its blocks
    # are labelled -1 downstream so LoRA-A skips them and B zero-fills them.
    buckets = tl.where(virtual_ids < 0, NUM_BUCKETS - 1, virtual_ids)
    tl.atomic_add(counts_ptr + buckets, 1, mask=pair_mask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _padded_scan_kernel(
    counts_ptr,
    block_cum_ptr,
    cursor_ptr,
    bucket_end_ptr,
    num_pairs_post_padded_ptr,
    num_buckets,
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Exclusive scan of block-padded bucket sizes. One CTA, O(V), sequential.

    Also restores the counts-are-zero invariant: each chunk's counts are
    zeroed AFTER being read, so the next forward's histogram needs no memset.
    ``bucket_end`` (= first slot + count) is materialized here because K3's
    fill needs it after counts are gone.
    """
    if USE_PDL:
        # First dependent access is the counts load below; no useful preamble.
        tl.extra.cuda.gdc_wait()
    running = 0
    for base in range(0, num_buckets, CHUNK):
        offs = base + tl.arange(0, CHUNK)
        mask = offs < num_buckets
        counts = tl.load(counts_ptr + offs, mask=mask, other=0)
        tl.store(counts_ptr + offs, 0, mask=mask)
        blocks = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        block_start = running + tl.cumsum(blocks) - blocks
        tl.store(block_cum_ptr + offs, block_start, mask=mask)
        slot_start = block_start * BLOCK_SIZE_M
        tl.store(cursor_ptr + offs, slot_start, mask=mask)
        tl.store(bucket_end_ptr + offs, slot_start + counts, mask=mask)
        running += tl.sum(blocks)
    tl.store(block_cum_ptr + num_buckets, running)
    tl.store(num_pairs_post_padded_ptr, running * BLOCK_SIZE_M)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _expand_and_scatter_kernel(
    topk_ids_ptr,
    token_slots_ptr,
    lora_expert_map_ptr,
    cursor_ptr,
    bucket_end_ptr,
    block_cum_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs,
    routed_expert_id_bound,
    num_blocks,
    num_block_programs,
    NUM_BUCKETS: tl.constexpr,
    NUM_VIRTUAL: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Two scan consumers on one grid, split by program id.

    Block half (``pid < num_block_programs``): binary-search ``block_cum`` for
    each block's owning bucket, store the label (-1 for the sentinel bucket and
    for blocks past the padded end), and fill the padding tail from
    ``bucket_end`` with one coalesced 2D store.

    Scatter half: recompute the fused key (two loads and integer math — CHEAPER
    than storing and reloading a [T, K] key array, which is the round trip this
    kernel exists to remove), claim a slot from the bucket cursor, store the
    pair id. The key recompute is scan-independent, so under PDL it runs before
    the wait.
    """
    pid = tl.program_id(0)
    if pid < num_block_programs:
        block_ids = pid * BLOCK + tl.arange(0, BLOCK)
        block_mask = block_ids < num_blocks
        if USE_PDL:
            # Everything below reads scan output (block_cum / bucket_end).
            tl.extra.cuda.gdc_wait()
        low = tl.zeros(block_ids.shape, dtype=tl.int32)
        high = tl.full(block_ids.shape, NUM_BUCKETS, dtype=tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (low + high) // 2
            bound = tl.load(
                block_cum_ptr + tl.minimum(mid + 1, NUM_BUCKETS),
                mask=block_mask,
                other=0,
            )
            take = block_ids >= bound
            low = tl.where(take & (low < high), mid + 1, low)
            high = tl.where(take | (low >= high), high, mid)
        owner = tl.minimum(low, NUM_BUCKETS - 1)
        total_blocks = tl.load(block_cum_ptr + NUM_BUCKETS)
        in_plan = block_mask & (block_ids < total_blocks)
        labelled = in_plan & (owner < NUM_VIRTUAL)
        tl.store(
            block_virtual_expert_ids_ptr + block_ids,
            tl.where(labelled, owner, -1),
            mask=block_mask,
        )
        # Coalesced 2D fill of every in-plan block's padding tail, sentinel
        # bucket included (a -1 block's slots ARE read by the aligned B kernels).
        real_end = tl.load(bucket_end_ptr + owner, mask=in_plan, other=0)
        slots = block_ids[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]
        tl.store(
            sorted_pair_ids_ptr + slots,
            num_pairs,
            mask=in_plan[:, None] & (slots >= real_end[:, None]),
        )
        return

    pair_ids = (pid - num_block_programs) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        lora_expert_map_ptr,
        pair_ids,
        pair_mask,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=USE_LORA_EXPERT_MAP,
        SHARED_OUTER=SHARED_OUTER,
    )
    buckets = tl.where(virtual_ids < 0, NUM_BUCKETS - 1, virtual_ids)
    if USE_PDL:
        # The cursor is scan output; the key recompute above is not.
        tl.extra.cuda.gdc_wait()
    slots = tl.atomic_add(cursor_ptr + buckets, 1, mask=pair_mask)
    tl.store(sorted_pair_ids_ptr + slots, pair_ids, mask=pair_mask)


class _BucketMetadata:
    """Per-(device, num_buckets) persistent scratch with a standing invariant.

    ``counts`` is zero between calls: zeroed once at creation, and re-zeroed by
    the scan kernel after its last read every forward. This is the serial
    direct-call fallback; production supplies runner-owned metadata instead.

    Direct callers that do not supply ``num_pairs_post_padded_out`` receive the
    cached scalar below, so every same-bucket build overwrites the previous
    direct call's value. Concurrent direct callers must supply both their own
    scratch and output scalar.
    """

    def __init__(self, num_buckets: int, device: torch.device) -> None:
        self.counts = torch.zeros(num_buckets, dtype=torch.int32, device=device)
        self.block_cum = torch.empty(num_buckets + 1, dtype=torch.int32, device=device)
        self.cursor = torch.empty(num_buckets, dtype=torch.int32, device=device)
        self.bucket_end = torch.empty(num_buckets, dtype=torch.int32, device=device)
        self.num_pairs_post_padded = torch.empty(1, dtype=torch.int32, device=device)

    @property
    def scratch(self) -> FusedAlignScratch:
        return FusedAlignScratch(
            counts=self.counts,
            block_cumulative=self.block_cum,
            cursor=self.cursor,
            bucket_end=self.bucket_end,
        )


_metadata_cache: dict[tuple[int, int], _BucketMetadata] = {}


def _bucket_metadata(num_buckets: int, device: torch.device) -> _BucketMetadata:
    key = (device.index if device.index is not None else -1, num_buckets)
    meta = _metadata_cache.get(key)
    if meta is None:
        meta = _BucketMetadata(num_buckets, device)
        _metadata_cache[key] = meta
    return meta


def _validate_scratch(
    scratch: FusedAlignScratch,
    *,
    num_buckets: int,
    device: torch.device,
) -> None:
    contracts = (
        ("counts", scratch.counts, (num_buckets,)),
        ("block_cumulative", scratch.block_cumulative, (num_buckets + 1,)),
        ("cursor", scratch.cursor, (num_buckets,)),
        ("bucket_end", scratch.bucket_end, (num_buckets,)),
    )
    for name, tensor, shape in contracts:
        if (
            tensor.shape != shape
            or tensor.dtype is not torch.int32
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"fused align scratch {name} must be a contiguous int32 "
                f"{list(shape)} tensor on {device}"
            )


def fused_align_block_size(
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    *,
    lora_experts_per_adapter: int,
    max_loras: int,
    block_size: int,
    capacity: int,
    lora_expert_map: torch.Tensor | None = None,
    shared_outer_local_expert_count: int | None = None,
    num_pairs_post_padded_out: torch.Tensor | None = None,
    scratch: FusedAlignScratch | None = None,
    use_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded)``.

    Same plan contract as the incumbent align, computed from the SOURCE
    tensors: no ``virtual_topk_ids`` is written or read anywhere. A caller
    retaining multiple routes may provide graph-stable scratch and
    ``num_pairs_post_padded_out`` storage; otherwise direct calls use serial
    per-bucket fallback metadata.
    """
    device = topk_ids.device
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    num_virtual = lora_experts_per_adapter * max_loras
    num_buckets = num_virtual + 1
    # int32 key math holds only below 2**31; nothing upstream enforces it, and
    # a wrapped key would silently land in a valid-looking bucket.
    if num_buckets >= 2**31 or capacity >= 2**31:
        raise ValueError(
            f"fused align uses int32 plan math: num_buckets={num_buckets} and "
            f"capacity={capacity} must both be < 2**31"
        )
    validate_shared_outer(
        shared_outer_local_expert_count=shared_outer_local_expert_count,
        lora_expert_map=lora_expert_map,
        lora_experts_per_adapter=lora_experts_per_adapter,
    )
    if num_pairs_post_padded_out is not None:
        if (
            num_pairs_post_padded_out.shape != (1,)
            or num_pairs_post_padded_out.dtype is not torch.int32
            or num_pairs_post_padded_out.device != device
            or not num_pairs_post_padded_out.is_contiguous()
        ):
            raise ValueError(
                "num_pairs_post_padded_out must be a contiguous int32 [1] "
                f"tensor on {device}"
            )
    shared_outer = shared_outer_local_expert_count is not None
    use_map = lora_expert_map is not None
    # Own name, not a reassignment of the parameter (see routing.py).
    map_arg = lora_expert_map if use_map else topk_ids
    routed_expert_id_bound = (
        shared_outer_local_expert_count
        if shared_outer
        else (map_arg.numel() if use_map else 0)
    )
    num_blocks = capacity // block_size

    # Every host-fallible operation happens BEFORE the first launch, so an
    # exception cannot leave this entry's counts dirty between K1 and K2.
    fallback = (
        _bucket_metadata(num_buckets, device)
        if scratch is None or num_pairs_post_padded_out is None
        else None
    )
    active_scratch = fallback.scratch if scratch is None else scratch
    _validate_scratch(active_scratch, num_buckets=num_buckets, device=device)
    padded_count = (
        fallback.num_pairs_post_padded
        if num_pairs_post_padded_out is None
        else num_pairs_post_padded_out
    )
    sorted_pair_ids = torch.empty(capacity, dtype=torch.int32, device=device)
    block_virtual_expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=device)
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    _fused_hist_kernel[(triton.cdiv(max(num_pairs, 1), HIST_BLOCK),)](
        topk_ids,
        token_slots,
        map_arg,
        active_scratch.counts,
        num_pairs,
        routed_expert_id_bound,
        NUM_BUCKETS=num_buckets,
        LORA_EXPERTS_PER_ADAPTER=lora_experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        USE_LORA_EXPERT_MAP=use_map,
        SHARED_OUTER=shared_outer,
        BLOCK=HIST_BLOCK,
        USE_PDL=use_pdl,
        num_warps=HIST_WARPS,
    )
    _padded_scan_kernel[(1,)](
        active_scratch.counts,
        active_scratch.block_cumulative,
        active_scratch.cursor,
        active_scratch.bucket_end,
        padded_count,
        num_buckets,
        BLOCK_SIZE_M=block_size,
        CHUNK=SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=SCAN_WARPS,
        **pdl_kwargs,
    )
    num_block_programs = triton.cdiv(max(num_blocks, 1), EXPAND_BLOCK)
    num_pair_programs = triton.cdiv(max(num_pairs, 1), EXPAND_BLOCK)
    _expand_and_scatter_kernel[(num_block_programs + num_pair_programs,)](
        topk_ids,
        token_slots,
        map_arg,
        active_scratch.cursor,
        active_scratch.bucket_end,
        active_scratch.block_cumulative,
        sorted_pair_ids,
        block_virtual_expert_ids,
        num_pairs,
        routed_expert_id_bound,
        num_blocks,
        num_block_programs,
        NUM_BUCKETS=num_buckets,
        NUM_VIRTUAL=num_virtual,
        LORA_EXPERTS_PER_ADAPTER=lora_experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        USE_LORA_EXPERT_MAP=use_map,
        SHARED_OUTER=shared_outer,
        BLOCK=EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        # The search interval is [0, NUM_BUCKETS], not [0, NUM_BUCKETS).
        # It therefore contains NUM_BUCKETS + 1 possible states.  Using
        # (num_buckets - 1).bit_length() is one iteration short whenever the
        # bucket count is a power of two.  The smallest production example is
        # one shared-outer adapter plus the sentinel bucket: with one step,
        # an all-sentinel route is incorrectly labelled adapter 0.
        SEARCH_STEPS=max(1, num_buckets.bit_length()),
        USE_PDL=use_pdl,
        num_warps=EXPAND_WARPS,
        **pdl_kwargs,
    )
    return sorted_pair_ids, block_virtual_expert_ids, padded_count
