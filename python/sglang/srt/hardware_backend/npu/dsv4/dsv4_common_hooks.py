"""Helpers used by mem_cache/common.py to wire DSV4-NPU per-req tables.

mem_cache/common.py runs platform-agnostic alloc flow. When the model is
DSV4 on NPU, ``alloc_paged_token_slots_{extend,decode}`` already stashed the
:class:`DSV4OutCacheLoc` the allocator returned onto
``batch.out_cache_loc_dsv4``. After each ``alloc_extend`` / ``alloc_decode``
these hooks then:

  1. Read the bundle from ``batch.out_cache_loc_dsv4``.
  2. Write the per-pool slot ids into the per-req tables on the
     :class:`DSV4NPUReqToTokenPool`.

Non-DSV4 paths leave ``batch.out_cache_loc_dsv4`` None, so this module is a
no-op for them.

TODO: the disagg DSV4 path bypasses these hooks — it calls
``allocator.alloc_extend`` directly then ``req_to_token_pool.write`` without
going through ``mem_cache/common.py`` (see ``disaggregation/decode.py``). The
DSV4OutCacheLoc bundle is still produced but never written into the per-req
tables, so disagg + DSV4 is unsupported here (c-pages leak). Fixing requires
calling these hooks from disagg's per-req alloc loop, or moving the write
into the allocator itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


def maybe_write_dsv4_extend(
    batch: ScheduleBatch,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> None:
    """Post-alloc_extend hook for DSV4. No-op when allocator/pool is not DSV4.

    For each compressed pool (c4 / c128), spreads the flat
    ``out_c{4,128}_loc`` tensor across requests using per-req extend
    counts (``seq_lens[i] // ratio - prefix_lens[i] // ratio``) and writes
    the resulting slot ids into ``req_to_token_c{4,128}[req, prefix:seq]``.

    Also writes ``req_to_token_swa[req, prefix:seq]`` with the swa slots
    derived from out_full_loc via the SWA index mapping.
    """
    # Bundle stashed on batch.out_cache_loc_dsv4 by mem_cache/common.py;
    # None on CUDA / non-V4 paths → no-op.
    bundle = batch.out_cache_loc_dsv4
    if bundle is None:
        return

    req_to_token_pool = batch.req_to_token_pool
    if not hasattr(req_to_token_pool, "write_c4"):
        return  # non-DSV4 pool; skip defensively (shouldn't happen)

    # SWA writes: prefix..seq token positions, one slot per raw token.
    _write_per_req_slice(
        req_to_token_pool.write_swa,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_swa_loc,
        ratio=1,
    )

    # c4 / c128 writes: prefix//ratio .. seq//ratio compressed positions.
    _write_per_req_slice(
        req_to_token_pool.write_c4,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_c4_loc,
        ratio=4,
    )
    _write_per_req_slice(
        req_to_token_pool.write_c128,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_c128_loc,
        ratio=128,
    )

    # c4_state / c128_state writes: tail-only. Bundle length is
    # sum(c{N}_state_alloc_len_i), NOT total raw extend tokens; each req's slots
    # go at raw positions [req.c{N}_state_alloc_offset, seq_len).
    if bundle.out_c4_state_loc is not None and hasattr(
        req_to_token_pool, "write_c4_state"
    ):
        _write_state_tail_per_req(
            req_to_token_pool.write_c4_state,
            req_pool_indices_cpu,
            [getattr(r, "c4_state_alloc_offset", 0) for r in batch.reqs],
            seq_lens_cpu,
            bundle.out_c4_state_loc,
        )
    if bundle.out_c128_state_loc is not None and hasattr(
        req_to_token_pool, "write_c128_state"
    ):
        _write_state_tail_per_req(
            req_to_token_pool.write_c128_state,
            req_pool_indices_cpu,
            [getattr(r, "c128_state_alloc_offset", 0) for r in batch.reqs],
            seq_lens_cpu,
            bundle.out_c128_state_loc,
        )


def maybe_write_dsv4_decode(
    batch: ScheduleBatch,
    seq_lens_cpu: torch.Tensor,
    token_per_req: int,
) -> None:
    """Post-alloc_decode hook for DSV4. Spreads the new token slot ids
    (one per req for swa, gated by ratio boundary for c4/c128) into the
    per-req tables on DSV4NPUReqToTokenPool.

    ``seq_lens_cpu`` is the POST-decode seq len (already incremented by
    ``token_per_req``); the new compressed tokens go at positions
    ``[(old_seq) // ratio, (new_seq) // ratio)``.
    """
    # Bundle stashed on batch.out_cache_loc_dsv4 by mem_cache/common.py;
    # None on CUDA / non-V4 paths → no-op.
    bundle = batch.out_cache_loc_dsv4
    if bundle is None:
        return

    req_to_token_pool = batch.req_to_token_pool
    if not hasattr(req_to_token_pool, "write_c4"):
        return

    prefix_lens_cpu = (seq_lens_cpu - token_per_req).clamp(min=0)
    req_pool_indices_cpu = batch.req_pool_indices.cpu()

    _write_per_req_slice(
        req_to_token_pool.write_swa,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_swa_loc,
        ratio=1,
    )
    _write_per_req_slice(
        req_to_token_pool.write_c4,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_c4_loc,
        ratio=4,
    )
    _write_per_req_slice(
        req_to_token_pool.write_c128,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_c128_loc,
        ratio=128,
    )

    # State table decode writes: one slot per raw decode token (ratio=1).
    if bundle.out_c4_state_loc is not None and hasattr(
        req_to_token_pool, "write_c4_state"
    ):
        _write_per_req_slice(
            req_to_token_pool.write_c4_state,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            seq_lens_cpu,
            bundle.out_c4_state_loc,
            ratio=1,
        )
    if bundle.out_c128_state_loc is not None and hasattr(
        req_to_token_pool, "write_c128_state"
    ):
        _write_per_req_slice(
            req_to_token_pool.write_c128_state,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            seq_lens_cpu,
            bundle.out_c128_state_loc,
            ratio=1,
        )


def _write_per_req(
    write_fn,
    req_pool_indices_cpu: torch.Tensor,
    flat_loc: torch.Tensor,
    bounds_fn,
) -> None:
    """Distribute a flat ``[total_alloc]`` slot tensor across reqs.

    ``bounds_fn(i) -> (lo, hi)`` gives req i's write window; the matching
    ``hi - lo`` slots are sliced off ``flat_loc`` in order and written via
    ``write_fn((req_idx, slice(lo, hi)), values)``. flat_loc may be None /
    empty when the alloc path bypassed DSV4NPUTokenToKVPoolAllocator (e.g.
    page_size=1 or HiSparse wrapper); skip then.
    """
    if flat_loc is None or flat_loc.numel() == 0:
        return
    pt = 0
    for i in range(req_pool_indices_cpu.shape[0]):
        lo, hi = bounds_fn(i)
        alloc_len = max(0, hi - lo)
        if alloc_len == 0:
            continue
        req_idx = int(req_pool_indices_cpu[i].item())
        chunk = flat_loc[pt : pt + alloc_len].to(torch.int32)
        write_fn((req_idx, slice(lo, hi)), chunk)
        pt += alloc_len


def _write_state_tail_per_req(
    write_fn,
    req_pool_indices_cpu: torch.Tensor,
    state_alloc_offsets: list,
    seq_lens_cpu: torch.Tensor,
    flat_loc: torch.Tensor,
) -> None:
    """Tail-only state write: req i's slots go at ``[state_alloc_offsets[i],
    seq_lens[i])`` in ``req_to_token_c{N}_state``."""
    _write_per_req(
        write_fn,
        req_pool_indices_cpu,
        flat_loc,
        lambda i: (int(state_alloc_offsets[i]), int(seq_lens_cpu[i].item())),
    )


def _write_per_req_slice(
    write_fn,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    flat_loc: torch.Tensor,
    ratio: int,
) -> None:
    """Compressed-position write: req i's slots go at
    ``[prefix_lens[i] // ratio, seq_lens[i] // ratio)``."""
    _write_per_req(
        write_fn,
        req_pool_indices_cpu,
        flat_loc,
        lambda i: (
            int(prefix_lens_cpu[i].item()) // ratio,
            int(seq_lens_cpu[i].item()) // ratio,
        ),
    )


def maybe_evict_dsv4_state(batch: ScheduleBatch, req: Req, pre_len: int) -> None:
    """Per-decode evict for the DSV4-NPU compress-state pools, independent of
    SWA evict cadence. Called every decode step from ``ScheduleBatch``.

    The state pool is small (~2 pages c4 / ~3 pages c128 of raw positions per
    req) — with a large sliding_window (SWA evict fires every
    ``eviction_interval`` and needs ``pre_len > sliding_window + page_size`` to
    free anything) the pool exhausts before the first SWA frontier advance, so
    we drain it here on its own cadence.

    Retention windows (kernel read window + decode lookahead margin):
    c4 = 8 + 16, c128 = 128 + 64 raw positions — intentionally smaller than one
    SWA page so the first eviction fires before the small pool fills. Watermarks
    are page-aligned so freed slots are whole pages reclaimable by the paged
    allocator. ``req.c{4,128}_state_alloc_offset`` (read/written via getattr/
    setattr) is the low-water mark. No-op on non-DSV4-NPU paths.
    """
    allocator = batch.token_to_kv_pool_allocator
    pool = batch.req_to_token_pool
    if not hasattr(allocator, "c4_state_attn_allocator") or (
        allocator.c4_state_attn_allocator is None
        and allocator.c128_state_attn_allocator is None
    ):
        return

    page_size = batch.tree_cache.page_size
    c4_watermark = ((max(0, pre_len - (8 + 16))) // page_size) * page_size
    c128_watermark = ((max(0, pre_len - (128 + 64))) // page_size) * page_size

    _free_state_range(
        allocator.c4_state_attn_allocator,
        pool,
        "req_to_token_c4_state",
        req,
        "c4_state_alloc_offset",
        c4_watermark,
    )
    _free_state_range(
        allocator.c128_state_attn_allocator,
        pool,
        "req_to_token_c128_state",
        req,
        "c128_state_alloc_offset",
        c128_watermark,
    )


def maybe_evict_dsv4_state_on_swa(
    allocator, pool, req: Req, new_swa_evicted_seqlen: int
) -> None:
    """Free compress-state slots that ride along with SWA eviction.

    State at raw positions < ``swa_evicted_seqlen`` is no longer readable (the
    compressor only reads the trailing ``2*ratio`` window) and is returned to
    its paged allocator to keep the small state pool from exhausting on long
    generations. No-op when the DSV4-NPU state allocators are absent.

    This path is needed for small-sliding-window models where
    ``sliding_window < retention`` (e.g. c128 retention 192 > window 128):
    in that case the watermark-based eviction alone may not free slots
    fast enough, and the SWA-ride eviction is the primary reclaim mechanism.
    For typical large-window models (DS-V4 with window >> 192), the
    watermark eviction always runs first, making this path a no-op.
    """
    if not hasattr(allocator, "c4_state_attn_allocator"):
        return
    _free_state_range(
        allocator.c4_state_attn_allocator,
        pool,
        "req_to_token_c4_state",
        req,
        "c4_state_alloc_offset",
        new_swa_evicted_seqlen,
    )
    _free_state_range(
        allocator.c128_state_attn_allocator,
        pool,
        "req_to_token_c128_state",
        req,
        "c128_state_alloc_offset",
        new_swa_evicted_seqlen,
    )


def _free_state_range(
    state_allocator,
    pool,
    table_attr: str,
    req: Req,
    offset_attr: str,
    watermark: int,
) -> None:
    """Free ``[alloc_offset, watermark)`` raw-position state slots for ``req``
    and advance its low-water mark. No-op when the allocator/table is absent or
    the watermark hasn't advanced past the current offset."""
    offset = getattr(req, offset_attr, 0)
    if state_allocator is None or not hasattr(pool, table_attr) or watermark <= offset:
        return
    free_slots = getattr(pool, table_attr)[req.req_pool_idx, offset:watermark]
    state_allocator.free(free_slots.to(torch.int64))
    setattr(req, offset_attr, watermark)
