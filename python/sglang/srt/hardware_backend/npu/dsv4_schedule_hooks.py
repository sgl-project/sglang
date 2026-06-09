"""DSV4-NPU compress-state scheduling hooks for ``managers/schedule_batch.py``.

The DSV4-NPU fused compressor keeps a small, separately-paged ``c{4,128}_state``
pool (kv + score), allocated tail-only and evicted on its own cadence. The
bookkeeping is pure DSV4-NPU logic, so it lives here rather than in the
community ``ScheduleBatch`` — that class only calls these free functions at the
relevant points (extend / decode prepare, per-decode evict, SWA evict).

All functions are **gated internally** on the allocator exposing
``c4_state_attn_allocator`` and are a no-op for CUDA / non-DSV4 paths. Per-req
state bookkeeping is read/written via ``getattr``/``setattr`` so the community
``Req`` / ``ScheduleBatch`` classes need no DSV4-specific field declarations:

  * ``req.c{4,128}_state_kv_len`` — cumulative slot count this req has consumed
    in the paged state pool (drives the paged allocator's prefix/seq contract;
    never decreases on eviction). Default 0.
  * ``req.c{4,128}_state_alloc_offset`` — low-water raw-position mark; slots in
    ``[0, offset)`` were already released to the state allocator (tied to the
    SWA eviction frontier). Default 0. Read by ``dsv4_common_hooks`` too.
  * ``batch.dsv4_state_lens`` — the per-pool ``DSV4StateLens`` bundle consumed
    by ``DSV4NPUTokenToKVPoolAllocator`` via ``mem_cache/common.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from sglang.srt.model_executor.forward_batch_info import DSV4StateLens

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


def _has_state_pools(batch: "ScheduleBatch") -> bool:
    allocator = batch.token_to_kv_pool_allocator
    return (
        hasattr(allocator, "c4_state_attn_allocator")
        and allocator.c4_state_attn_allocator is not None
    )


def maybe_compute_dsv4_state_lens_extend(
    batch: "ScheduleBatch", reqs: List["Req"], seq_lens: List[int]
) -> None:
    """Per-req c{4,128}_state pool alloc lens for extend (tail-only).

    State pool stores only the trailing portion of each sequence (the c{N}
    compressor's read/write window); the tail length depends on raw seq_len's
    alignment to the SWA page boundary (128)::

        c4_alloc_len  = tail + 128 if (tail <= 3 and seq_len >= 128) else tail
        c128_alloc_len = tail                  where tail = seq_len % 128

    Long prefills don't allocate state slots for already-compressed positions —
    only the trailing partial window — so the small paged state pool (~256
    slots/req) stays sufficient even for 28k-token prompts.

    Reference: iforgetmyname/sglang dsv4_release schedule_batch.py L1742-1747.
    """
    if not _has_state_pools(batch):
        return
    c4_prefix_list: List[int] = []
    c4_seq_list: List[int] = []
    c128_prefix_list: List[int] = []
    c128_seq_list: List[int] = []
    for req, seq_len in zip(reqs, seq_lens):
        tail = seq_len % 128
        c4_alloc_len = tail + 128 if (tail <= 3 and seq_len >= 128) else tail
        c128_alloc_len = tail

        prev_c4 = getattr(req, "c4_state_kv_len", 0)
        prev_c128 = getattr(req, "c128_state_kv_len", 0)
        new_c4 = prev_c4 + c4_alloc_len
        new_c128 = prev_c128 + c128_alloc_len

        c4_prefix_list.append(prev_c4)
        c4_seq_list.append(new_c4)
        c128_prefix_list.append(prev_c128)
        c128_seq_list.append(new_c128)

        req.c4_state_kv_len = new_c4
        req.c128_state_kv_len = new_c128
        req.c4_state_alloc_offset = seq_len - c4_alloc_len
        req.c128_state_alloc_offset = seq_len - c128_alloc_len

    batch.dsv4_state_lens = _build_state_lens(
        batch,
        c4_prefix_list,
        c4_seq_list,
        c128_prefix_list,
        c128_seq_list,
        c4_extend_num_tokens=int(
            sum(s - p for s, p in zip(c4_seq_list, c4_prefix_list))
        ),
        c128_extend_num_tokens=int(
            sum(s - p for s, p in zip(c128_seq_list, c128_prefix_list))
        ),
    )


def maybe_compute_dsv4_state_lens_decode(batch: "ScheduleBatch", bs: int) -> None:
    """Per-req c{4,128}_state pool alloc lens for decode.

    Decode adds exactly 1 state slot per req per pool (each generated token
    produces one new state slot). ``c{N}_state_alloc_offset`` does NOT advance
    during decode — it only advances when SWA evict releases old raw positions
    (see :func:`maybe_evict_dsv4_state_on_swa`).
    """
    if not _has_state_pools(batch):
        return
    c4_prefix_list: List[int] = []
    c4_seq_list: List[int] = []
    c128_prefix_list: List[int] = []
    c128_seq_list: List[int] = []
    for req in batch.reqs:
        prev_c4 = getattr(req, "c4_state_kv_len", 0)
        prev_c128 = getattr(req, "c128_state_kv_len", 0)
        new_c4 = prev_c4 + 1
        new_c128 = prev_c128 + 1

        c4_prefix_list.append(prev_c4)
        c4_seq_list.append(new_c4)
        c128_prefix_list.append(prev_c128)
        c128_seq_list.append(new_c128)

        req.c4_state_kv_len = new_c4
        req.c128_state_kv_len = new_c128

    batch.dsv4_state_lens = _build_state_lens(
        batch,
        c4_prefix_list,
        c4_seq_list,
        c128_prefix_list,
        c128_seq_list,
        c4_extend_num_tokens=bs,
        c128_extend_num_tokens=bs,
    )


def _build_state_lens(
    batch: "ScheduleBatch",
    c4_prefix_list: List[int],
    c4_seq_list: List[int],
    c128_prefix_list: List[int],
    c128_seq_list: List[int],
    *,
    c4_extend_num_tokens: int,
    c128_extend_num_tokens: int,
) -> DSV4StateLens:
    c4_prefix_lens_cpu = torch.tensor(c4_prefix_list, dtype=torch.int64)
    c4_seq_lens_cpu = torch.tensor(c4_seq_list, dtype=torch.int64)
    c128_prefix_lens_cpu = torch.tensor(c128_prefix_list, dtype=torch.int64)
    c128_seq_lens_cpu = torch.tensor(c128_seq_list, dtype=torch.int64)
    return DSV4StateLens(
        c4_prefix_lens=c4_prefix_lens_cpu.to(batch.device, non_blocking=True),
        c4_prefix_lens_cpu=c4_prefix_lens_cpu,
        c4_seq_lens=c4_seq_lens_cpu.to(batch.device, non_blocking=True),
        c4_seq_lens_cpu=c4_seq_lens_cpu,
        c4_extend_num_tokens=c4_extend_num_tokens,
        c128_prefix_lens=c128_prefix_lens_cpu.to(batch.device, non_blocking=True),
        c128_prefix_lens_cpu=c128_prefix_lens_cpu,
        c128_seq_lens=c128_seq_lens_cpu.to(batch.device, non_blocking=True),
        c128_seq_lens_cpu=c128_seq_lens_cpu,
        c128_extend_num_tokens=c128_extend_num_tokens,
    )


def maybe_evict_dsv4_state(batch: "ScheduleBatch", req: "Req", pre_len: int) -> None:
    """Per-decode evict for the compress-state pools, independent of SWA evict.

    The state pool is small (~2 pages c4 / ~3 pages c128 of raw positions per
    req) — with a large sliding_window (SWA evict fires every
    ``eviction_interval`` and needs ``pre_len > sliding_window + page_size`` to
    free anything) the state pool exhausts before the first SWA frontier
    advance. Reference (iforgetmyname/sglang dsv4_release) uses a dedicated
    ``evict_swa_c4c128_state`` path on its tree cache; we evict here instead.

    Retention windows (kernel read window + decode lookahead margin) mirror
    reference chunk_cache.py:140-144: c4 = 8 + 16, c128 = 128 + 64 raw
    positions. Intentionally smaller than one SWA page so the first eviction
    fires before the small pool fills. Watermarks are page-aligned so freed
    slots correspond to whole pages reclaimable by the paged allocator.
    """
    allocator = batch.token_to_kv_pool_allocator
    pool = batch.req_to_token_pool
    if not hasattr(allocator, "c4_state_attn_allocator") or (
        allocator.c4_state_attn_allocator is None
        and allocator.c128_state_attn_allocator is None
    ):
        return

    page_size = batch.tree_cache.page_size
    C4_RETENTION = 8 + 16
    C128_RETENTION = 128 + 64
    c4_watermark = ((max(0, pre_len - C4_RETENTION)) // page_size) * page_size
    c128_watermark = ((max(0, pre_len - C128_RETENTION)) // page_size) * page_size

    _free_state_range(
        allocator.c4_state_attn_allocator, pool, "req_to_token_c4_state",
        req, "c4_state_alloc_offset", c4_watermark,
    )
    _free_state_range(
        allocator.c128_state_attn_allocator, pool, "req_to_token_c128_state",
        req, "c128_state_alloc_offset", c128_watermark,
    )


def maybe_evict_dsv4_state_on_swa(
    batch: "ScheduleBatch", req: "Req", new_swa_evicted_seqlen: int
) -> None:
    """Free compress-state slots that ride along with SWA eviction.

    State at raw positions < ``swa_evicted_seqlen`` is no longer readable (the
    compressor only reads the trailing ``2*ratio`` window) and is returned to
    its paged allocator to keep the small state pool from exhausting on long
    generations. Mirrors reference chunk_cache state eviction. No-op when the
    DSV4-NPU state allocators are absent.
    """
    allocator = batch.token_to_kv_pool_allocator
    pool = batch.req_to_token_pool
    if not hasattr(allocator, "c4_state_attn_allocator"):
        return
    _free_state_range(
        allocator.c4_state_attn_allocator, pool, "req_to_token_c4_state",
        req, "c4_state_alloc_offset", new_swa_evicted_seqlen,
    )
    _free_state_range(
        allocator.c128_state_attn_allocator, pool, "req_to_token_c128_state",
        req, "c128_state_alloc_offset", new_swa_evicted_seqlen,
    )


def _free_state_range(
    state_allocator,
    pool,
    table_attr: str,
    req: "Req",
    offset_attr: str,
    watermark: int,
) -> None:
    """Free ``[alloc_offset, watermark)`` raw-position state slots for ``req``
    and advance its low-water mark. No-op when the allocator/table is absent or
    the watermark hasn't advanced past the current offset."""
    offset = getattr(req, offset_attr, 0)
    if (
        state_allocator is None
        or not hasattr(pool, table_attr)
        or watermark <= offset
    ):
        return
    free_slots = getattr(pool, table_attr)[req.req_pool_idx, offset:watermark]
    state_allocator.free(free_slots.to(torch.int64))
    setattr(req, offset_attr, watermark)
