"""Helpers used by mem_cache/common.py to wire DSV4-NPU KV tables.

mem_cache/common.py runs platform-agnostic alloc flow. When the model is
DSV4 on NPU, ``alloc_paged_token_slots_{extend,decode}`` already stashed the
:class:`DSV4OutCacheLoc` the allocator returned onto
``batch.out_cache_loc_dsv4``. After each ``alloc_extend`` / ``alloc_decode``
these hooks then:

  1. Read the bundle from ``batch.out_cache_loc_dsv4``.
  2. Write newly allocated C128 page ids into the per-request sidecar.

Compressor state is fixed ring storage and does not participate in this
allocation/write path. PD reuses the public SWA/C128-state payloads and only
builds an NPU-specific payload for the independently addressed C128 KV pool.

Non-DSV4 paths leave ``batch.out_cache_loc_dsv4`` None, so this module is a
no-op for them.

The disagg per-req prealloc path does not build a ``ScheduleBatch`` and so
bypasses the batch hook; it writes the same sidecar via
``write_dsv4_prealloc_tables`` (driven by ``dsv4_unwrap_prealloc``).
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

    Spreads the flat ``out_c128_loc`` tensor across requests and writes newly
    allocated page ids into ``req_to_c128_sidecar``. C4 locations are derived
    from the full-token table.

    """
    # Bundle stashed on batch.out_cache_loc_dsv4 by mem_cache/common.py;
    # None on CUDA / non-V4 paths → no-op.
    bundle = batch.out_cache_loc_dsv4
    if bundle is None:
        return

    req_to_token_pool = batch.req_to_token_pool
    if not hasattr(req_to_token_pool, "write_c128"):
        return  # non-DSV4 pool; skip defensively (shouldn't happen)

    _write_dsv4_tables(
        req_to_token_pool,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle,
    )


def dsv4_state_payloads(
    req_to_token_pool,
    req_pool_idx: int,
    seq_len: int,
    page_size: int,
    *,
    prefix_len: int = 0,
):
    """Build the only NPU-specific DSV4 PD payload: C128 KV pages."""

    import numpy as np

    from sglang.srt.disaggregation.ascend.conn import AscendStateType

    seq_len = max(0, int(seq_len))
    prefix_len = max(0, min(int(prefix_len), seq_len))

    def c128_kv_pages():
        c128_page_size = req_to_token_pool.c128_page_size
        lo = prefix_len // (128 * c128_page_size)
        hi = (seq_len // 128 + c128_page_size - 1) // c128_page_size
        if hi <= lo:
            return np.empty((0,), dtype=np.int32)
        pages = (
            req_to_token_pool.req_to_c128_sidecar[req_pool_idx, lo:hi]
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        return pages[pages > 0]

    return {AscendStateType.DSV4_C128: c128_kv_pages}


def dsv4_prealloc_kwargs(allocator, req, fill_len, req_to_token_pool, *, device):
    """Extra ``alloc_extend(_swa_tail)`` kwargs for the DSV4 allocator; ``{}`` for
    non-DSV4 so callers can splat it unconditionally."""
    if not hasattr(allocator, "c128_attn_allocator"):
        return {}
    return dict(
        req_pool_indices=torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device=device
        ),
        req_to_token_pool=req_to_token_pool,
    )


def dsv4_unwrap_prealloc(kv_loc, req_to_token_pool, req, prefix_len, fill_len):
    """Unwrap a DSV4OutCacheLoc bundle to its full-pool loc and write the
    per-req tables; a plain tensor (non-DSV4) passes through unchanged."""
    if kv_loc is None or not hasattr(kv_loc, "out_full_loc"):
        return kv_loc
    write_dsv4_prealloc_tables(req_to_token_pool, req, prefix_len, fill_len, kv_loc)
    return kv_loc.out_full_loc


def write_dsv4_prealloc_tables(
    req_to_token_pool,
    req: Req,
    prefix_len: int,
    fill_len: int,
    bundle,
) -> None:
    """Write the DSV4 per-req tables for one request on the disagg-decode
    prealloc path (no ScheduleBatch); no-op without bundle / DSV4 tables."""
    if bundle is None or not hasattr(req_to_token_pool, "write_c128"):
        return
    rp = torch.tensor([req.req_pool_idx])
    pl = torch.tensor([prefix_len])
    sl = torch.tensor([fill_len])

    _write_dsv4_tables(
        req_to_token_pool,
        rp,
        pl,
        sl,
        bundle,
    )


def _write_dsv4_tables(
    req_to_token_pool,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    bundle,
) -> None:
    """Write newly allocated C128 page ids into the request sidecar."""
    _write_per_req_slice(
        req_to_token_pool.write_c128,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_c128_loc,
        ratio=128,
    )


def maybe_write_dsv4_decode(
    batch: ScheduleBatch,
    seq_lens_cpu: torch.Tensor,
    token_per_req: int,
) -> None:
    """Post-alloc_decode hook for DSV4. Spreads new C128 KV slot ids into
    the per-req sidecar on DSV4NPUReqToTokenPool.

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
    if not hasattr(req_to_token_pool, "write_c128"):
        return

    prefix_lens_cpu = (seq_lens_cpu - token_per_req).clamp(min=0)
    req_pool_indices_cpu = batch.req_pool_indices.cpu()

    _write_per_req_slice(
        req_to_token_pool.write_c128,
        req_pool_indices_cpu,
        prefix_lens_cpu,
        seq_lens_cpu,
        bundle.out_c128_loc,
        ratio=128,
    )


def maybe_build_dsv4_verify_bundle(
    batch: ScheduleBatch,
    draft_token_num: int,
    *,
    live_seq_lens_cpu: torch.Tensor | None = None,
):
    """Build the DSV4 cache-location view for one target-verify pass.

    Spec-v2 reserves cache ahead of time, so target verify must select only the
    current draft interval from the per-request DSV4 tables instead of reusing
    the larger allocation bundle produced during decode preparation.
    """
    pool = batch.req_to_token_pool
    if not hasattr(pool, "req_to_c128_sidecar"):
        return None
    reserve_bundle = batch.out_cache_loc_dsv4
    if reserve_bundle is None:
        return None

    req_indices = batch.req_pool_indices_cpu.tolist()

    if live_seq_lens_cpu is None:
        live_seq_lens_cpu = batch.seq_lens_cpu
    if live_seq_lens_cpu is None:
        live_seq_lens_cpu = batch.seq_lens[: len(req_indices)].cpu()
    live_seq_lens = live_seq_lens_cpu[: len(req_indices)].tolist()

    verify_lens = [int(draft_token_num)] * len(req_indices)

    def flatten_interval(table: torch.Tensor, ratio: int) -> torch.Tensor:
        page_size = pool.c128_page_size
        chunks = []
        for req_idx, live_seq_len, verify_len in zip(
            req_indices, live_seq_lens, verify_lens
        ):
            start = int(live_seq_len) // ratio
            end = (int(live_seq_len) + int(verify_len)) // ratio
            if end > start:
                positions = torch.arange(start, end, device=table.device)
                pages = table[int(req_idx), positions // page_size]
                chunks.append(pages * page_size + positions % page_size)
        return torch.cat(chunks) if chunks else table.new_empty((0,))

    out_full_loc = batch.out_cache_loc
    out_c4_loc = out_full_loc[(out_full_loc >= 0) & ((out_full_loc % 4) == 3)] // 4
    return type(reserve_bundle)(
        out_full_loc=out_full_loc,
        out_swa_loc=batch.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            out_full_loc
        ),
        out_c4_loc=out_c4_loc,
        out_c128_loc=flatten_interval(pool.req_to_c128_sidecar, 128),
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
