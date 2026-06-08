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

TODO: disagg DSV4 path bypasses ``alloc_for_extend`` / ``alloc_for_decode``
entirely (see ``disaggregation/decode.py``: it calls
``allocator.alloc_extend`` directly then ``req_to_token_pool.write`` without
going through ``mem_cache/common.py``). The new DSV4OutCacheLoc bundle is
still produced by the allocator on those paths but never written into the
per-req tables, so disagg + DSV4 is unsupported here — same status as the
``8c1e87b`` commit message noted ("Disagg decode and PP profiling paths
bypass release_kv_cache; c-pages leak there"). Fixing requires either
calling these hooks from disagg's per-req alloc loop or moving the write
into the allocator itself.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def maybe_write_dsv4_extend(
    batch: "ScheduleBatch",
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
    # The DSV4 allocator returns the DSV4OutCacheLoc bundle, which
    # mem_cache/common.py already stashed on batch.out_cache_loc_dsv4. Read it
    # here (no allocator side-channel). None on CUDA / non-V4 paths → no-op.
    bundle = batch.out_cache_loc_dsv4
    if bundle is None:
        return

    req_to_token_pool = batch.req_to_token_pool
    if not hasattr(req_to_token_pool, "write_c4"):
        # Non-DSV4 ReqToTokenPool — should not happen if dispatch is wired
        # correctly, but skip defensively.
        return

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

    # c4_state / c128_state writes: tail-only. Bundle is now of length
    # ``sum(c{N}_state_alloc_len_i)`` (NOT total raw extend tokens) per
    # ScheduleBatch._compute_dsv4_state_lens_extend. Each req's slot ids go
    # at raw positions ``[req.c{N}_state_alloc_offset, seq_len)`` —
    # corresponds to the trailing window the compressor reads / writes.
    if bundle.out_c4_state_loc is not None and hasattr(
        req_to_token_pool, "write_c4_state"
    ):
        _write_state_tail_per_req(
            req_to_token_pool.write_c4_state,
            req_pool_indices_cpu,
            [r.c4_state_alloc_offset for r in batch.reqs],
            seq_lens_cpu,
            bundle.out_c4_state_loc,
        )
    if bundle.out_c128_state_loc is not None and hasattr(
        req_to_token_pool, "write_c128_state"
    ):
        _write_state_tail_per_req(
            req_to_token_pool.write_c128_state,
            req_pool_indices_cpu,
            [r.c128_state_alloc_offset for r in batch.reqs],
            seq_lens_cpu,
            bundle.out_c128_state_loc,
        )


def maybe_write_dsv4_decode(
    batch: "ScheduleBatch",
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
    # The DSV4 allocator returns the DSV4OutCacheLoc bundle, which
    # mem_cache/common.py already stashed on batch.out_cache_loc_dsv4. Read it
    # here (no allocator side-channel). None on CUDA / non-V4 paths → no-op.
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


def _write_state_tail_per_req(
    write_fn,
    req_pool_indices_cpu: torch.Tensor,
    state_alloc_offsets: list,
    seq_lens_cpu: torch.Tensor,
    flat_loc: torch.Tensor,
) -> None:
    """Distribute the tail-only state bundle across reqs.

    ``flat_loc`` length == ``sum_i (seq_lens_i - state_alloc_offsets_i)``
    (set by the allocator's c{N}_state_attn_allocator.alloc_extend, which
    received per-req prefix/seq diffs equal to per-req alloc_lens). Each
    req's bundle slice goes at raw positions
    ``[state_alloc_offsets[i], seq_lens[i])`` in ``req_to_token_c{N}_state``.

    flat_loc may be None when the c{N}_state allocator is uninitialized
    (model has no c{N} layers); skip then.
    """
    if flat_loc is None or flat_loc.numel() == 0:
        return
    pt = 0
    n_reqs = req_pool_indices_cpu.shape[0]
    for i in range(n_reqs):
        offset = int(state_alloc_offsets[i])
        seq = int(seq_lens_cpu[i].item())
        alloc_len = max(0, seq - offset)
        if alloc_len == 0:
            continue
        req_idx = int(req_pool_indices_cpu[i].item())
        chunk = flat_loc[pt : pt + alloc_len].to(torch.int32)
        write_fn((req_idx, slice(offset, seq)), chunk)
        pt += alloc_len


def _write_per_req_slice(
    write_fn,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    flat_loc: torch.Tensor,
    ratio: int,
) -> None:
    """Distribute a flat ``[total_alloc]`` slot tensor across reqs and
    call ``write_fn((req_idx, slice(prefix//ratio, seq//ratio)), values)``.

    Skip-writes when a req contributes 0 new compressed tokens for this
    ratio (extend straddling a ratio boundary with prefix_len already past
    boundary, or seq_len < ratio).

    flat_loc may be None when the upstream alloc path bypassed
    DSV4NPUTokenToKVPoolAllocator.alloc_extend (e.g., page_size=1 path
    going through allocator.alloc(), or HiSparse wrapper). In that case
    skip — there's nothing to write, and the per-req table for this pool
    keeps its stale prior content (attention backend tolerates this since
    it slices [:seq_lens] only).
    """
    if flat_loc is None or flat_loc.numel() == 0:
        return
    pt = 0
    n_reqs = req_pool_indices_cpu.shape[0]
    for i in range(n_reqs):
        c_prefix = int(prefix_lens_cpu[i].item()) // ratio
        c_seq = int(seq_lens_cpu[i].item()) // ratio
        c_extend = max(0, c_seq - c_prefix)
        if c_extend == 0:
            continue
        req_idx = int(req_pool_indices_cpu[i].item())
        chunk = flat_loc[pt : pt + c_extend].to(torch.int32)
        write_fn((req_idx, slice(c_prefix, c_seq)), chunk)
        pt += c_extend
    # If flat_loc was sized to total extend_num_tokens, pt should equal
    # flat_loc.numel(); otherwise the allocator returned extra padding
    # which we silently ignore here.
