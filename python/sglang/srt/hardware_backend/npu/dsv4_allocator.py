"""DSV4-NPU SWA + c4/c128 paged allocator.

Subclasses :class:`SWATokenToKVPoolAllocator` to add paged allocation for
the c4 and c128 compressed-KV pools alongside the existing full + SWA
allocators. Replaces the legacy in-pool free-list page allocator.

Allocation flow (``alloc_extend`` / ``alloc_decode``):
  1. super() allocates full + SWA pool slots (``out_full_loc``).
  2. Compute c4/c128 extend counts from seq_lens_cpu / prefix_lens_cpu
     (one compressed token per ``ratio`` raw tokens, gated on
     ``seq_len % ratio == 0`` at the chunk boundary) and allocate that many
     c4/c128 KV slots via the standard :class:`NPUPagedTokenToKVPoolAllocator`
     driven against the ``c4_kv_pool`` / ``c128_kv_pool`` sub-buffers of
     :class:`DeepSeekV4TokenToKVPool`.
  3. Allocate the c4/c128 **state**-pool slots the same way (paged allocator
     against the NPUCompressStatePool buffers), tail-only per req, using the
     per-req lens packed in the ``DSV4StateLens`` the scheduler built.
  4. **Return** a :class:`DSV4OutCacheLoc` bundling full / swa / c4 / c128 KV
     slots AND the c4/c128 state slots.

Naming map (one pool family per concept):
  * ``c4_kv_pool`` / ``c128_kv_pool``         — compressed-KV pools (DeepSeekV4SingleKVPool)
  * ``c4_indexer_kv_pool``                    — c4 lightning-indexer K pool (DeepSeekV4IndexerPool)
  * ``compress_state_pools[layer]``           — attention compress-state pools (NPUCompressStatePool)
  * ``indexer_compress_state_pools[layer]``   — c4 indexer compress-state pools
  * ``c{4,128}_attn_allocator``               — paged allocators for the c4/c128 KV pools
  * ``c{4,128}_state_attn_allocator``         — paged allocators for the attention state pools

State slots are part of the bundle (``out_c{4,128}_state_loc``): the NPU
fused compressor uses a paged state pool (``cache_mode=1``), so state slots
come from the paged allocator, NOT from a ring-buffer hash. (The base
class' ``translate_kv_loc_to_compress_state_loc`` ring-hash is the CUDA-only
path and is disabled on NPU — see dsv4_memory_pool.py.)

The bundle is the explicit return value (no side-channel): mem_cache/common.py
unpacks ``out_full_loc`` for its generic return, stashes the bundle on
``batch.out_cache_loc_dsv4``, and writes the per-req tables
(``req_to_token_c{4,128}`` and the per-token ``req_to_token_c{4,128}_state``)
on the DSV4NPUReqToTokenPool.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc, DSV4StateLens


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    """Return the slot id of the last token already allocated for each req,
    or -1 when ``prefix_lens[i] == 0`` (fresh req with no prior allocation).

    Looks up ``req_to_token[req, prefix_lens-1]``. Used by the c-pool
    branch of :class:`DSV4NPUTokenToKVPoolAllocator` to anchor the paged
    allocator's ``alloc_extend`` on the real previous tail slot rather
    than the broken ``-1``-everywhere sentinel that the legacy code used
    (which forced every cross-boundary decode to open a fresh c-pool
    page, violating intra-page slot continuity assumed by the kernel's
    ``cmp_block_table``).

    Result dtype matches ``prefix_lens`` dtype (matches the caller's
    last_loc convention; paged allocator's ``alloc_extend`` debug assert
    requires ``(last_loc + 1) % page_size == prefix_lens % page_size``).
    """
    req_pool_indices = req_pool_indices.to(torch.int64)
    safe_idx = (prefix_lens.to(torch.int64) - 1).clamp(min=0)
    looked_up = req_to_token[req_pool_indices, safe_idx].to(prefix_lens.dtype)
    return torch.where(
        prefix_lens > 0,
        looked_up,
        torch.full_like(prefix_lens, -1),
    )


class DSV4NPUTokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """SWA allocator + c4/c128 paged allocators for DSV4 on NPU."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache,
        need_sort: bool,
    ):
        super().__init__(
            size=size,
            size_swa=size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

        # Reach into the DSV4 pool for the c4 / c128 sub-buffers. The sub-pool
        # objects implement KVCache so they slot into NPUPagedTokenToKVPoolAllocator
        # directly.
        c4_kv_pool = kvcache.c4_kv_pool
        c128_kv_pool = kvcache.c128_kv_pool

        # c4_size / c128_size live on the parent V4 pool. They are tracked
        # in raw "compressed tokens" units (one slot per compressed token);
        # the paged allocator handles the page-size granularity.
        self.c4_attn_allocator = NPUPagedTokenToKVPoolAllocator(
            kvcache.c4_size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=c4_kv_pool,
            need_sort=need_sort,
        )
        self.c128_attn_allocator = NPUPagedTokenToKVPoolAllocator(
            kvcache.c128_size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=c128_kv_pool,
            need_sort=need_sort,
        )

        # State pool allocators (NPU-only paged path). Layer 0's attention /
        # indexer state pools serve as the KVCache pointer here — all layers
        # share the same per-pool size budget driven by
        # ``c{4,128}_state_pool_size``, and per-layer buffer access happens
        # via DeepSeekV4TokenToKVPool.get_state_cache
        # (which routes to ``compress_state_pools[layer_id]``). The allocator
        # only needs a KVCache-conforming object for its constructor; the
        # actual slot allocation is layer-agnostic.
        #
        # ``c{4,128}_state_pool_size`` is the ALLOCATOR size (= num_usable_pages
        # * page_size, output of :func:`npu_state_pool_size`); the underlying
        # NPUCompressStatePool buffer over-allocates one page so the
        # ``free_pages = arange(1, num_pages+1)`` convention works without
        # OOB. Each pool has its own state slot space so allocations are
        # independent.
        self.c4_state_attn_allocator: Optional[NPUPagedTokenToKVPoolAllocator] = None
        self.c128_state_attn_allocator: Optional[NPUPagedTokenToKVPoolAllocator] = None
        if (
            getattr(kvcache, "compress_state_pools", None) is not None
            and len(kvcache.compress_state_pools) > 0
        ):
            c4_state_pool = next(
                (
                    p
                    for ratio, p in zip(
                        kvcache.compression_ratios, kvcache.compress_state_pools
                    )
                    if ratio == 4 and p is not None
                ),
                None,
            )
            c128_state_pool = next(
                (
                    p
                    for ratio, p in zip(
                        kvcache.compression_ratios, kvcache.compress_state_pools
                    )
                    if ratio == 128 and p is not None
                ),
                None,
            )
            if c4_state_pool is not None and kvcache.c4_state_pool_size > 0:
                self.c4_state_attn_allocator = NPUPagedTokenToKVPoolAllocator(
                    kvcache.c4_state_pool_size,
                    page_size=page_size,
                    dtype=dtype,
                    device=device,
                    kvcache=c4_state_pool,
                    need_sort=need_sort,
                )
            if c128_state_pool is not None and kvcache.c128_state_pool_size > 0:
                self.c128_state_attn_allocator = NPUPagedTokenToKVPoolAllocator(
                    kvcache.c128_state_pool_size,
                    page_size=page_size,
                    dtype=dtype,
                    device=device,
                    kvcache=c128_state_pool,
                    need_sort=need_sort,
                )

        # Cached zero-length int64 tensor returned by _alloc_c_extend when a
        # batch produces no new compressed tokens this step (e.g. decode of
        # reqs that haven't yet closed a ratio chunk). Avoids per-step
        # torch.empty churn.
        self._empty_loc = torch.empty((0,), dtype=torch.int64, device=device)

        # The c-pool / state-pool last_loc lookup needs the per-req
        # ``req_to_token_c{4,128}[_state]`` tables. Rather than a permanent
        # two-way binding with DSV4NPUReqToTokenPool, common.py passes the
        # pool into ``alloc_extend`` / ``alloc_decode`` per call, which stash
        # it here for the duration of that call (read by ``_alloc_c_extend`` /
        # ``_alloc_state_extend``). None outside an alloc call.
        self._cur_req_to_token_pool = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_c_extend_counts(
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        ratio: int,
    ) -> int:
        """Number of new compressed-K tokens this extend produces across the
        batch, at compression ``ratio``. Equals
        ``sum_i (seq_lens[i] // ratio - prefix_lens[i] // ratio)``.
        """
        if prefix_lens_cpu is None or seq_lens_cpu is None:
            return 0
        # Both tensors are host-side ints; do the arithmetic on CPU.
        diff = (seq_lens_cpu // ratio) - (prefix_lens_cpu // ratio)
        diff = diff.clamp(min=0)
        return int(diff.sum().item())

    def _alloc_state_extend(
        self,
        allocator: Optional[NPUPagedTokenToKVPoolAllocator],
        raw_prefix_lens: torch.Tensor,
        state_prefix_lens: torch.Tensor,
        state_prefix_lens_cpu: torch.Tensor,
        state_seq_lens: torch.Tensor,
        state_seq_lens_cpu: torch.Tensor,
        req_pool_indices: torch.Tensor,
        last_loc_dtype: torch.dtype,
        state_extend_num_tokens: int,
        ratio: int,
    ) -> torch.Tensor:
        """Allocate state-pool slots for an extend at ``ratio``.

        State pool is a **separate paged slot space** from raw token
        positions: each req allocates only the trailing window
        (``c4_alloc_len`` / ``c128_alloc_len`` per-req, computed by
        ``ScheduleBatch._compute_dsv4_state_lens_extend``). The state pool's
        cumulative slot count for a req grows monotonically across
        extends/decodes; this method takes ``state_prefix_lens`` (prev
        cumulative count) and ``state_seq_lens`` (new cumulative count) for
        the underlying paged allocator.

        ``state_last_loc`` is looked up from ``req_to_token_c{ratio}_state``
        at raw position ``raw_prefix_lens - 1`` — the LAST raw position
        already populated by the previous extend / decode step. For fresh
        reqs (raw_prefix_lens == 0), ``get_last_loc`` returns -1 (its
        sentinel for "no prior allocation"). For decode, caller passes
        ``raw_prefix_lens = pre_decode_seq_lens`` so the lookup hits the
        position of the previous decode's token.

        Returns ``self._empty_loc`` when the allocator is not initialized
        (e.g., ``c{ratio}_state_pool_size == 0`` — DSV4 model has no
        c{ratio} layers) or when ``state_extend_num_tokens == 0``.
        """
        if allocator is None or state_extend_num_tokens == 0:
            return self._empty_loc

        assert self._cur_req_to_token_pool is not None, (
            "DSV4NPUTokenToKVPoolAllocator.alloc_extend/alloc_decode must be "
            "called with req_to_token_pool= (forwarded by "
            "alloc_paged_token_slots_*); state-pool last_loc lookup needs the "
            "per-req tables."
        )
        state_table = (
            self._cur_req_to_token_pool.req_to_token_c4_state
            if ratio == 4
            else self._cur_req_to_token_pool.req_to_token_c128_state
        )
        # last_loc lookup uses RAW position (table is indexed by raw token
        # position, holding state-pool slot ids at populated tail positions).
        # ``raw_prefix_lens - 1`` = last position of previous extend / decode.
        state_last_loc = get_last_loc(
            state_table, req_pool_indices, raw_prefix_lens
        ).to(last_loc_dtype)

        result = allocator.alloc_extend(
            state_prefix_lens,
            state_prefix_lens_cpu,
            state_seq_lens,
            state_seq_lens_cpu,
            state_last_loc,
            state_extend_num_tokens,
        )
        if result is None:
            raise RuntimeError(
                f"DSV4 c{ratio} state pool exhausted: need "
                f"{state_extend_num_tokens} new slots, "
                f"available={allocator.available_size()}. Either raise "
                f"--mem-fraction-static, lower --max-running-requests, or "
                f"check that DSV4NPUTokenToKVPoolAllocator.free(req=...) is "
                f"releasing state slots on req finish."
            )
        return result

    def _alloc_c_extend(
        self,
        allocator: NPUPagedTokenToKVPoolAllocator,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices: torch.Tensor,
        last_loc_dtype: torch.dtype,
        ratio: int,
    ) -> torch.Tensor:
        """Allocate compressed slots for an extend operation at ``ratio``.

        Per-request prefix / seq lengths are translated from raw token
        units to compressed units by integer-dividing by ``ratio``. The
        c-pool last_loc is looked up from ``req_to_token_c{ratio}`` table
        via :func:`get_last_loc` — this is the slot id of the previous
        c-pool token for each req (or -1 for fresh reqs with no prior
        c-pool allocation). The paged allocator then either continues
        in the same page (slot = last_loc + 1) or opens a new page (at
        ratio boundary), preserving intra-page slot continuity which the
        kernel's ``cmp_block_table`` reader relies on.
        """
        c_prefix = (prefix_lens // ratio).to(prefix_lens.dtype)
        c_seq = (seq_lens // ratio).to(seq_lens.dtype)
        c_prefix_cpu = prefix_lens_cpu // ratio
        c_seq_cpu = seq_lens_cpu // ratio
        c_extend = self._compute_c_extend_counts(prefix_lens_cpu, seq_lens_cpu, ratio)
        if c_extend == 0:
            return self._empty_loc

        assert self._cur_req_to_token_pool is not None, (
            "DSV4NPUTokenToKVPoolAllocator.alloc_extend/alloc_decode must be "
            "called with req_to_token_pool= (forwarded by "
            "alloc_paged_token_slots_*); c-pool last_loc lookup needs the "
            "per-req tables."
        )
        c_table = (
            self._cur_req_to_token_pool.req_to_token_c4
            if ratio == 4
            else self._cur_req_to_token_pool.req_to_token_c128
        )
        c_last_loc = get_last_loc(c_table, req_pool_indices, c_prefix).to(
            last_loc_dtype
        )

        result = allocator.alloc_extend(
            c_prefix,
            c_prefix_cpu,
            c_seq,
            c_seq_cpu,
            c_last_loc,
            c_extend,
        )
        if result is None:
            raise RuntimeError(
                f"DSV4 c{ratio} pool exhausted: need {c_extend} new slots, "
                f"available={allocator.available_size()}. "
                f"Reduce --max-running-requests or raise --mem-fraction-static, "
                f"or check that DSV4NPUTokenToKVPoolAllocator.free(req=...) is "
                f"being driven from DSV4NPUReqToTokenPool.free."
            )
        return result

    def _alloc_c_and_state(
        self,
        out_full_loc: torch.Tensor,
        out_swa_loc: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc_dtype: torch.dtype,
        req_pool_indices: Optional[torch.Tensor],
        dsv4_state_lens: Optional[DSV4StateLens],
    ) -> DSV4OutCacheLoc:
        """Allocate c4/c128 KV + state slots and bundle them with full/swa loc.

        Shared by alloc_extend / alloc_decode; those differ only in how
        prefix_lens is derived and whether out_swa_loc is asserted.
        """
        assert req_pool_indices is not None, (
            "DSV4NPUTokenToKVPoolAllocator requires req_pool_indices; "
            "alloc_paged_token_slots_* must forward batch.req_pool_indices."
        )
        out_c4_loc = self._alloc_c_extend(
            self.c4_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            req_pool_indices,
            last_loc_dtype,
            ratio=4,
        )
        out_c128_loc = self._alloc_c_extend(
            self.c128_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            req_pool_indices,
            last_loc_dtype,
            ratio=128,
        )
        # State-pool: tail-only per-req allocation; lens precomputed by
        # ScheduleBatch._compute_dsv4_state_lens_* (packed into DSV4StateLens).
        # Raw prefix_lens drives the state last_loc lookup.
        assert dsv4_state_lens is not None, (
            "DSV4NPUTokenToKVPoolAllocator requires dsv4_state_lens "
            "(ScheduleBatch._compute_dsv4_state_lens_*, forwarded by "
            "alloc_paged_token_slots_*)."
        )
        out_c4_state_loc = self._alloc_state_extend(
            self.c4_state_attn_allocator,
            prefix_lens,
            dsv4_state_lens.c4_prefix_lens, dsv4_state_lens.c4_prefix_lens_cpu,
            dsv4_state_lens.c4_seq_lens, dsv4_state_lens.c4_seq_lens_cpu,
            req_pool_indices,
            last_loc_dtype,
            dsv4_state_lens.c4_extend_num_tokens,
            ratio=4,
        )
        out_c128_state_loc = self._alloc_state_extend(
            self.c128_state_attn_allocator,
            prefix_lens,
            dsv4_state_lens.c128_prefix_lens, dsv4_state_lens.c128_prefix_lens_cpu,
            dsv4_state_lens.c128_seq_lens, dsv4_state_lens.c128_seq_lens_cpu,
            req_pool_indices,
            last_loc_dtype,
            dsv4_state_lens.c128_extend_num_tokens,
            ratio=128,
        )
        return DSV4OutCacheLoc(
            out_full_loc=out_full_loc,
            out_swa_loc=out_swa_loc,
            out_c4_loc=out_c4_loc,
            out_c128_loc=out_c128_loc,
            out_c4_state_loc=out_c4_state_loc,
            out_c128_state_loc=out_c128_state_loc,
        )

    # ------------------------------------------------------------------
    # alloc overrides
    # ------------------------------------------------------------------

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        *,
        req_pool_indices: Optional[torch.Tensor] = None,
        dsv4_state_lens: Optional[DSV4StateLens] = None,
        req_to_token_pool=None,
    ) -> Optional[DSV4OutCacheLoc]:
        # Stash the per-req tables for this call's c-pool / state last_loc
        # lookups (read by _alloc_c_extend / _alloc_state_extend). Passed in
        # by alloc_paged_token_slots_extend — no permanent allocator->pool ref.
        self._cur_req_to_token_pool = req_to_token_pool
        out_full_loc = super().alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        if out_full_loc is None:
            return None

        out_swa_loc = self.translate_loc_from_full_to_swa(out_full_loc)
        assert out_swa_loc is not None, (
            "translate_loc_from_full_to_swa returned None — "
            "full_to_swa_index_mapping not initialized?"
        )

        return self._alloc_c_and_state(
            out_full_loc,
            out_swa_loc,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc.dtype,
            req_pool_indices,
            dsv4_state_lens,
        )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        *,
        req_pool_indices: Optional[torch.Tensor] = None,
        dsv4_state_lens: Optional[DSV4StateLens] = None,
        req_to_token_pool=None,
    ) -> Optional[DSV4OutCacheLoc]:
        # See alloc_extend: stash per-req tables for this call's last_loc lookups.
        self._cur_req_to_token_pool = req_to_token_pool
        out_full_loc = super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        if out_full_loc is None:
            return None

        out_swa_loc = self.translate_loc_from_full_to_swa(out_full_loc)

        # For decode, we add one token per req. A new compressed-K token
        # is emitted when seq_lens[i] % ratio == 0 (post-decode). Model
        # this as an extend from (seq_lens-1)//ratio to seq_lens//ratio;
        # _alloc_c_extend looks up real c-pool last_loc from req_to_token_c{ratio}
        # via get_last_loc, ensuring intra-page slot continuity.
        prefix_lens = (seq_lens - 1).clamp(min=0)
        prefix_lens_cpu = (seq_lens_cpu - 1).clamp(min=0)

        return self._alloc_c_and_state(
            out_full_loc,
            out_swa_loc,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc.dtype,
            req_pool_indices,
            dsv4_state_lens,
        )

    # ------------------------------------------------------------------
    # Free / clear
    # ------------------------------------------------------------------

    def c4_available_size(self):
        return self.c4_attn_allocator.available_size()

    def c128_available_size(self):
        return self.c128_attn_allocator.available_size()

    def c4_state_available_size(self) -> int:
        if self.c4_state_attn_allocator is None:
            return 0
        return self.c4_state_attn_allocator.available_size()

    def c128_state_available_size(self) -> int:
        if self.c128_state_attn_allocator is None:
            return 0
        return self.c128_state_attn_allocator.available_size()

    def free(
        self,
        free_index: Optional[torch.Tensor] = None,
        *,
        req=None,
        req_to_token_pool=None,
    ):
        """Unified free path for full/swa/c4/c128 pools.

        Two callable forms:
          * ``free(free_index)`` — legacy SWA path. Releases full + SWA
            slots only. Used by tail eviction, radix eviction, etc. (no
            req identity available there, so c-pool free can't fire).
          * ``free(req=req, req_to_token_pool=pool)`` — invoked from
            :meth:`DSV4NPUReqToTokenPool.free` when a request finishes.
            Reads the per-req c-pool slot lists from
            ``req_to_token_c{4,128}[req.req_pool_idx, :kv_committed_len // ratio]``
            and hands them to the c-pool
            :class:`NPUPagedTokenToKVPoolAllocator.free`, which dedupes
            by page and returns whole pages to the free list.

        Both forms can fire in the same call (free_index + req kwargs);
        each is processed independently.
        """
        if free_index is not None:
            super().free(free_index)

        if req is None or req_to_token_pool is None:
            return

        kv_len = req.kv_committed_len
        if kv_len <= 0:
            return
        req_pool_idx = req.req_pool_idx
        if req_pool_idx is None:
            return

        c4_n = kv_len // 4
        if c4_n > 0 and hasattr(req_to_token_pool, "req_to_token_c4"):
            c4_slots = req_to_token_pool.req_to_token_c4[req_pool_idx, :c4_n]
            # to int64 — paged allocator's free does cpu()//page_size on it.
            self.c4_attn_allocator.free(c4_slots.to(torch.int64))

        c128_n = kv_len // 128
        if c128_n > 0 and hasattr(req_to_token_pool, "req_to_token_c128"):
            c128_slots = req_to_token_pool.req_to_token_c128[req_pool_idx, :c128_n]
            self.c128_attn_allocator.free(c128_slots.to(torch.int64))

        # State pool slots are 1-per-raw-token. Free only the tail
        # ``[c{N}_state_alloc_offset, kv_len)`` — the prefix
        # ``[0, c{N}_state_alloc_offset)`` was already returned by
        # ScheduleBatch._evict_swa as SWA evicted the corresponding raw
        # KV slots (state ride-along eviction). Without that offset we
        # double-free, which the paged allocator's debug_mode assert
        # catches (and corrupts the free list either way).
        c4_state_off = getattr(req, "c4_state_alloc_offset", 0)
        c128_state_off = getattr(req, "c128_state_alloc_offset", 0)
        if (
            self.c4_state_attn_allocator is not None
            and hasattr(req_to_token_pool, "req_to_token_c4_state")
            and kv_len > c4_state_off
        ):
            c4_state_slots = req_to_token_pool.req_to_token_c4_state[
                req_pool_idx, c4_state_off:kv_len
            ]
            self.c4_state_attn_allocator.free(c4_state_slots.to(torch.int64))
        if (
            self.c128_state_attn_allocator is not None
            and hasattr(req_to_token_pool, "req_to_token_c128_state")
            and kv_len > c128_state_off
        ):
            c128_state_slots = req_to_token_pool.req_to_token_c128_state[
                req_pool_idx, c128_state_off:kv_len
            ]
            self.c128_state_attn_allocator.free(c128_state_slots.to(torch.int64))

    def clear(self):
        super().clear()
        # SWATokenToKVPoolAllocator.__init__ calls self.clear() BEFORE our
        # __init__ has created the c4/c128 sub-allocators. Guard against that
        # early call — the sub-allocators are freshly initialized when
        # __init__ later constructs them, so there's nothing to clear yet.
        if hasattr(self, "c4_attn_allocator"):
            self.c4_attn_allocator.clear()
        if hasattr(self, "c128_attn_allocator"):
            self.c128_attn_allocator.clear()
        # State allocators may be None (no c{N} layers in this model config).
        if getattr(self, "c4_state_attn_allocator", None) is not None:
            self.c4_state_attn_allocator.clear()
        if getattr(self, "c128_state_attn_allocator", None) is not None:
            self.c128_state_attn_allocator.clear()
