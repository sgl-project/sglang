"""DSV4-NPU SWA + c4/c128 paged allocator.

Subclasses :class:`SWATokenToKVPoolAllocator` to add paged allocation for
the c4 and c128 compressed-KV pools alongside the existing full + SWA
allocators. Replaces the legacy in-pool free-list page allocator (see
commit ``8c1e87b``).

Allocation flow:
  1. ``alloc_extend`` / ``alloc_decode`` calls super() to allocate full +
     SWA pool slots, just like today (returns ``out_full_loc`` tensor for
     signature compatibility).
  2. Compute c4/c128 extend counts from seq_lens_cpu / prefix_lens_cpu
     (one compressed token per ``ratio`` raw tokens, gated on
     ``seq_len % ratio == 0`` at the chunk boundary).
  3. Allocate the corresponding number of c4/c128 slots via the standard
     :class:`NPUPagedTokenToKVPoolAllocator` driven against the
     ``c4_kv_pool`` / ``c128_kv_pool`` sub-buffers of
     :class:`DeepSeekV4TokenToKVPool`.
  4. Allocate the c4/c128 attention + indexer state-pool slots the same
     way (paged ``NPUPagedTokenToKVPoolAllocator`` against the
     NPUCompressStatePool buffers), tail-only per req.
  5. Bundle full / swa / c4 / c128 KV slots AND the c4/c128 state slots
     into a :class:`DSV4OutCacheLoc` and stash on ``self._last_dsv4_alloc``.

State pool slots ARE part of the bundle (``out_c{4,128}_state_loc``): the
NPU fused compressor uses a paged state pool (``cache_mode=1``), so state
slots come from the paged allocator, NOT from a ring-buffer hash. (The
base class' ``translate_kv_loc_to_compress_state_loc`` ring-hash is the
CUDA-only path and is disabled on NPU — see dsv4_memory_pool.py.)

mem_cache/common.py fetches the bundle via ``get_last_dsv4_alloc()``
after each alloc, stashes it on ``batch.out_cache_loc_dsv4``, and writes
the per-req tables (``req_to_token_c{4,128}`` and the per-token
``req_to_token_c{4,128}_state``) on the DSV4NPUReqToTokenPool.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    """Return the slot id of the last token already allocated for each req,
    or -1 when ``prefix_lens[i] == 0`` (fresh req with no prior allocation).

    Mirrors ``iforgetmyname/sglang dsv4_release get_last_loc_torch`` —
    looks up ``req_to_token[req, prefix_lens-1]``. Used by the c-pool
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
        # via DeepSeekV4TokenToKVPool.get_attention/indexer_compress_state_cache
        # (which routes to ``compress_state_pools[layer_id]``). The allocator
        # only needs a KVCache-conforming object for its constructor; the
        # actual slot allocation is layer-agnostic.
        #
        # ``c{4,128}_state_pool_size`` is the ALLOCATOR size (= num_usable_pages
        # * page_size, output of :func:`npu_state_pool_size`); the underlying
        # NPUCompressStatePool buffer over-allocates one page so the
        # ``free_pages = arange(1, num_pages+1)`` convention works without
        # OOB. Each pool has its own state slot space (kv attention vs
        # indexer) so allocations are independent.
        self.c4_state_attn_allocator: Optional[NPUPagedTokenToKVPoolAllocator] = None
        self.c128_state_attn_allocator: Optional[NPUPagedTokenToKVPoolAllocator] = None
        self.c4_index_state_attn_allocator: Optional[
            NPUPagedTokenToKVPoolAllocator
        ] = None
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
            c4_index_state_pool = next(
                (
                    p
                    for ratio, p in zip(
                        kvcache.compression_ratios,
                        kvcache.indexer_compress_state_pools,
                    )
                    if ratio == 4 and p is not None
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
            if c4_index_state_pool is not None and kvcache.c4_state_pool_size > 0:
                self.c4_index_state_attn_allocator = NPUPagedTokenToKVPoolAllocator(
                    kvcache.c4_state_pool_size,
                    page_size=page_size,
                    dtype=dtype,
                    device=device,
                    kvcache=c4_index_state_pool,
                    need_sort=need_sort,
                )

        # Cached zero-length int64 tensor returned by _alloc_c_extend when a
        # batch produces no new compressed tokens this step (e.g. decode of
        # reqs that haven't yet closed a ratio chunk). Avoids per-step
        # torch.empty churn.
        self._empty_loc = torch.empty((0,), dtype=torch.int64, device=device)
        self._last_dsv4_alloc: Optional[DSV4OutCacheLoc] = None

        # Back-reference to DSV4NPUReqToTokenPool, wired by
        # ``register_req_to_token_pool`` (called from the pool's
        # ``register_dsv4_allocator``). Needed so ``_alloc_c_extend`` can
        # look up real c-pool last_loc via ``get_last_loc`` on the
        # per-req ``req_to_token_c{4,128}`` table. Stays None until
        # registration; ``_alloc_c_extend`` asserts non-None.
        self._req_to_token_pool = None

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

    def register_req_to_token_pool(self, req_to_token_pool) -> None:
        """Wire the DSV4NPUReqToTokenPool back-ref so ``_alloc_c_extend``
        / ``_alloc_state_extend`` can look up the real per-pool last_loc
        from the per-req ``req_to_token_c{4,128}`` and
        ``req_to_token_c{4,128}_state`` tables. Called from
        :meth:`DSV4NPUReqToTokenPool.register_dsv4_allocator` right after
        the pool object is constructed."""
        self._req_to_token_pool = req_to_token_pool

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

        assert self._req_to_token_pool is not None, (
            "DSV4NPUTokenToKVPoolAllocator.register_req_to_token_pool was not "
            "called — state-pool last_loc lookup needs the per-req table "
            "back-ref. Wire it from DSV4NPUReqToTokenPool.register_dsv4_allocator."
        )
        state_table = (
            self._req_to_token_pool.req_to_token_c4_state
            if ratio == 4
            else self._req_to_token_pool.req_to_token_c128_state
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

        The previous implementation hard-coded ``last_loc=-1`` everywhere,
        which forced every cross-boundary decode to open a fresh c-pool
        page; later writes for one req's consecutive compressed tokens
        ended up scattered across many pages, breaking the kernel's
        assumption that page-aligned chunks of ``page_size`` compressed
        positions live in one physical page. Symptom: long-output AIME
        decoded into garbage past raw seq_len ~= 512.
        """
        c_prefix = (prefix_lens // ratio).to(prefix_lens.dtype)
        c_seq = (seq_lens // ratio).to(seq_lens.dtype)
        c_prefix_cpu = prefix_lens_cpu // ratio
        c_seq_cpu = seq_lens_cpu // ratio
        c_extend = self._compute_c_extend_counts(prefix_lens_cpu, seq_lens_cpu, ratio)
        if c_extend == 0:
            return self._empty_loc

        assert self._req_to_token_pool is not None, (
            "DSV4NPUTokenToKVPoolAllocator.register_req_to_token_pool was not "
            "called — c-pool last_loc lookup needs the per-req table back-ref. "
            "Wire it from DSV4NPUReqToTokenPool.register_dsv4_allocator."
        )
        c_table = (
            self._req_to_token_pool.req_to_token_c4
            if ratio == 4
            else self._req_to_token_pool.req_to_token_c128
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
        c4_state_prefix_lens: Optional[torch.Tensor] = None,
        c4_state_prefix_lens_cpu: Optional[torch.Tensor] = None,
        c4_state_seq_lens: Optional[torch.Tensor] = None,
        c4_state_seq_lens_cpu: Optional[torch.Tensor] = None,
        c4_state_extend_num_tokens: Optional[int] = None,
        c128_state_prefix_lens: Optional[torch.Tensor] = None,
        c128_state_prefix_lens_cpu: Optional[torch.Tensor] = None,
        c128_state_seq_lens: Optional[torch.Tensor] = None,
        c128_state_seq_lens_cpu: Optional[torch.Tensor] = None,
        c128_state_extend_num_tokens: Optional[int] = None,
    ):
        out_full_loc = super().alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        if out_full_loc is None:
            self._last_dsv4_alloc = None
            return None

        out_swa_loc = self.translate_loc_from_full_to_swa(out_full_loc)
        assert out_swa_loc is not None, (
            "translate_loc_from_full_to_swa returned None — "
            "full_to_swa_index_mapping not initialized?"
        )

        # req_pool_indices is REQUIRED for the c-pool last_loc lookup.
        # mem_cache/common.py:alloc_paged_token_slots_extend passes it via
        # the hasattr-gated extra_kwargs path when the allocator is a
        # DSV4NPUTokenToKVPoolAllocator instance.
        assert req_pool_indices is not None, (
            "DSV4NPUTokenToKVPoolAllocator.alloc_extend requires req_pool_indices. "
            "Caller (alloc_paged_token_slots_extend) must forward "
            "batch.req_pool_indices."
        )

        out_c4_loc = self._alloc_c_extend(
            self.c4_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            req_pool_indices,
            last_loc.dtype,
            ratio=4,
        )
        out_c128_loc = self._alloc_c_extend(
            self.c128_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            req_pool_indices,
            last_loc.dtype,
            ratio=128,
        )

        # State-pool extends: tail-only per-req allocation. ScheduleBatch's
        # ``_compute_dsv4_state_lens_extend`` precomputes per-pool prefix /
        # seq tensors (in state-pool cumulative slot space) and the sum
        # extend_num_tokens. We pass them through to ``_alloc_state_extend``
        # which feeds the underlying paged allocator.
        assert c4_state_seq_lens is not None and c128_state_seq_lens is not None, (
            "DSV4NPUTokenToKVPoolAllocator.alloc_extend requires "
            "c{4,128}_state_{prefix,seq}_lens kwargs (set by "
            "ScheduleBatch._compute_dsv4_state_lens_extend / forwarded by "
            "alloc_paged_token_slots_extend's dsv4_state_kwargs)."
        )
        out_c4_state_loc = self._alloc_state_extend(
            self.c4_state_attn_allocator,
            prefix_lens,
            c4_state_prefix_lens, c4_state_prefix_lens_cpu,
            c4_state_seq_lens, c4_state_seq_lens_cpu,
            req_pool_indices,
            last_loc.dtype,
            c4_state_extend_num_tokens,
            ratio=4,
        )
        out_c128_state_loc = self._alloc_state_extend(
            self.c128_state_attn_allocator,
            prefix_lens,
            c128_state_prefix_lens, c128_state_prefix_lens_cpu,
            c128_state_seq_lens, c128_state_seq_lens_cpu,
            req_pool_indices,
            last_loc.dtype,
            c128_state_extend_num_tokens,
            ratio=128,
        )

        self._last_dsv4_alloc = DSV4OutCacheLoc(
            out_full_loc=out_full_loc,
            out_swa_loc=out_swa_loc,
            out_c4_loc=out_c4_loc,
            out_c128_loc=out_c128_loc,
            out_c4_state_loc=out_c4_state_loc,
            out_c128_state_loc=out_c128_state_loc,
        )
        return out_full_loc

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        *,
        req_pool_indices: Optional[torch.Tensor] = None,
        c4_state_prefix_lens: Optional[torch.Tensor] = None,
        c4_state_prefix_lens_cpu: Optional[torch.Tensor] = None,
        c4_state_seq_lens: Optional[torch.Tensor] = None,
        c4_state_seq_lens_cpu: Optional[torch.Tensor] = None,
        c4_state_extend_num_tokens: Optional[int] = None,
        c128_state_prefix_lens: Optional[torch.Tensor] = None,
        c128_state_prefix_lens_cpu: Optional[torch.Tensor] = None,
        c128_state_seq_lens: Optional[torch.Tensor] = None,
        c128_state_seq_lens_cpu: Optional[torch.Tensor] = None,
        c128_state_extend_num_tokens: Optional[int] = None,
    ):
        out_full_loc = super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        if out_full_loc is None:
            self._last_dsv4_alloc = None
            return None

        out_swa_loc = self.translate_loc_from_full_to_swa(out_full_loc)

        # For decode, we add one token per req. A new compressed-K token
        # is emitted when seq_lens[i] % ratio == 0 (post-decode). Model
        # this as an extend from (seq_lens-1)//ratio to seq_lens//ratio;
        # _alloc_c_extend looks up real c-pool last_loc from req_to_token_c{ratio}
        # via get_last_loc, ensuring intra-page slot continuity.
        prefix_lens = (seq_lens - 1).clamp(min=0)
        prefix_lens_cpu = (seq_lens_cpu - 1).clamp(min=0)

        assert req_pool_indices is not None, (
            "DSV4NPUTokenToKVPoolAllocator.alloc_decode requires req_pool_indices. "
            "Caller (alloc_paged_token_slots_decode) must forward "
            "batch.req_pool_indices."
        )

        out_c4_loc = self._alloc_c_extend(
            self.c4_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            req_pool_indices,
            last_loc.dtype,
            ratio=4,
        )
        out_c128_loc = self._alloc_c_extend(
            self.c128_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            req_pool_indices,
            last_loc.dtype,
            ratio=128,
        )

        # State-pool decode: per-req cumulative slot lens precomputed by
        # ScheduleBatch._compute_dsv4_state_lens_decode (each req +1 slot per
        # pool). Raw prefix_lens (pre-decode seq_lens) drives the
        # last_loc lookup at req_to_token_c{N}_state[req, pre_decode_seq_len-1].
        assert c4_state_seq_lens is not None and c128_state_seq_lens is not None, (
            "DSV4NPUTokenToKVPoolAllocator.alloc_decode requires "
            "c{4,128}_state_{prefix,seq}_lens kwargs (set by "
            "ScheduleBatch._compute_dsv4_state_lens_decode / forwarded by "
            "alloc_paged_token_slots_decode's dsv4_state_kwargs)."
        )
        out_c4_state_loc = self._alloc_state_extend(
            self.c4_state_attn_allocator,
            prefix_lens,
            c4_state_prefix_lens, c4_state_prefix_lens_cpu,
            c4_state_seq_lens, c4_state_seq_lens_cpu,
            req_pool_indices,
            last_loc.dtype,
            c4_state_extend_num_tokens,
            ratio=4,
        )
        out_c128_state_loc = self._alloc_state_extend(
            self.c128_state_attn_allocator,
            prefix_lens,
            c128_state_prefix_lens, c128_state_prefix_lens_cpu,
            c128_state_seq_lens, c128_state_seq_lens_cpu,
            req_pool_indices,
            last_loc.dtype,
            c128_state_extend_num_tokens,
            ratio=128,
        )

        self._last_dsv4_alloc = DSV4OutCacheLoc(
            out_full_loc=out_full_loc,
            out_swa_loc=out_swa_loc,
            out_c4_loc=out_c4_loc,
            out_c128_loc=out_c128_loc,
            out_c4_state_loc=out_c4_state_loc,
            out_c128_state_loc=out_c128_state_loc,
        )
        return out_full_loc

    def get_last_dsv4_alloc(self) -> Optional[DSV4OutCacheLoc]:
        return self._last_dsv4_alloc

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
            if self.c4_index_state_attn_allocator is not None:
                # Indexer state slot ids share the c4_state table (NPU
                # convention); separate allocator pool space, freed in
                # parallel.
                self.c4_index_state_attn_allocator.free(
                    c4_state_slots.to(torch.int64)
                )
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
        # SWATokenToKVPoolAllocator.__init__ calls self.clear() at line 375,
        # BEFORE our __init__ has created the c4/c128 sub-allocators. Guard
        # against that early call — the sub-allocators are freshly initialized
        # when __init__ later constructs them, so there's nothing to clear yet.
        if hasattr(self, "c4_attn_allocator"):
            self.c4_attn_allocator.clear()
        if hasattr(self, "c128_attn_allocator"):
            self.c128_attn_allocator.clear()
        # State allocators may be None (no c{N} layers in this model config).
        if getattr(self, "c4_state_attn_allocator", None) is not None:
            self.c4_state_attn_allocator.clear()
        if getattr(self, "c128_state_attn_allocator", None) is not None:
            self.c128_state_attn_allocator.clear()
        if getattr(self, "c4_index_state_attn_allocator", None) is not None:
            self.c4_index_state_attn_allocator.clear()
        self._last_dsv4_alloc = None
