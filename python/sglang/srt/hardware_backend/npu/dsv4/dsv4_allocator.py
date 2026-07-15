"""DSV4-NPU SWA + c4/c128 paged allocator.

Subclasses :class:`SWATokenToKVPoolAllocator` and adds paged allocation for the
c4/c128 compressed-KV pools and their tail-only compress-state pools, alongside
the parent's full + SWA pools.

Per ``alloc_extend`` / ``alloc_decode``:
  1. super() allocates the full + SWA slots (``out_full_loc``).
  2. Allocate c4/c128 KV slots — one compressed token per ``ratio`` raw tokens
     (``seq_len // ratio - prefix_len // ratio``) — via the standard
     :class:`NPUPagedTokenToKVPoolAllocator` over the pool's c4/c128 KV buffers.
  3. Allocate the c4/c128 compress-state slots the same way, tail-only per req,
     using the per-req lens the scheduler packed into ``DSV4StateLens``.
  4. Return a :class:`DSV4OutCacheLoc` bundling all five slot families.

State slots are paged because the NPU fused compressor runs ``cache_mode=1``; the
base class' ``translate_kv_loc_to_compress_state_loc`` ring-hash is the CUDA-only
path and is unused on NPU. The bundle is the explicit return value:
mem_cache/common.py unpacks ``out_full_loc`` and stashes the bundle on
``batch.out_cache_loc_dsv4``; ``DSV4NPUReqToTokenPool`` writes the per-req
``req_to_token_c{4,128}[_state]`` tables that :meth:`free` and the last_loc
lookups read back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.hardware_backend.npu.allocator_npu import (
    NPUPagedTokenToKVPoolAllocator,
    NPUSWATokenToKVPoolAllocator,
    get_last_loc,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_write_dsv4_extend,
)
from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc, DSV4StateLens

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def alloc_paged_token_slots_extend_npu(
    *args,
    batch=None,
    dsv4_allocator: Optional[DSV4NPUTokenToKVPoolAllocator] = None,
    **kwargs,
):
    if batch is not None and is_deepseek_v4(batch.model_config.hf_config):
        assert dsv4_allocator is batch.token_to_kv_pool_allocator
        return alloc_paged_token_slots_reserve_extend(
            *args,
            batch=batch,
            dsv4_allocator=dsv4_allocator,
            **kwargs,
        )
    return alloc_paged_token_slots_extend(
        *args,
        batch=batch,
        dsv4_allocator=dsv4_allocator,
        **kwargs,
    )


def alloc_paged_token_slots_reserve_extend(
    tree_cache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    *,
    req_pool_indices: Optional[torch.Tensor] = None,
    dsv4_state_lens: Optional[DSV4StateLens] = None,
    dsv4_allocator: Optional[DSV4NPUTokenToKVPoolAllocator] = None,
    batch=None,
):
    """Allocate reserved draft slots and update DSV4 per-request tables."""
    if dsv4_state_lens is None and batch is not None:
        dsv4_state_lens = (
            dsv4_allocator.compute_dsv4_state_lens_reserve(
                batch.reqs, prefix_lens_cpu, seq_lens_cpu
            )
            if dsv4_allocator is not None
            else None
        )

    out_cache_loc = alloc_paged_token_slots_extend(
        tree_cache,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        req_pool_indices=req_pool_indices,
        dsv4_state_lens=dsv4_state_lens,
        dsv4_allocator=dsv4_allocator,
        batch=batch,
    )
    if batch is not None:
        maybe_write_dsv4_extend(
            batch,
            batch.req_pool_indices_cpu,
            prefix_lens_cpu,
            seq_lens_cpu,
            c4_state_alloc_offsets=prefix_lens_cpu,
            c128_state_alloc_offsets=prefix_lens_cpu,
        )
    return out_cache_loc


class DSV4NPUTokenToKVPoolAllocator(NPUSWATokenToKVPoolAllocator):
    """SWA allocator + c4/c128 KV and compress-state paged allocators for DSV4 on NPU."""

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

        def mk(pool_size, pool):
            # c4/c128 KV and state sub-pools implement KVCache, so they drop into
            # the standard paged allocator. pool_size is in compressed-token units.
            return NPUPagedTokenToKVPoolAllocator(
                pool_size,
                page_size=page_size,
                dtype=dtype,
                device=device,
                kvcache=pool,
                need_sort=need_sort,
            )

        self.c4_attn_allocator = mk(kvcache.c4_size, kvcache.c4_kv_pool)
        self.c128_attn_allocator = mk(kvcache.c128_size, kvcache.c128_kv_pool)

        # State allocators (paged, NPU-only). Any layer's pool works as KVCache
        # pointer (slot alloc is layer-agnostic); None when no c{ratio} layers or
        # zero budget.
        self.c4_state_attn_allocator: Optional[NPUPagedTokenToKVPoolAllocator] = None
        self.c128_state_attn_allocator: Optional[NPUPagedTokenToKVPoolAllocator] = None
        state_pools = getattr(kvcache, "compress_state_pools", None)
        if state_pools:

            def first_state_pool(want_ratio):
                return next(
                    (
                        p
                        for r, p in zip(kvcache.compression_ratios, state_pools)
                        if r == want_ratio and p is not None
                    ),
                    None,
                )

            c4_state_pool = first_state_pool(4)
            c128_state_pool = first_state_pool(128)
            if c4_state_pool is not None and kvcache.c4_state_pool_size > 0:
                self.c4_state_attn_allocator = mk(
                    kvcache.c4_state_pool_size, c4_state_pool
                )
            if c128_state_pool is not None and kvcache.c128_state_pool_size > 0:
                self.c128_state_attn_allocator = mk(
                    kvcache.c128_state_pool_size, c128_state_pool
                )

        # Returned by the c-pool helpers when a step adds no compressed tokens.
        self._empty_loc = torch.empty((0,), dtype=torch.int64, device=device)

        # Per-call handle to the DSV4NPUReqToTokenPool, stashed by alloc_extend/
        # alloc_decode for last_loc lookups; avoids a permanent allocator->pool ref.
        self._cur_req_to_token_pool = None

    @staticmethod
    def _compute_c_extend_counts(
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        ratio: int,
    ) -> int:
        """New compressed-K tokens this extend produces across the batch:
        ``sum_i (seq_lens[i] // ratio - prefix_lens[i] // ratio)``."""
        if prefix_lens_cpu is None or seq_lens_cpu is None:
            return 0
        diff = ((seq_lens_cpu // ratio) - (prefix_lens_cpu // ratio)).clamp(min=0)
        return int(diff.sum().item())

    @staticmethod
    def _pool_exhausted(
        ratio: int, kind: str, need: int, available: int
    ) -> RuntimeError:
        return RuntimeError(
            f"DSV4 c{ratio} {kind} pool exhausted: need {need} new slots, "
            f"available={available}. Raise --mem-fraction-static, lower "
            f"--max-running-requests, or check that "
            f"DSV4NPUTokenToKVPoolAllocator.free(req=...) releases {kind} slots "
            f"on req finish."
        )

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
        """Allocate tail-only state-pool slots for an extend at ``ratio``.

        The state pool is a separate paged slot space; each req allocates only
        its trailing window (cumulative lens precomputed by
        ``ScheduleBatch._compute_dsv4_state_lens_*`` and passed via
        ``DSV4StateLens``). ``state_last_loc`` is looked up from
        ``req_to_token_c{ratio}_state`` at the RAW position
        ``raw_prefix_lens - 1`` (the last position the previous extend/decode
        populated). Returns ``_empty_loc`` when the allocator is absent (no
        c{ratio} layers) or there is nothing to add.
        """
        if allocator is None or state_extend_num_tokens == 0:
            return self._empty_loc

        assert self._cur_req_to_token_pool is not None, (
            "alloc_extend/alloc_decode must be called with req_to_token_pool= "
            "for the state-pool last_loc lookup."
        )
        state_table = (
            self._cur_req_to_token_pool.req_to_token_c4_state
            if ratio == 4
            else self._cur_req_to_token_pool.req_to_token_c128_state
        )
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
            raise self._pool_exhausted(
                ratio, "state", state_extend_num_tokens, allocator.available_size()
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
        """Allocate compressed-KV slots for an extend at ``ratio``.

        Prefix/seq lens are translated to compressed units (``// ratio``); the
        c-pool last_loc comes from ``req_to_token_c{ratio}`` via
        :func:`get_last_loc` so the paged allocator continues in-page (or opens
        a fresh page at a ratio boundary), keeping the intra-page continuity the
        ``cmp_block_table`` reader relies on. Returns ``_empty_loc`` when this
        step closes no compressed token.
        """
        c_extend = self._compute_c_extend_counts(prefix_lens_cpu, seq_lens_cpu, ratio)
        if c_extend == 0:
            return self._empty_loc

        assert self._cur_req_to_token_pool is not None, (
            "alloc_extend/alloc_decode must be called with req_to_token_pool= "
            "for the c-pool last_loc lookup."
        )
        c_table = (
            self._cur_req_to_token_pool.req_to_token_c4
            if ratio == 4
            else self._cur_req_to_token_pool.req_to_token_c128
        )
        c_prefix = (prefix_lens // ratio).to(prefix_lens.dtype)
        c_seq = (seq_lens // ratio).to(seq_lens.dtype)
        c_last_loc = get_last_loc(c_table, req_pool_indices, c_prefix).to(
            last_loc_dtype
        )

        result = allocator.alloc_extend(
            c_prefix,
            prefix_lens_cpu // ratio,
            c_seq,
            seq_lens_cpu // ratio,
            c_last_loc,
            c_extend,
        )
        if result is None:
            raise self._pool_exhausted(
                ratio, "KV", c_extend, allocator.available_size()
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

        Shared by alloc_extend / alloc_decode (which differ only in how
        prefix_lens is derived). State lens are tail-only, precomputed by
        ScheduleBatch._compute_dsv4_state_lens_*; raw prefix_lens drives the
        state last_loc lookup.
        """
        assert req_pool_indices is not None, (
            "DSV4NPUTokenToKVPoolAllocator requires req_pool_indices "
            "(forwarded from batch.req_pool_indices)."
        )
        if dsv4_state_lens is not None:
            out_c4_state_loc = self._alloc_state_extend(
                self.c4_state_attn_allocator,
                prefix_lens,
                dsv4_state_lens.c4_prefix_lens,
                dsv4_state_lens.c4_prefix_lens_cpu,
                dsv4_state_lens.c4_seq_lens,
                dsv4_state_lens.c4_seq_lens_cpu,
                req_pool_indices,
                last_loc_dtype,
                dsv4_state_lens.c4_extend_num_tokens,
                ratio=4,
            )
            out_c128_state_loc = self._alloc_state_extend(
                self.c128_state_attn_allocator,
                prefix_lens,
                dsv4_state_lens.c128_prefix_lens,
                dsv4_state_lens.c128_prefix_lens_cpu,
                dsv4_state_lens.c128_seq_lens,
                dsv4_state_lens.c128_seq_lens_cpu,
                req_pool_indices,
                last_loc_dtype,
                dsv4_state_lens.c128_extend_num_tokens,
                ratio=128,
            )
        else:
            out_c4_state_loc = self._empty_loc
            out_c128_state_loc = self._empty_loc
        out_c4_loc = self._alloc_c_extend(
            self.c4_attn_allocator,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            req_pool_indices,
            last_loc_dtype,
            ratio=4,
        )
        out_c128_loc = self._alloc_c_extend(
            self.c128_attn_allocator,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            req_pool_indices,
            last_loc_dtype,
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

    def compute_dsv4_state_lens_extend(
        self, reqs: List[Req], seq_lens: List[int]
    ) -> Optional[DSV4StateLens]:
        """Per-req c{4,128}_state pool alloc lens for extend (tail-only).

        State pool stores only the trailing portion of each sequence (the c{N}
        compressor's read/write window); the tail length depends on raw
        seq_len's alignment to the SWA page boundary (128)::

            c4_alloc_len  = tail + 128 if (tail <= 3 and seq_len >= 128) else tail
            c128_alloc_len = tail                  where tail = seq_len % 128

        Long prefills allocate only the trailing partial window, not slots for
        already-compressed positions, so the small paged state pool (~256
        slots/req) stays sufficient even for 28k-token prompts.

        Mutates per-req cumulative state via getattr/setattr so the community
        ``Req`` needs no DSV4 field declarations:
          * ``req.c{4,128}_state_kv_len`` — cumulative slot count (prefix for
            the paged allocator; never decreases on eviction).
          * ``req.c{4,128}_state_alloc_offset`` — low-water raw-position mark
            for eviction (see ``dsv4_common_hooks.maybe_evict_dsv4_state``).

        Returns None when this model has no paged state pools (CUDA / non-V4 /
        zero budget) — callers pass that straight through as ``dsv4_state_lens``.
        """
        if self.c4_state_attn_allocator is None:
            return None
        c4_prefix: List[int] = []
        c4_seq: List[int] = []
        c128_prefix: List[int] = []
        c128_seq: List[int] = []
        for req, seq_len in zip(reqs, seq_lens):
            tail = seq_len % 128
            c4_alloc_len = tail + 128 if (tail <= 3 and seq_len >= 128) else tail
            c128_alloc_len = tail

            prev_c4 = getattr(req, "c4_state_kv_len", 0)
            prev_c128 = getattr(req, "c128_state_kv_len", 0)
            new_c4 = prev_c4 + c4_alloc_len
            new_c128 = prev_c128 + c128_alloc_len

            c4_prefix.append(prev_c4)
            c4_seq.append(new_c4)
            c128_prefix.append(prev_c128)
            c128_seq.append(new_c128)

            req.c4_state_kv_len = new_c4
            req.c128_state_kv_len = new_c128
            req.c4_state_alloc_offset = seq_len - c4_alloc_len
            req.c128_state_alloc_offset = seq_len - c128_alloc_len

        return self._pack_state_lens(
            c4_prefix,
            c4_seq,
            c128_prefix,
            c128_seq,
            c4_extend_num_tokens=int(sum(s - p for s, p in zip(c4_seq, c4_prefix))),
            c128_extend_num_tokens=int(
                sum(s - p for s, p in zip(c128_seq, c128_prefix))
            ),
        )

    def compute_dsv4_state_lens_decode(
        self, reqs: List[Req]
    ) -> Optional[DSV4StateLens]:
        """Per-req c{4,128}_state pool alloc lens for decode: exactly 1 new
        state slot per req per pool. ``c{N}_state_alloc_offset`` does NOT
        advance here (only eviction advances it). Returns None when there are
        no paged state pools."""
        if self.c4_state_attn_allocator is None:
            return None
        c4_prefix: List[int] = []
        c4_seq: List[int] = []
        c128_prefix: List[int] = []
        c128_seq: List[int] = []
        for req in reqs:
            prev_c4 = getattr(req, "c4_state_kv_len", 0)
            prev_c128 = getattr(req, "c128_state_kv_len", 0)
            c4_prefix.append(prev_c4)
            c4_seq.append(prev_c4 + 1)
            c128_prefix.append(prev_c128)
            c128_seq.append(prev_c128 + 1)
            req.c4_state_kv_len = prev_c4 + 1
            req.c128_state_kv_len = prev_c128 + 1

        bs = len(reqs)
        return self._pack_state_lens(
            c4_prefix,
            c4_seq,
            c128_prefix,
            c128_seq,
            c4_extend_num_tokens=bs,
            c128_extend_num_tokens=bs,
        )

    def compute_dsv4_state_lens_reserve(
        self, reqs: List[Req], prefix_lens: List[int], seq_lens: List[int]
    ) -> Optional[DSV4StateLens]:
        """Allocate state slots for a speculative pre-reserved raw interval."""
        if self.c4_state_attn_allocator is None:
            return None

        c4_prefix: List[int] = []
        c4_seq: List[int] = []
        c128_prefix: List[int] = []
        c128_seq: List[int] = []
        for req, prefix_len, seq_len in zip(reqs, prefix_lens, seq_lens):
            reserve = max(0, int(seq_len) - int(prefix_len))
            prev_c4 = getattr(req, "c4_state_kv_len", 0)
            prev_c128 = getattr(req, "c128_state_kv_len", 0)
            c4_prefix.append(prev_c4)
            c4_seq.append(prev_c4 + reserve)
            c128_prefix.append(prev_c128)
            c128_seq.append(prev_c128 + reserve)
            req.c4_state_kv_len = prev_c4 + reserve
            req.c128_state_kv_len = prev_c128 + reserve

        total = sum(max(0, int(s) - int(p)) for p, s in zip(prefix_lens, seq_lens))
        return self._pack_state_lens(
            c4_prefix,
            c4_seq,
            c128_prefix,
            c128_seq,
            c4_extend_num_tokens=total,
            c128_extend_num_tokens=total,
        )

    def _pack_state_lens(
        self,
        c4_prefix: List[int],
        c4_seq: List[int],
        c128_prefix: List[int],
        c128_seq: List[int],
        *,
        c4_extend_num_tokens: int,
        c128_extend_num_tokens: int,
    ) -> DSV4StateLens:
        c4_prefix_cpu = torch.tensor(c4_prefix, dtype=torch.int64)
        c4_seq_cpu = torch.tensor(c4_seq, dtype=torch.int64)
        c128_prefix_cpu = torch.tensor(c128_prefix, dtype=torch.int64)
        c128_seq_cpu = torch.tensor(c128_seq, dtype=torch.int64)
        return DSV4StateLens(
            c4_prefix_lens=c4_prefix_cpu.to(self.device, non_blocking=True),
            c4_prefix_lens_cpu=c4_prefix_cpu,
            c4_seq_lens=c4_seq_cpu.to(self.device, non_blocking=True),
            c4_seq_lens_cpu=c4_seq_cpu,
            c4_extend_num_tokens=c4_extend_num_tokens,
            c128_prefix_lens=c128_prefix_cpu.to(self.device, non_blocking=True),
            c128_prefix_lens_cpu=c128_prefix_cpu,
            c128_seq_lens=c128_seq_cpu.to(self.device, non_blocking=True),
            c128_seq_lens_cpu=c128_seq_cpu,
            c128_extend_num_tokens=c128_extend_num_tokens,
        )

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
        # Stash per-req tables for this call's last_loc lookups (read by
        # _alloc_c_extend / _alloc_state_extend); no permanent allocator->pool ref.
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
        self._cur_req_to_token_pool = req_to_token_pool
        out_full_loc = super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        if out_full_loc is None:
            return None

        out_swa_loc = self.translate_loc_from_full_to_swa(out_full_loc)
        # One new token per req. Model as an extend from (seq_len-1)//ratio to
        # seq_len//ratio so _alloc_c_extend anchors on the real c-pool last_loc.
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

    def free(
        self,
        free_index: Optional[torch.Tensor] = None,
        *,
        req=None,
        req_to_token_pool=None,
    ):
        """Unified free for full/swa/c4/c128 pools. Two forms (may co-fire):

          * ``free(free_index)`` — full + SWA only (tail/radix eviction; no req
            identity, so c-pool free can't run).
          * ``free(req=, req_to_token_pool=)`` — from DSV4NPUReqToTokenPool.free
            on req finish: reads the per-req slot lists from
            ``req_to_token_c{4,128}[_state]`` and returns them to the c-pools
            (the paged allocator dedupes by page).

        KV pools free ``[0, kv_len // ratio)``. State pools are 1-per-raw-token
        and free only the tail ``[c{N}_state_alloc_offset, kv_len)`` — the prefix
        was already returned by ScheduleBatch._evict_swa (state rides SWA
        eviction); freeing it again would double-free (caught by the paged
        allocator's debug_mode assert, corrupts the free list otherwise).
        """
        if free_index is not None:
            super().free(free_index)

        if req is None or req_to_token_pool is None:
            return
        kv_len = max(req.kv_committed_len, req.kv.kv_allocated_len)
        req_pool_idx = req.req_pool_idx
        if kv_len <= 0 or req_pool_idx is None:
            return

        # KV pools: free the leading [0, kv_len // ratio) compressed slots.
        for ratio, allocator, table_attr in (
            (4, self.c4_attn_allocator, "req_to_token_c4"),
            (128, self.c128_attn_allocator, "req_to_token_c128"),
        ):
            n = kv_len // ratio
            if n > 0 and hasattr(req_to_token_pool, table_attr):
                slots = getattr(req_to_token_pool, table_attr)[req_pool_idx, :n]
                # to int64 — paged allocator's free does cpu()//page_size on it.
                allocator.free(slots.to(torch.int64))

        # State pools: free only the tail [c{N}_state_alloc_offset, kv_len).
        for ratio, allocator, table_attr, off_attr in (
            (
                4,
                self.c4_state_attn_allocator,
                "req_to_token_c4_state",
                "c4_state_alloc_offset",
            ),
            (
                128,
                self.c128_state_attn_allocator,
                "req_to_token_c128_state",
                "c128_state_alloc_offset",
            ),
        ):
            if allocator is None or not hasattr(req_to_token_pool, table_attr):
                continue
            off = getattr(req, off_attr, 0)
            if kv_len > off:
                slots = getattr(req_to_token_pool, table_attr)[req_pool_idx, off:kv_len]
                allocator.free(slots.to(torch.int64))

    def backup_state(self):
        # EAGLE/NEXTN draft preprocess allocates speculative c{4,128} KV via
        # alloc_extend(backup_state=True) and rolls it back with restore_state.
        # The base SWATokenToKVPoolAllocator only snapshots the full + SWA pools,
        # so without this override the draft's c{4,128} (+ state) slots are never
        # rolled back -> they leak every draft step until the c4 pool exhausts.
        # Snapshot the sub-allocators alongside the base pools.
        return (
            super().backup_state(),
            self.c4_attn_allocator.backup_state(),
            self.c128_attn_allocator.backup_state(),
            (
                self.c4_state_attn_allocator.backup_state()
                if self.c4_state_attn_allocator is not None
                else None
            ),
            (
                self.c128_state_attn_allocator.backup_state()
                if self.c128_state_attn_allocator is not None
                else None
            ),
        )

    def restore_state(self, state):
        base, c4, c128, c4_state, c128_state = state
        super().restore_state(base)
        self.c4_attn_allocator.restore_state(c4)
        self.c128_attn_allocator.restore_state(c128)
        if self.c4_state_attn_allocator is not None and c4_state is not None:
            self.c4_state_attn_allocator.restore_state(c4_state)
        if self.c128_state_attn_allocator is not None and c128_state is not None:
            self.c128_state_attn_allocator.restore_state(c128_state)

    def clear(self):
        super().clear()
        # super().__init__ calls clear() before our sub-allocators exist;
        # getattr(..., None) tolerates that and the always-None state allocators.
        for attr in (
            "c4_attn_allocator",
            "c128_attn_allocator",
            "c4_state_attn_allocator",
            "c128_state_attn_allocator",
        ):
            allocator = getattr(self, attr, None)
            if allocator is not None:
                allocator.clear()
