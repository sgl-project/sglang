"""DSV4-NPU SWA + c128 paged KV allocator.

Subclasses :class:`SWATokenToKVPoolAllocator` and adds paged allocation for the
C128 compressed-KV pool alongside the parent's full + SWA pools. C4 KV slots
are derived from full slots. Compressor state is fixed ring storage owned by
the KV pool and never enters this token allocator.

Per ``alloc_extend`` / ``alloc_decode``:
  1. super() allocates the full + SWA slots (``out_full_loc``).
  2. Derive c4 KV slots from full and allocate c128 KV slots — one compressed
     token per ``ratio`` raw tokens
     (``seq_len // ratio - prefix_len // ratio``).
  3. Return a :class:`DSV4OutCacheLoc` containing only KV slot families.

The bundle is the explicit return value:
mem_cache/common.py unpacks ``out_full_loc`` and stashes the bundle on
``batch.out_cache_loc_dsv4``; ``DSV4NPUReqToTokenPool`` writes the per-req
``req_to_c128_sidecar`` table that :meth:`free` and the last_loc lookup read back.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_write_dsv4_extend,
)
from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc


def get_last_loc(
    req_to_c128_sidecar: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Slot id of each req's last already-allocated token, or -1 when
    ``prefix_lens[i] == 0`` (fresh req).

    Looks up the C128 sidecar page to anchor the paged allocator's
    ``alloc_extend`` on the real previous tail slot, preserving intra-page slot
    continuity. Result dtype matches ``prefix_lens``.
    """
    req_pool_indices = req_pool_indices.to(torch.int64)
    last_pos = (prefix_lens.to(torch.int64) - 1).clamp(min=0)
    page_ids = req_to_c128_sidecar[req_pool_indices, last_pos // page_size].to(
        prefix_lens.dtype
    )
    last_loc = page_ids * page_size + last_pos.to(prefix_lens.dtype) % page_size
    return torch.where(prefix_lens > 0, last_loc, torch.full_like(prefix_lens, -1))


def alloc_paged_token_slots_extend_npu(*args, batch=None, **kwargs):
    if batch is not None and is_deepseek_v4(batch.model_config.hf_config):
        return alloc_paged_token_slots_reserve_extend(*args, batch=batch, **kwargs)
    return alloc_paged_token_slots_extend(*args, batch=batch, **kwargs)


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
    batch=None,
):
    """Allocate reserved draft KV slots and update DSV4 KV tables."""
    out_cache_loc = alloc_paged_token_slots_extend(
        tree_cache,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        req_pool_indices=req_pool_indices,
        batch=batch,
    )
    if batch is not None:
        maybe_write_dsv4_extend(
            batch,
            batch.req_pool_indices_cpu,
            prefix_lens_cpu,
            seq_lens_cpu,
        )
    return out_cache_loc


class DSV4NPUTokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """SWA allocator + C128 KV allocator and full-derived C4 locations."""

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
            # C128 KV sub-pool implements KVCache, so it drops into the standard
            # paged allocator. pool_size is in compressed-token units.
            return NPUPagedTokenToKVPoolAllocator(
                pool_size,
                page_size=pool.kernel_page_size,
                dtype=dtype,
                device=device,
                kvcache=pool,
                need_sort=need_sort,
            )

        self.c128_attn_allocator = mk(kvcache.c128_size, kvcache.c128_kv_pool)
        self.c128_page_refcount = torch.zeros(
            self.c128_attn_allocator.num_pages + 1,
            dtype=torch.int32,
            device=device,
        )

        # Returned by the c-pool helpers when a step adds no compressed tokens.
        self._empty_loc = torch.empty((0,), dtype=torch.int64, device=device)

        # Per-call handle to the DSV4NPUReqToTokenPool, stashed by alloc_extend/
        # alloc_decode for the C128 KV last_loc lookup.
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
    def _derive_c4_loc_from_full(out_full_loc: torch.Tensor) -> torch.Tensor:
        """Map full slots closing a 4-token group to their C4 slots."""
        completed_group = (out_full_loc >= 0) & ((out_full_loc % 4) == 3)
        return out_full_loc[completed_group] // 4

    def retain_c128_pages(self, page_ids: torch.Tensor) -> None:
        page_ids = page_ids.to(torch.int64).view(-1)
        if page_ids.numel() == 0:
            return
        self.c128_page_refcount.index_add_(
            0,
            page_ids,
            torch.ones_like(page_ids, dtype=self.c128_page_refcount.dtype),
        )

    def release_c128_pages(self, page_ids: torch.Tensor) -> None:
        page_ids = torch.unique(page_ids.to(torch.int64).view(-1))
        page_ids = page_ids[page_ids > 0]
        if page_ids.numel() == 0:
            return
        self.c128_page_refcount.index_add_(
            0,
            page_ids,
            -torch.ones_like(page_ids, dtype=self.c128_page_refcount.dtype),
        )
        free_pages = page_ids[self.c128_page_refcount[page_ids] == 0]
        if free_pages.numel() > 0:
            self.c128_attn_allocator.free(
                free_pages * self.c128_attn_allocator.page_size
            )

    def replace_req_c128_prefix(
        self, req_pool_idx: int, page_ids: torch.Tensor, req_to_token_pool
    ) -> None:
        table = req_to_token_pool.req_to_c128_sidecar
        page_ids = page_ids.to(device=table.device, dtype=table.dtype).view(-1)
        old = table[req_pool_idx, : page_ids.numel()].clone()
        changed = old != page_ids
        self.release_c128_pages(old[changed])
        self.retain_c128_pages(page_ids[changed])
        table[req_pool_idx, : page_ids.numel()] = page_ids

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
        c-pool last_loc comes from ``req_to_c128_sidecar`` via
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
        c_table = self._cur_req_to_token_pool.req_to_c128_sidecar
        c_prefix = (prefix_lens // ratio).to(prefix_lens.dtype)
        c_seq = (seq_lens // ratio).to(seq_lens.dtype)
        c_last_loc = get_last_loc(
            c_table, req_pool_indices, c_prefix, allocator.page_size
        ).to(last_loc_dtype)

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

    def _has_c128_sidecar_capacity(
        self, prefix_lens_cpu: torch.Tensor, seq_lens_cpu: torch.Tensor
    ) -> bool:
        ratio = 128
        page_size = self.c128_attn_allocator.page_size
        prefix_groups = (prefix_lens_cpu // ratio + page_size - 1) // page_size
        seq_groups = (seq_lens_cpu // ratio + page_size - 1) // page_size
        need = int((seq_groups - prefix_groups).clamp(min=0).sum().item())
        return need <= self.c128_attn_allocator.available_size() // page_size

    def _alloc_compressed_kv(
        self,
        out_full_loc: torch.Tensor,
        out_swa_loc: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc_dtype: torch.dtype,
        req_pool_indices: Optional[torch.Tensor],
    ) -> DSV4OutCacheLoc:
        """Allocate C128 KV and derive C4 KV, then bundle all KV locations."""
        assert req_pool_indices is not None, (
            "DSV4NPUTokenToKVPoolAllocator requires req_pool_indices "
            "(forwarded from batch.req_pool_indices)."
        )
        out_c4_loc = self._derive_c4_loc_from_full(out_full_loc)
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
        req_to_token_pool=None,
    ) -> Optional[DSV4OutCacheLoc]:
        # Stash per-req tables for this call's last_loc lookups (read by
        # _alloc_c_extend / _alloc_state_extend); no permanent allocator->pool ref.
        self._cur_req_to_token_pool = req_to_token_pool
        if not self._has_c128_sidecar_capacity(prefix_lens_cpu, seq_lens_cpu):
            return None
        out_full_loc = super().alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        return self._wrap_full_alloc(
            out_full_loc,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc.dtype,
            req_pool_indices,
        )

    def _wrap_full_alloc(
        self,
        out_full_loc,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        loc_dtype,
        req_pool_indices,
    ) -> Optional[DSV4OutCacheLoc]:
        # Shared tail of alloc_extend / alloc_extend_swa_tail: translate the full
        # loc to swa, then add the c4/c128 KV pools into a DSV4OutCacheLoc.
        if out_full_loc is None:
            return None
        out_swa_loc = self.translate_loc_from_full_to_swa(out_full_loc)
        assert out_swa_loc is not None, (
            "translate_loc_from_full_to_swa returned None — "
            "full_to_swa_index_mapping not initialized?"
        )
        return self._alloc_compressed_kv(
            out_full_loc,
            out_swa_loc,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            loc_dtype,
            req_pool_indices,
        )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        *,
        req_pool_indices: Optional[torch.Tensor] = None,
        req_to_token_pool=None,
    ) -> Optional[DSV4OutCacheLoc]:
        self._cur_req_to_token_pool = req_to_token_pool
        if not self._has_c128_sidecar_capacity(
            (seq_lens_cpu - 1).clamp(min=0), seq_lens_cpu
        ):
            return None
        out_full_loc = super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        if out_full_loc is None:
            return None

        out_swa_loc = self.translate_loc_from_full_to_swa(out_full_loc)
        # One new token per req. Model as an extend from (seq_len-1)//ratio to
        # seq_len//ratio so _alloc_c_extend anchors on the real c-pool last_loc.
        prefix_lens = (seq_lens - 1).clamp(min=0)
        prefix_lens_cpu = (seq_lens_cpu - 1).clamp(min=0)
        return self._alloc_compressed_kv(
            out_full_loc,
            out_swa_loc,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc.dtype,
            req_pool_indices,
        )

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        swa_tail_len: int,
        *,
        req_pool_indices: Optional[torch.Tensor] = None,
        req_to_token_pool=None,
    ) -> Optional[DSV4OutCacheLoc]:
        """Disagg-decode prealloc variant of :meth:`alloc_extend`: super() does
        full+swa-tail, then _alloc_compressed_kv adds c4/c128 KV → DSV4OutCacheLoc.
        """
        self._cur_req_to_token_pool = req_to_token_pool
        if not self._has_c128_sidecar_capacity(prefix_lens_cpu, seq_lens_cpu):
            return None
        out_full_loc = super().alloc_extend_swa_tail(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            swa_tail_len,
        )
        return self._wrap_full_alloc(
            out_full_loc,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc.dtype,
            req_pool_indices,
        )

    def free(
        self,
        free_index: Optional[torch.Tensor] = None,
        *,
        req=None,
        req_to_token_pool=None,
    ):
        """Unified free for full/SWA/C4/C128 KV and C128 request state.

        Two forms may co-fire:

          * ``free(free_index)`` — full + SWA only (tail/radix eviction; no req
            identity, so c-pool free can't run).
          * ``free(req=, req_to_token_pool=)`` — from DSV4NPUReqToTokenPool.free
            on request finish: returns C128 KV pages and clears that request's
            fixed C128 state bank before the req_pool_idx can be reused.
        """
        if free_index is not None:
            super().free(free_index)

        if req is None or req_to_token_pool is None:
            return
        kv_len = max(req.kv_committed_len, req.kv.kv_allocated_len)
        req_pool_idx = req.req_pool_idx
        if kv_len <= 0 or req_pool_idx is None:
            return

        row = req_to_token_pool.req_to_c128_sidecar[int(req_pool_idx)]
        self.release_c128_pages(row[row > 0])
        row.zero_()
        self.get_kvcache().clear_c128_req_state(int(req_pool_idx))

    def available_size(self):
        return min(
            super().available_size(),
            self.c128_attn_allocator.available_size() * 128,
        )

    def resize(self, config) -> None:
        self.c128_attn_allocator.size = int(config.c128_max_total_num_tokens)
        self.c128_attn_allocator.num_pages = (
            self.c128_attn_allocator.size // self.c128_attn_allocator.page_size
        )
        super().resize(config)

    def clear(self):
        super().clear()
        # super().__init__ calls clear() before our C128 allocator exists.
        for attr in ("c128_attn_allocator",):
            allocator = getattr(self, attr, None)
            if allocator is not None:
                allocator.clear()
        refcount = getattr(self, "c128_page_refcount", None)
        if refcount is not None:
            refcount.zero_()
