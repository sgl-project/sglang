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
  4. Bundle full / swa / c4 / c128 slot tensors into a
     :class:`DSV4OutCacheLoc` and stash on ``self._last_dsv4_alloc``.

State-pool slots (``out_c4_state_loc`` / ``out_c128_state_loc``) are not
allocated here — they are derived on the fly by the attention backend
from raw KV slots via ``translate_kv_loc_to_compress_state_loc``. We
emit empty placeholder tensors for those two fields.

mem_cache/common.py fetches the bundle via ``get_last_dsv4_alloc()``
after each alloc, stashes it on ``batch.out_cache_loc_dsv4``, and writes
the per-req tables on the DSV4NPUReqToTokenPool.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc


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

        # Empty placeholder used for the unused state fields of DSV4OutCacheLoc.
        # Cached as a single zero-length int64 tensor on the right device so we
        # don't churn allocations on every alloc_extend / alloc_decode.
        # TODO: allocate state pool slots here too. State pool is currently
        # derived on-the-fly in the attention backend via
        # translate_kv_loc_to_compress_state_loc; req_to_token_c{4,128}_state
        # tables exist on DSV4NPUReqToTokenPool but stay zero-filled until
        # this allocator emits real out_c{4,128}_state_loc values.
        self._empty_loc = torch.empty((0,), dtype=torch.int64, device=device)
        self._last_dsv4_alloc: Optional[DSV4OutCacheLoc] = None

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

    def _alloc_c_extend(
        self,
        allocator: NPUPagedTokenToKVPoolAllocator,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        ratio: int,
    ) -> torch.Tensor:
        """Allocate compressed slots for an extend operation at ``ratio``.

        Per-request prefix / seq lengths are translated from raw token
        units to compressed units by integer-dividing by ``ratio``. last_loc
        is used as the prefix tail-page anchor for the paged allocator's
        ``alloc_extend``.
        """
        c_prefix = (prefix_lens // ratio).to(prefix_lens.dtype)
        c_seq = (seq_lens // ratio).to(seq_lens.dtype)
        c_prefix_cpu = prefix_lens_cpu // ratio
        c_seq_cpu = seq_lens_cpu // ratio
        c_extend = self._compute_c_extend_counts(prefix_lens_cpu, seq_lens_cpu, ratio)
        if c_extend == 0:
            return self._empty_loc
        result = allocator.alloc_extend(
            c_prefix,
            c_prefix_cpu,
            c_seq,
            c_seq_cpu,
            last_loc,
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

        # Pass a dummy last_loc tensor (one -1 per req) to the c-pool extend.
        # The compressed paged allocator uses last_loc to anchor the prefix
        # tail page, but since compressed sequences grow monotonically inside
        # their own pool and we don't track per-req c-pool tails, -1 makes
        # the allocator start a fresh tail page each extend. That is wasteful
        # if extend_num_tokens straddles a page boundary, but acceptable for
        # an initial cut. TODO: track c-pool last_loc per-req.
        c_last_loc = torch.full(
            (prefix_lens.shape[0],), -1, dtype=last_loc.dtype, device=last_loc.device,
        )

        out_c4_loc = self._alloc_c_extend(
            self.c4_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            c_last_loc,
            ratio=4,
        )
        out_c128_loc = self._alloc_c_extend(
            self.c128_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            c_last_loc,
            ratio=128,
        )

        self._last_dsv4_alloc = DSV4OutCacheLoc(
            out_full_loc=out_full_loc,
            out_swa_loc=out_swa_loc,
            out_c4_loc=out_c4_loc,
            out_c128_loc=out_c128_loc,
            out_c4_state_loc=self._empty_loc,
            out_c128_state_loc=self._empty_loc,
        )
        return out_full_loc

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        out_full_loc = super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        if out_full_loc is None:
            self._last_dsv4_alloc = None
            return None

        out_swa_loc = self.translate_loc_from_full_to_swa(out_full_loc)

        # For decode, we add one token per req. A new compressed-K token
        # is emitted when seq_lens[i] % ratio == 0 (post-decode). We model
        # this as an extend from (seq_lens - 1) // ratio to seq_lens // ratio.
        prefix_lens = (seq_lens - 1).clamp(min=0)
        prefix_lens_cpu = (seq_lens_cpu - 1).clamp(min=0)
        c_last_loc = torch.full(
            (seq_lens.shape[0],), -1, dtype=last_loc.dtype, device=last_loc.device,
        )

        out_c4_loc = self._alloc_c_extend(
            self.c4_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            c_last_loc,
            ratio=4,
        )
        out_c128_loc = self._alloc_c_extend(
            self.c128_attn_allocator,
            prefix_lens, prefix_lens_cpu,
            seq_lens, seq_lens_cpu,
            c_last_loc,
            ratio=128,
        )

        self._last_dsv4_alloc = DSV4OutCacheLoc(
            out_full_loc=out_full_loc,
            out_swa_loc=out_swa_loc,
            out_c4_loc=out_c4_loc,
            out_c128_loc=out_c128_loc,
            out_c4_state_loc=self._empty_loc,
            out_c128_state_loc=self._empty_loc,
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
        self._last_dsv4_alloc = None
