"""DSV4-NPU memory pool subclass.

Subclasses :class:`DeepSeekV4TokenToKVPool` to neutralize the legacy
self-written c-page free-list allocator (see commit ``8c1e87b``). c4/c128
page allocation is delegated to :class:`DSV4NPUTokenToKVPoolAllocator`,
which drives the standard ``NPUPagedTokenToKVPoolAllocator`` on the
c4_kv_pool / c128_kv_pool sub-buffers and writes per-req slot ids into
``DSV4NPUReqToTokenPool``'s tables.

The base class's ``req_to_token_c{4,128}_pages`` + free-list state still
gets allocated in ``__init__`` for now (small memory cost; not on the hot
path). D8 in the refactor plan will remove it from the base class.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class DSV4NPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    """NPU DSV4 KV pool that delegates c-page allocation to a standard
    paged allocator (see ``DSV4NPUTokenToKVPoolAllocator``).
    """

    # ------------------------------------------------------------------
    # Neutralize the legacy c-page allocator hooks. mem_cache/common.py
    # calls these on alloc_extend / alloc_decode / release_kv_cache, so
    # they must exist; but with the new design the actual c-page bookkeeping
    # lives in ``DSV4NPUTokenToKVPoolAllocator`` + DSV4NPUReqToTokenPool.
    # ------------------------------------------------------------------

    def alloc_c_pages_for_batch(
        self,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        # No-op: c-page allocation happens inside DSV4NPUTokenToKVPoolAllocator
        # at alloc_extend / alloc_decode time. The slot ids are stashed in
        # the allocator's _last_dsv4_alloc and written to per-req tables by
        # mem_cache/common.py.
        pass

    def free_c_pages(self, req_pool_idx: int) -> None:
        # No-op: c-page free is driven by the allocator's free() override,
        # which returns each sub-pool's pages via the underlying
        # NPUPagedTokenToKVPoolAllocator.free().
        pass

    def clear_c_pages(self) -> None:
        # No-op for the same reason: clearing the allocator clears all
        # sub-pool free lists.
        pass
