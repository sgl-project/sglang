"""A dLLM request's in-flight denoise block must not be cached.

``dllm_incomplete_ids`` is reassigned in full on every denoise step, and the
dLLM page size is pinned to the block size, so that block is a whole trailing
page of the radix key. Caching it makes the next step's key diverge there: the
match drops below ``cache_protected_len``, ``cache_unfinished_req``'s
``free_segment`` slice inverts to empty (freeing nothing), and the new node
claims pages the previous node still owns -- inflating ``evictable_size()``
past what the pool physically holds and tripping the idle pool invariant.

Regression test for the over-count reported in #35270 (distinct from the
deficit-shaped leak in #32992).
"""

import unittest
from array import array

import torch

from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

PAGE_SIZE = 32


class _MockDllmReq(ReqDllmMixin):
    """Minimal Req carrying only what cache_unfinished_req touches."""

    def __init__(self, fill_ids, incomplete_ids=(), req_pool_idx=0):
        self.full_untruncated_fill_ids = array("q", fill_ids)
        self.extend_range = Range(0, len(self.full_untruncated_fill_ids))
        self.dllm_incomplete_ids = array("q", incomplete_ids)
        self.req_pool_idx = req_pool_idx
        self.cache_protected_len = 0
        self.last_node = None
        self.extra_key = None
        self.cache_salt = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.priority = 0

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_range.end]


def _make_cache(page_size=PAGE_SIZE, pool_tokens=1024):
    """Real paged pools: page-aligned KV indices are essential to this bug."""
    kv = MHATokenToKVPool(
        size=pool_tokens,
        page_size=page_size,
        dtype=torch.float16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=pool_tokens,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=kv,
        need_sort=False,
    )
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=512, device="cpu", enable_memory_saver=False
    )
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            eviction_policy="lru",
            enable_kv_cache_events=False,
        )
    )
    return cache, allocator, req_to_token_pool.req_to_token


def _tree_pages(cache, page_size=PAGE_SIZE):
    """Distinct physical pages referenced anywhere in the tree."""
    pages, stack = set(), [cache.root_node]
    while stack:
        node = stack.pop()
        if node is not cache.root_node:
            pages.update(v // page_size for v in node.value.tolist())
        stack.extend(node.children.values())
    return pages


class TestDllmVolatileBlockNotCached(unittest.TestCase):
    def test_in_flight_block_is_excluded_from_cacheable_ids(self):
        stable, block = list(range(64)), list(range(900, 932))
        req = _MockDllmReq(stable + block, incomplete_ids=block)
        self.assertEqual(list(req.get_cacheable_fill_ids()), stable)

    def test_resolved_block_is_cacheable(self):
        """With no block in flight the cacheable ids are the full fill ids."""
        fill = list(range(96))
        req = _MockDllmReq(fill, incomplete_ids=())
        self.assertEqual(list(req.get_cacheable_fill_ids()), fill)

    def test_accounting_does_not_exceed_owned_pages(self):
        cache, allocator, req_to_token = _make_cache()
        stable = list(range(64))

        req = _MockDllmReq(stable, incomplete_ids=[])
        req.last_node = cache.root_node
        # Settled prefix + the block's slots, allocated once (denoised in place).
        req_to_token[0, :96] = allocator.alloc(96).to(torch.int64)

        for block in (list(range(900, 932)), list(range(800, 832))):
            req.full_untruncated_fill_ids = array("q", stable + block)
            req.extend_range = Range(0, len(req.full_untruncated_fill_ids))
            req.dllm_incomplete_ids = array("q", block)
            cache.cache_unfinished_req(req)

        owned = len(_tree_pages(cache)) * PAGE_SIZE
        accounted = cache.evictable_size() + cache.protected_size()
        self.assertLessEqual(
            accounted,
            owned,
            f"evictable+protected={accounted} exceeds the {owned} tokens the "
            f"tree actually owns -- the same pages are counted under two nodes",
        )


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

if __name__ == "__main__":
    unittest.main()
