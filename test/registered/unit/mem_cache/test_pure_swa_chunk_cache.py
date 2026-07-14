"""Unit tests for all-SWA ChunkCache release semantics."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.chunk_cache import ChunkCache, PureSWAChunkCache
from sglang.srt.utils.common import ceil_align
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self, page_size: int = 1) -> None:
        self.page_size = page_size
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.detach().cpu().clone())


class _FakeReq:
    req_pool_idx = 0
    swa_evict_floor = 4
    kv = SimpleNamespace(swa_evicted_seqlen=6)

    def pop_committed_kv_cache(self):
        return 8


class TestPureSWAChunkCache(CustomTestCase):
    def test_chunk_cache_returns_zero_handoff_without_freeing(self) -> None:
        """Plain ChunkCache delegates the full allocation to its caller."""
        cache = ChunkCache.__new__(ChunkCache)

        result = cache.cache_finished_req(_FakeReq(), kv_len_to_handle=8)

        self.assertEqual(result.unhandled_kv_start, 0)

    def test_finished_req_matches_old_local_ceil_result(self):
        """PureSWA ChunkCache consumes the boundary formerly rounded locally."""
        cache = PureSWAChunkCache.__new__(PureSWAChunkCache)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(10, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = _FakeAllocator(page_size=4)
        cache.page_size = 4

        result = cache.cache_finished_req(_FakeReq(), kv_len_to_handle=8)

        self.assertEqual(_FakeReq.swa_evict_floor, ceil_align(3, cache.page_size))
        self.assertEqual(result.unhandled_kv_start, 8)
        self.assertEqual(len(cache.token_to_kv_pool_allocator.freed), 1)
        freed = cache.token_to_kv_pool_allocator.freed[0]
        self.assertTrue(torch.equal(freed, torch.tensor([0, 1, 2, 3, 6, 7])))

    def test_finished_req_rejects_misaligned_floor_before_free(self):
        """PureSWA ChunkCache rejects a malformed floor before allocator mutation."""
        cache = PureSWAChunkCache.__new__(PureSWAChunkCache)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(10, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = _FakeAllocator(page_size=4)
        cache.page_size = 4
        req = _FakeReq()
        req.swa_evict_floor = 3

        with self.assertRaisesRegex(AssertionError, "swa_evict_floor"):
            cache.cache_finished_req(req, kv_len_to_handle=8)

        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [])

    def test_finished_req_allows_evicted_boundary_past_cacheable_end(self):
        """An SWA hole extending past K leaves only its live prefix to free."""
        cache = PureSWAChunkCache.__new__(PureSWAChunkCache)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(10, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = _FakeAllocator()
        cache.page_size = 1
        req = SimpleNamespace(
            req_pool_idx=0,
            swa_evict_floor=3,
            kv=SimpleNamespace(swa_evicted_seqlen=10),
        )

        result = cache.cache_finished_req(req, kv_len_to_handle=8)

        self.assertEqual(result.unhandled_kv_start, 8)
        self.assertEqual(len(cache.token_to_kv_pool_allocator.freed), 1)
        freed = cache.token_to_kv_pool_allocator.freed[0]
        self.assertTrue(torch.equal(freed, torch.tensor([0, 1, 2])))


if __name__ == "__main__":
    unittest.main()
