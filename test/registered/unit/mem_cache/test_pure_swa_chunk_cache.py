"""Unit tests for all-SWA ChunkCache release semantics."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.chunk_cache import PureSWAChunkCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.detach().cpu().clone())


class _FakeReq:
    req_pool_idx = 0
    cache_protected_len = 0
    swa_evict_floor = 3
    kv = SimpleNamespace(swa_evicted_seqlen=6)

    def pop_committed_kv_cache(self):
        return 8


def _make_cache(page_size: int) -> PureSWAChunkCache:
    cache = PureSWAChunkCache.__new__(PureSWAChunkCache)
    cache.page_size = page_size
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(16, dtype=torch.int64).unsqueeze(0)
    )
    cache.token_to_kv_pool_allocator = _FakeAllocator()
    return cache


class TestPureSWAChunkCache(CustomTestCase):
    def test_finished_req_skips_already_evicted_swa_range(self):
        """The range already freed by SWA window eviction is not freed a second time."""
        cache = _make_cache(page_size=1)

        result = cache.cache_finished_req(_FakeReq(), kv_len_to_handle=8)

        self.assertEqual(len(cache.token_to_kv_pool_allocator.freed), 1)
        freed = cache.token_to_kv_pool_allocator.freed[0]
        self.assertTrue(torch.equal(freed, torch.tensor([0, 1, 2, 6, 7])))
        self.assertEqual(result.unhandled_kv_start, 8)

    def test_finished_req_ceils_evict_floor_and_floors_the_committed_end(self):
        """free_swa_out_of_window_slots never writes its ceil'd evict_floor back to the req."""
        cache = _make_cache(page_size=4)
        req = _FakeReq()
        req.swa_evict_floor = 3
        req.kv = SimpleNamespace(swa_evicted_seqlen=8)

        result = cache.cache_finished_req(req, kv_len_to_handle=14)

        freed = cache.token_to_kv_pool_allocator.freed[0]
        self.assertTrue(torch.equal(freed, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])))
        self.assertEqual(result.unhandled_kv_start, 12)

    def test_finished_req_frees_everything_below_committed_when_there_is_no_hole(self):
        """With no SWA eviction the whole page-aligned prefix is freed in one call."""
        cache = _make_cache(page_size=4)
        req = _FakeReq()
        req.swa_evict_floor = 0
        req.kv = SimpleNamespace(swa_evicted_seqlen=0)

        result = cache.cache_finished_req(req, kv_len_to_handle=10)

        freed = cache.token_to_kv_pool_allocator.freed[0]
        self.assertTrue(torch.equal(freed, torch.arange(8)))
        self.assertEqual(result.unhandled_kv_start, 8)

    def test_finished_req_asserts_evicted_seqlen_stays_below_the_boundary(self):
        """A hole above the reported boundary cannot be described by one int."""
        cache = _make_cache(page_size=4)
        req = _FakeReq()
        req.swa_evict_floor = 0
        req.kv = SimpleNamespace(swa_evicted_seqlen=12)

        with self.assertRaises(AssertionError):
            cache.cache_finished_req(req, kv_len_to_handle=10)

    def test_finished_req_asserts_cache_protected_len_is_zero(self):
        """ChunkCache never protects a prefix, so a non-zero value means the caller is confused."""
        cache = _make_cache(page_size=4)
        req = _FakeReq()
        req.cache_protected_len = 4

        with self.assertRaises(AssertionError):
            cache.cache_finished_req(req, kv_len_to_handle=8)


if __name__ == "__main__":
    unittest.main()
