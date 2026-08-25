"""Unit tests for all-SWA ChunkCache release semantics."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.chunk_cache import ChunkCache, PureSWAChunkCache
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
    swa_evict_floor = 3
    cache_protected_len = 0
    kv = SimpleNamespace(swa_evicted_seqlen=6)

    def pop_committed_kv_cache(self):
        return 8


class TestPureSWAChunkCache(CustomTestCase):
    def _make_cache(self):
        cache = PureSWAChunkCache.__new__(PureSWAChunkCache)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(10, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = _FakeAllocator()
        return cache

    def test_finished_req_skips_already_evicted_swa_range(self):
        cache = self._make_cache()

        cache.cache_finished_req(_FakeReq(), kv_len_to_handle=8)

        self.assertEqual(len(cache.token_to_kv_pool_allocator.freed), 1)
        freed = cache.token_to_kv_pool_allocator.freed[0]
        self.assertTrue(torch.equal(freed, torch.tensor([0, 1, 2, 6, 7])))

    def test_finished_req_skips_protected_prefix(self):
        cache = self._make_cache()
        req = _FakeReq()
        req.cache_protected_len = 2

        cache.cache_finished_req(req, kv_len_to_handle=8)

        freed = cache.token_to_kv_pool_allocator.freed[0]
        self.assertTrue(torch.equal(freed, torch.tensor([2, 6, 7])))


class _RecordingAllocator:
    def __init__(self):
        self.calls = []

    def free_segments(self, segments, *, swa_evicted_seqlen=None):
        owned_segments = [
            (indices.detach().cpu().clone(), start_pos)
            for indices, start_pos in segments
        ]
        self.calls.append((owned_segments, swa_evicted_seqlen))


class TestChunkCache(CustomTestCase):
    def test_finished_req_passes_swa_eviction_frontier(self):
        cache = ChunkCache.__new__(ChunkCache)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(12, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = _RecordingAllocator()
        req = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=2,
            kv=SimpleNamespace(swa_evicted_seqlen=8),
        )

        cache.cache_finished_req(req, kv_len_to_handle=11)

        [(segments, swa_evicted_seqlen)] = cache.token_to_kv_pool_allocator.calls
        [(indices, start_pos)] = segments
        self.assertTrue(torch.equal(indices, torch.arange(2, 11)))
        self.assertEqual(start_pos, 2)
        self.assertEqual(swa_evicted_seqlen, 8)


if __name__ == "__main__":
    unittest.main()
