"""Unit tests for all-SWA ChunkCache release semantics."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.chunk_cache import PureSWAChunkCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeAllocator:
    """Records what free_kv_row hands back. On an all-SWA allocator the
    full-side call is a no-op, so rows routed there are the ones skipped."""

    page_size = 1

    def __init__(self):
        self.freed = []
        self.skipped = []

    def free_segment(self, indices, *, start_pos):
        self.freed.extend(indices.tolist())

    def free_segments(self, segments):
        for indices, start_pos in segments:
            self.free_segment(indices, start_pos=start_pos)

    def free_full_segments(self, segments):
        for indices, _ in segments:
            self.skipped.extend(indices.tolist())


def _make_req(*, cache_protected_len=0):
    return SimpleNamespace(
        kv=ReqKvInfo(
            req_pool_idx=0,
            cache_protected_len=cache_protected_len,
            swa_evict_floor=3,
            swa_evicted_seqlen=6,
        )
    )


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

        cache.cache_finished_req(_make_req(), owned_kv_len=8)

        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [0, 1, 2, 6, 7])
        self.assertEqual(cache.token_to_kv_pool_allocator.skipped, [3, 4, 5])

    def test_finished_req_skips_protected_prefix(self):
        cache = self._make_cache()

        cache.cache_finished_req(_make_req(cache_protected_len=2), owned_kv_len=8)

        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [2, 6, 7])
        self.assertEqual(cache.token_to_kv_pool_allocator.skipped, [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
