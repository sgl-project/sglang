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
    """Rows routed to the full side are skipped: all-SWA has no full pool."""

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


def _make_req():
    return SimpleNamespace(
        last_node=None,
        kv=ReqKvInfo(
            req_pool_idx=0,
            cache_protected_len=2,
            swa_evict_floor=3,
            swa_evicted_seqlen=6,
        ),
    )


class TestPureSWAChunkCache(CustomTestCase):
    def _make_cache(self):
        cache = PureSWAChunkCache.__new__(PureSWAChunkCache)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(10, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = _FakeAllocator()
        return cache

    def test_finished_req_skips_protected_prefix_and_evicted_range(self):
        cache = self._make_cache()

        # protected 2, floor 3, cursor 6: [2, 3) and [6, 8) go back, [3, 6) is dead
        cache.cache_finished_req(_make_req(), owned_kv_len=8)

        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [2, 6, 7])
        self.assertEqual(cache.token_to_kv_pool_allocator.skipped, [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
