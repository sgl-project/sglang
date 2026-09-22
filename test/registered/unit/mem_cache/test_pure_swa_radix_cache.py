"""Unit tests for all-SWA RadixCache release semantics."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
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


def _make_cache():
    allocator = _FakeAllocator()
    cache = PureSWARadixCache.__new__(PureSWARadixCache)
    cache.disable = False
    cache.is_eagle = False
    cache.page_size = 1
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(8, dtype=torch.int64).unsqueeze(0)
    )
    cache.token_to_kv_pool_allocator = allocator
    return cache, allocator


def _make_req(*, swa_evicted_seqlen):
    return SimpleNamespace(
        origin_input_ids=list(range(8)),
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        last_node=None,
        kv=ReqKvInfo(
            req_pool_idx=0,
            swa_evict_floor=4,
            swa_evicted_seqlen=swa_evicted_seqlen,
        ),
    )


class TestPureSWARadixCache(CustomTestCase):
    def test_no_insert_frees_window_after_evict_floor_before_swa_eviction(self):
        cache, allocator = _make_cache()

        cache.cache_finished_req(
            _make_req(swa_evicted_seqlen=0), is_insert=False, owned_kv_len=8
        )

        self.assertEqual(allocator.freed, list(range(8)))
        self.assertEqual(allocator.skipped, [])

    def test_no_insert_keeps_rows_below_the_floor_after_swa_eviction(self):
        cache, allocator = _make_cache()

        # Decode evicted [4, 6): the floor is 4, the cursor is 6. Rows below
        # the floor were shielded from eviction and are still this req's.
        cache.cache_finished_req(
            _make_req(swa_evicted_seqlen=6), is_insert=False, owned_kv_len=8
        )

        self.assertEqual(allocator.freed, [0, 1, 2, 3, 6, 7])
        self.assertEqual(allocator.skipped, [4, 5])


if __name__ == "__main__":
    unittest.main()
