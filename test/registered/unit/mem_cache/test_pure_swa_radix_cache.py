"""Unit tests for all-SWA RadixCache release semantics."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token

    def write(self, indices, values):
        self.req_to_token[indices] = values


class _FakeAllocator:
    """Rows routed to the full side are skipped: all-SWA has no full pool."""

    page_size = 1
    device = "cpu"

    def __init__(self):
        self.freed = []
        self.skipped = []
        self.window_freed = []

    def free_segment(self, indices, *, start_pos):
        self.freed.extend(indices.tolist())

    def free_segments(self, segments):
        for indices, start_pos in segments:
            self.free_segment(indices, start_pos=start_pos)

    def free_full_segments(self, segments):
        for indices, _ in segments:
            self.skipped.extend(indices.tolist())

    def free_swa_segment(self, indices, *, start_pos):
        self.window_freed.extend(indices.tolist())


def _make_cache(*, disable, num_tokens, page_size=1):
    return PureSWARadixCache(
        CacheInitParams(
            disable=disable,
            req_to_token_pool=_FakeReqToTokenPool(
                torch.arange(num_tokens, dtype=torch.int64).unsqueeze(0)
            ),
            token_to_kv_pool_allocator=_FakeAllocator(),
            page_size=page_size,
            sliding_window_size=4,
        )
    )


class TestPureSWARadixCache(CustomTestCase):
    def test_finish_inserts_up_to_the_evict_floor_and_frees_the_rest(self):
        cache = _make_cache(disable=False, num_tokens=8)
        allocator = cache.token_to_kv_pool_allocator
        token_ids = array("q", range(8))
        req = SimpleNamespace(
            origin_input_ids=token_ids,
            output_ids=array("q"),
            full_untruncated_fill_ids=token_ids,
            extra_key=None,
            cache_salt=None,
            last_node=cache.root_node,
            lock=None,
            priority=0,
            kv=ReqKvInfo(
                req_pool_idx=0,
                swa_evict_floor=4,
                component_evicted_seqlens={ComponentType.SWA: 6},
            ),
        )

        cache.checkpoint(req, up_to=8)
        cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, 8)])
        cache.unlock(req.lock)

        # [0, 4) went into the tree; [4, 6) was window-evicted; [6, 8) is freed.
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(match.device_prefix_len, 4)
        self.assertEqual(allocator.freed, [6, 7])
        self.assertEqual(allocator.skipped, [4, 5])


class TestDisabledPureSWARadixCache(CustomTestCase):
    def test_finished_req_skips_protected_prefix_and_evicted_range(self):
        cache = _make_cache(disable=True, num_tokens=10)
        allocator = cache.token_to_kv_pool_allocator
        token_ids = array("q", range(8))
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        req = SimpleNamespace(
            origin_input_ids=token_ids,
            output_ids=array("q"),
            full_untruncated_fill_ids=token_ids,
            extra_key=None,
            cache_salt=None,
            last_node=match.last_device_node,
            lock=cache.lock(match.last_device_node),
            priority=0,
            kv=ReqKvInfo(
                req_pool_idx=0,
                cache_protected_len=2,
                swa_evict_floor=3,
                component_evicted_seqlens={ComponentType.SWA: 6},
            ),
        )

        # protected 2, floor 3, cursor 6: [2, 3) and [6, 8) go back, [3, 6) is dead
        cache.checkpoint(req, up_to=8)
        cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, 8)])
        cache.unlock(req.lock)

        self.assertEqual(req.kv.cache_protected_len, 2)
        self.assertEqual(allocator.freed, [2, 6, 7])
        self.assertEqual(allocator.skipped, [3, 4, 5])
        self.assertEqual(cache.total_size(), 0)

    def test_window_eviction_frees_up_to_the_window(self):
        cache = _make_cache(disable=True, num_tokens=20, page_size=8)
        req = SimpleNamespace(kv=ReqKvInfo(req_pool_idx=0))

        cache.evict_sliding_windows(req, 20)

        # pre_len 20 - window 4 = 16, page-aligned: [0, 16) slides out. A
        # prefix-sharing cache would keep max(window, page) and stop at 8.
        self.assertEqual(cache.token_to_kv_pool_allocator.window_freed, list(range(16)))
        self.assertEqual(req.kv.get_evicted_seqlen(ComponentType.SWA), 16)


if __name__ == "__main__":
    unittest.main()
