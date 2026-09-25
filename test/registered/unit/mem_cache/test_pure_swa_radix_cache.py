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
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeAllocator:
    """Rows routed to the full side are skipped: all-SWA has no full pool."""

    page_size = 1
    device = "cpu"

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


class TestPureSWARadixCache(CustomTestCase):
    def test_finish_inserts_up_to_the_evict_floor_and_frees_the_rest(self):
        allocator = _FakeAllocator()
        cache = PureSWARadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.arange(8, dtype=torch.int64).unsqueeze(0)
                ),
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                sliding_window_size=4,
            )
        )
        token_ids = array("q", range(8))
        req = SimpleNamespace(
            origin_input_ids=token_ids,
            output_ids=array("q"),
            extra_key=None,
            cache_salt=None,
            last_node=None,
            priority=0,
            kv=ReqKvInfo(req_pool_idx=0, swa_evict_floor=4, swa_evicted_seqlen=6),
        )

        cache.cache_finished_req(req, owned_kv_len=8)

        # [0, 4) went into the tree; [4, 6) was window-evicted; [6, 8) is freed.
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(match.device_indices), 4)
        self.assertEqual(allocator.freed, [6, 7])
        self.assertEqual(allocator.skipped, [4, 5])


if __name__ == "__main__":
    unittest.main()
