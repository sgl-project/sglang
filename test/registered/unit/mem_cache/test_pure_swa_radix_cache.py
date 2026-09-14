"""Unit tests for all-SWA RadixCache release semantics."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        if indices.numel() == 0:
            return
        self.freed.append(indices.detach().cpu().tolist())


class TestPureSWARadixCache(CustomTestCase):
    def test_no_insert_frees_window_after_evict_floor_before_swa_eviction(self):
        allocator = _FakeAllocator()
        cache = PureSWARadixCache.__new__(PureSWARadixCache)
        cache.disable_finished_insert = False
        cache.disable = False
        cache.is_eagle = False
        cache.page_size = 1
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(8, dtype=torch.int64).unsqueeze(0)
        )
        cache.token_to_kv_pool_allocator = allocator
        req = SimpleNamespace(
            origin_input_ids=list(range(8)),
            output_ids=[],
            extra_key=None,
            cache_salt=None,
            last_node=None,
            kv=SimpleNamespace(
                req_pool_idx=0,
                cache_protected_len=0,
                swa_evict_floor=4,
                swa_evicted_seqlen=0,
            ),
        )

        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=8)

        self.assertEqual(allocator.freed, [[0, 1, 2, 3], [4, 5, 6, 7]])


if __name__ == "__main__":
    unittest.main()
