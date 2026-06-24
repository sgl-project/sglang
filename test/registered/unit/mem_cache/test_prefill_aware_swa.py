import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self):
        self.freed = []

    def free_swa(self, indices):
        self.freed.append(indices.clone())


class TestPrefillAwareSWA(CustomTestCase):
    def test_eviction_preserves_prefill(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=0,
            swa_evict_floor=10,
            swa_evicted_seqlen=0,
        )
        pool = SimpleNamespace(req_to_token=torch.arange(32).view(1, 32))
        allocator = _Allocator()

        free_swa_out_of_window_slots(
            req,
            pre_len=20,
            sliding_window_size=4,
            page_size=1,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            drop_page_margin=True,
        )

        self.assertEqual(req.swa_evicted_seqlen, 16)
        self.assertEqual(allocator.freed[0].tolist(), list(range(10, 16)))

    def test_finished_request_caches_only_prefill(self):
        component = SWAComponent.__new__(SWAComponent)
        req = SimpleNamespace(
            swa_evict_floor=10,
            swa_evicted_seqlen=16,
        )
        insert_params = InsertParams()

        cache_len = component.prepare_for_caching_req(
            req,
            insert_params,
            token_ids_len=24,
            is_finished=True,
        )

        self.assertEqual(cache_len, 10)
        self.assertEqual(insert_params.swa_evicted_seqlen, 0)


if __name__ == "__main__":
    unittest.main()
