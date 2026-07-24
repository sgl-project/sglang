import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecodeKVCacheOffloadManager(unittest.TestCase):
    def test_host_write_failure_does_not_create_offload_state(self):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        req = SimpleNamespace(
            rid="rid-1",
            req_pool_idx=0,
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[5, 6],
        )
        manager.cache_controller = SimpleNamespace(write=lambda **kwargs: None)
        manager.decode_host_mem_pool = object()
        manager.page_size = 2
        manager.offload_stride = 1
        manager.request_counter = 0
        manager.offloaded_state = {}
        manager.ongoing_offload = {}
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, 16)
        )
        manager._compute_prefix_hash = lambda tokens, prior_hash="": ["h"]

        self.assertFalse(manager.offload_kv_cache(req))

        self.assertEqual(manager.ongoing_offload, {})
        self.assertNotIn("rid-1", manager.offloaded_state)


if __name__ == "__main__":
    unittest.main()
