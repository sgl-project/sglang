"""CPU-only tests for UnifiedRadixCache request lock lifecycle."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestUnifiedRadixLockRefScenarios(unittest.TestCase):
    def test_no_insert_without_last_node_skips_lock_release(self):
        cache = object.__new__(UnifiedRadixCache)
        cache.session = MagicMock()
        cache.session.try_cache_finished_req.return_value = False
        cache.disable = False
        cache.req_to_token_pool = MagicMock()
        cache.req_to_token_pool.req_to_token = torch.arange(8).reshape(1, 8)
        cache.free_kv_row = MagicMock()
        cache._dec_req_lock = MagicMock()
        cache._components_tuple = ()
        cache.enable_session_radix_cache = False

        kv = SimpleNamespace(req_pool_idx=0, cache_protected_len=0)
        req = SimpleNamespace(
            origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q"),
            kv=kv,
            last_node=None,
            swa_prefix_lock_released=False,
        )

        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=3)

        cache.free_kv_row.assert_called_once_with(kv, [(0, 3)])
        cache._dec_req_lock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
