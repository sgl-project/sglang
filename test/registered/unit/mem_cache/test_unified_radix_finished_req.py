"""CPU-only tests for UnifiedRadixCache finished-request cleanup."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnifiedRadixFinishedReq(unittest.TestCase):
    def test_truncation_coalesces_adjacent_free_ranges(self):
        cache = object.__new__(UnifiedRadixCache)
        cache.session = MagicMock()
        cache.session.try_cache_finished_req.return_value = False
        cache.disable = False
        cache.tree_core = MagicMock()
        cache.tree_core.is_eagle = False
        cache.page_size = 8
        cache.req_to_token_pool = MagicMock()
        cache.req_to_token_pool.req_to_token = torch.arange(19).reshape(1, 19)
        cache.free_kv_row = MagicMock()
        cache._dec_req_lock = MagicMock()
        cache.enable_session_radix_cache = False

        component = MagicMock()
        component.prepare_for_caching_req.return_value = 13
        component.component_type = object()
        cache._components_tuple = (component,)
        cache.insert = MagicMock(
            return_value=SimpleNamespace(
                rotation_tail_declined=False,
                last_device_node=None,
            )
        )

        kv = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=0,
        )
        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=array("q", range(19)),
            output_ids=array("q"),
            kv=kv,
            kv_rotation_base=None,
            extra_key=None,
            cache_salt=None,
            last_node=object(),
            lock_receipt=MagicMock(),
            swa_prefix_lock_released=False,
        )

        cache.cache_finished_req(req, is_insert=True, kv_len_to_handle=19)

        # The page-unaligned radix tail [8, 13) and truncation tail [13, 19)
        # may share an allocator page, so they must be released as one span.
        cache.free_kv_row.assert_called_once_with(kv, [(8, 19)])


if __name__ == "__main__":
    unittest.main()
