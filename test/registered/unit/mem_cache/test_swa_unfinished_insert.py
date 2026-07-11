import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertResult
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSWAUnfinishedInsert(unittest.TestCase):
    def test_forwards_evicted_boundary_to_insert(self):
        cache = object.__new__(SWARadixCache)
        cache.disable = False
        cache.page_size = 1
        cache.is_eagle = False
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(4, dtype=torch.int64).reshape(1, 4),
            write=Mock(),
        )
        cache.insert = Mock(return_value=InsertResult(prefix_len=4))
        last_node = object()
        cache.match_prefix = Mock(
            return_value=SimpleNamespace(
                device_indices=torch.arange(4, dtype=torch.int64),
                last_device_node=last_node,
            )
        )
        cache.dec_lock_ref = Mock()
        cache.inc_lock_ref = Mock(
            return_value=SimpleNamespace(swa_uuid_for_lock="new-lock")
        )

        req = SimpleNamespace(
            req_pool_idx=0,
            extra_key=None,
            cache_protected_len=0,
            swa_evicted_seqlen=2,
            last_node=object(),
            swa_uuid_for_lock=None,
            swa_prefix_lock_released=False,
            prefix_indices=torch.empty(0, dtype=torch.int64),
        )
        req.get_fill_ids = lambda: [10, 11, 12, 13]

        cache.cache_unfinished_req(req, chunked=True)

        insert_params = cache.insert.call_args.args[0]
        self.assertEqual(insert_params.swa_evicted_seqlen, 2)
        self.assertIs(req.last_node, last_node)
        self.assertEqual(req.swa_uuid_for_lock, "new-lock")


if __name__ == "__main__":
    unittest.main()
