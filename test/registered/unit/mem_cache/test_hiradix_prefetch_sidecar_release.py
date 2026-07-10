import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiRadixPrefetchSidecarRelease(unittest.TestCase):
    def test_releases_kv_tail_beyond_sidecar_coverage(self):
        page_size = 4
        host_indices = torch.arange(12)
        pool_storage_result = PoolTransferResult.empty()
        pool_storage_result.update_extra_pool_hit_pages(
            {PoolName.INDEXER: [True, False, True]}
        )
        operation = SimpleNamespace(
            host_indices=host_indices,
            pool_storage_result=pool_storage_result,
            pool_transfers=[PoolTransfer(name=PoolName.INDEXER)],
        )
        last_host_node = SimpleNamespace(release_host=mock.Mock())
        controller = SimpleNamespace(
            terminate_prefetch=mock.Mock(
                return_value=(12, ["page-0", "page-1", "page-2"])
            ),
            mem_pool_host=SimpleNamespace(free=mock.Mock()),
            append_host_mem_release=mock.Mock(),
            prefetch_tokens_occupied=12,
        )
        cache = object.__new__(HiRadixCache)
        cache.page_size = page_size
        cache.cache_controller = controller
        cache.ongoing_prefetch = {
            "request": (
                last_host_node,
                list(range(12)),
                host_indices,
                operation,
            )
        }
        cache.can_terminate_prefetch = mock.Mock(return_value=True)
        cache._all_reduce_attn_groups = mock.Mock()
        cache._insert_helper_host = mock.Mock(return_value=0)
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.enable_storage_metrics = False

        self.assertTrue(cache.check_prefetch_progress("request"))

        inserted_indices = cache._insert_helper_host.call_args.args[2]
        released_indices = controller.append_host_mem_release.call_args.args[0]
        self.assertTrue(torch.equal(inserted_indices, host_indices[:page_size]))
        self.assertTrue(torch.equal(released_indices, host_indices[page_size:]))


if __name__ == "__main__":
    unittest.main()
