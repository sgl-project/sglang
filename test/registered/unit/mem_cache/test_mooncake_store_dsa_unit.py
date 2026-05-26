import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class FakeIndexerHostPool:
    def __init__(self, page_size=2, ptrs=None, sizes=None):
        self.page_size = page_size
        self.ptrs = ptrs or [111, 222]
        self.sizes = sizes or [11, 22]
        self.seen_indices = None
        self.buffers = [torch.zeros(8, dtype=torch.uint8)]

    def get_page_buffer_meta(self, indices):
        self.seen_indices = indices.clone()
        return self.ptrs, self.sizes

    def get_hybrid_pool_buffer(self):
        return self.buffers


class TestMooncakeStoreDsaIndexer(CustomTestCase):
    def _make_store(self):
        store = MooncakeStore.__new__(MooncakeStore)
        store.extra_backend_tag = None
        store.is_mla_backend = True
        store.mla_suffix = "0"
        store.mha_suffix = "0"
        store.storage_config = SimpleNamespace(should_split_heads=False)
        store.registered_pools = {}
        store.store = MagicMock()
        store.store.register_buffer.return_value = 0
        return store

    def test_register_mem_host_pool_v2_registers_indexer_buffer(self):
        store = self._make_store()
        host_pool = FakeIndexerHostPool()

        MooncakeStore.register_mem_host_pool_v2(store, host_pool, PoolName.INDEXER)

        self.assertIs(store.registered_pools[PoolName.INDEXER], host_pool)
        self.assertEqual(store.store.register_buffer.call_count, 1)

    def test_batch_exists_v2_requires_indexer_pages_for_prefix(self):
        store = self._make_store()
        store.extra_backend_tag = "tag"
        store.batch_exists = MagicMock(return_value=3)
        store._batch_exist = MagicMock(return_value=[1, 1, 0])

        transfer = PoolTransfer(
            name=PoolName.INDEXER,
            keys=["page0", "page1", "page2"],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )

        result = MooncakeStore.batch_exists_v2(
            store,
            ["page0", "page1", "page2"],
            pool_transfers=[transfer],
        )

        self.assertEqual(result.kv_hit_pages, 2)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.KV], 3)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.INDEXER], 2)
        store._batch_exist.assert_called_once_with(
            [
                "tag_page0_0_indexer",
                "tag_page1_0_indexer",
                "tag_page2_0_indexer",
            ]
        )

    def test_batch_get_v2_reads_indexer_pages_from_sidecar_pool(self):
        store = self._make_store()
        store.extra_backend_tag = "tag"
        host_pool = FakeIndexerHostPool(ptrs=[1001, 1002], sizes=[64, 64])
        store.registered_pools[PoolName.INDEXER] = host_pool
        store._get_batch_zero_copy_impl = MagicMock(return_value=[64, 64])

        transfer = PoolTransfer(
            name=PoolName.INDEXER,
            keys=["page0", "page1"],
            host_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        )

        result = MooncakeStore.batch_get_v2(store, [transfer])

        self.assertTrue(
            torch.equal(
                host_pool.seen_indices,
                torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            )
        )
        store._get_batch_zero_copy_impl.assert_called_once_with(
            ["tag_page0_0_indexer", "tag_page1_0_indexer"],
            [1001, 1002],
            [64, 64],
        )
        self.assertEqual(result[PoolName.INDEXER], [True, True])

    def test_batch_set_v2_writes_missing_indexer_pages_only(self):
        store = self._make_store()
        store.extra_backend_tag = "tag"
        host_pool = FakeIndexerHostPool(ptrs=[1001, 1002], sizes=[64, 64])
        store.registered_pools[PoolName.INDEXER] = host_pool
        store._batch_exist = MagicMock(return_value=[1, 0])
        store._put_batch_zero_copy_impl = MagicMock(return_value=[0])

        transfer = PoolTransfer(
            name=PoolName.INDEXER,
            keys=["page0", "page1"],
            host_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        )

        result = MooncakeStore.batch_set_v2(store, [transfer])

        store._batch_exist.assert_called_once_with(
            ["tag_page0_0_indexer", "tag_page1_0_indexer"]
        )
        store._put_batch_zero_copy_impl.assert_called_once_with(
            ["tag_page1_0_indexer"],
            [1002],
            host_pool.sizes[1:],
        )
        self.assertEqual(result[PoolName.INDEXER], [True, True])


if __name__ == "__main__":
    unittest.main()
