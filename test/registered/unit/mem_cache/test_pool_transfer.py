"""Unit tests for shared cache transfer types — no server or model loading."""

import unittest

import torch

from sglang.srt.mem_cache import hicache_storage
from sglang.srt.mem_cache.pool_transfer import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPoolTransfer(CustomTestCase):
    def test_legacy_hicache_storage_exports_are_preserved(self):
        self.assertIs(hicache_storage.PoolHitPolicy, PoolHitPolicy)
        self.assertIs(hicache_storage.PoolName, PoolName)
        self.assertIs(hicache_storage.PoolTransfer, PoolTransfer)
        self.assertIs(hicache_storage.PoolTransferResult, PoolTransferResult)

    def test_pool_transfer_defaults_and_cpu_indices(self):
        host_indices = torch.tensor([1, 2])
        transfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)

        self.assertEqual(str(transfer.name), "kv")
        self.assertEqual(transfer.host_indices.device.type, "cpu")
        self.assertTrue(torch.equal(transfer.host_indices, host_indices))
        self.assertIsNone(transfer.device_indices)
        self.assertIsNone(transfer.keys)
        self.assertEqual(transfer.hit_policy, PoolHitPolicy.ALL_PAGES)
        self.assertIsNone(transfer.nodes_to_load)
        self.assertIsNone(transfer.indices_from_pool)

    def test_pool_transfer_result_updates(self):
        result = PoolTransferResult.empty()

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.extra_pool_hit_pages, {})
        self.assertIsNone(result.restorable_prefix_pages)

        result.update_kv_hit_pages(3)
        result.update_kv_hit_pages(2)
        result.update_extra_pool_hit_pages({str(PoolName.MAMBA): 2})

        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(result.extra_pool_hit_pages, {"mamba": 2})


if __name__ == "__main__":
    unittest.main()
