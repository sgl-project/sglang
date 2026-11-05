"""Shared HiSparse mapping and device-pool selection contracts."""

import unittest

import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
    HiSparseMHAMainPool,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiSparsePoolMapping(CustomTestCase):
    def test_mapping_identity_composition_and_dtype(self):
        mapping = torch.arange(16, dtype=torch.int64) + 100
        indices = torch.tensor([0, 3, 4, 7], dtype=torch.int64)
        for pool_type in (
            HiSparseDSATokenToKVPool,
            HiSparseC4DevicePool,
            HiSparseMHAMainPool,
        ):
            with self.subTest(pool=pool_type):
                pool = object.__new__(pool_type)
                pool.compress_ratio = 4 if pool_type is HiSparseC4DevicePool else 1
                pool.register_mapping(mapping)
                self.assertIs(pool.full_to_hisparse_device_index_mapping, mapping)
                compressed = pool.translate_loc_from_full_to_compressed(indices)
                expected = (
                    torch.tensor([0, 1])
                    if pool_type is HiSparseC4DevicePool
                    else indices
                )
                torch.testing.assert_close(compressed, expected)
                torch.testing.assert_close(
                    pool.translate_loc_from_full_to_hisparse_device(indices),
                    mapping[expected],
                )
                public_dtype = (
                    torch.int32 if pool_type is HiSparseC4DevicePool else torch.int64
                )
                torch.testing.assert_close(
                    pool.translate_loc_to_hisparse_device(compressed),
                    mapping[expected].to(public_dtype),
                )
                torch.testing.assert_close(
                    pool._translate_loc_to_hisparse_device(compressed),
                    mapping[expected],
                )

    def test_mha_still_requires_registered_mapping(self):
        pool = object.__new__(HiSparseMHAMainPool)
        pool.full_to_hisparse_device_index_mapping = None
        for method in (
            pool.translate_loc_to_hisparse_device,
            pool._translate_loc_to_hisparse_device,
            pool.translate_loc_from_full_to_hisparse_device,
        ):
            with self.subTest(method=method), self.assertRaises(AssertionError):
                method(torch.tensor([0]))

    def test_selects_device_pool_without_changing_allocator_kvcache(self):
        dsa = object.__new__(HiSparseDSATokenToKVPool)
        mha = object.__new__(HiSparseMHAMainPool)
        minimax = object.__new__(MiniMaxSparseKVPool)
        minimax.main_pool = mha
        for kvcache, expected in ((dsa, dsa), (minimax, mha)):
            with self.subTest(kvcache=type(kvcache)):
                allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
                allocator._kvcache = kvcache
                self.assertIs(allocator.get_kvcache(), kvcache)
                self.assertIs(allocator.get_hisparse_kvcache(), expected)

        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator._kvcache = object()
        allocator.hisparse_kvcache = object.__new__(HiSparseC4DevicePool)
        self.assertIs(allocator.get_hisparse_kvcache(), allocator.hisparse_kvcache)
        self.assertIsNot(allocator.get_hisparse_kvcache(), allocator.get_kvcache())


if __name__ == "__main__":
    unittest.main()
