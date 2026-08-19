import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.swa import (
    DeepSeekV4DCPTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV4DCPAllocator(CustomTestCase):
    def _pool(self, supported: bool):
        pool = MagicMock(spec=BaseSWAKVPool)
        pool.supports_dsv4_dcp = supported
        pool.full_kv_pool = None
        pool.swa_kv_pool = None
        return pool

    def test_rejects_physical_pool_without_dcp_translation(self):
        with self.assertRaisesRegex(NotImplementedError, "unified-KV pool"):
            DeepSeekV4DCPTokenToKVPoolAllocator(
                physical_size_full=1024,
                physical_size_swa=512,
                physical_page_size=256,
                dcp_size=4,
                dcp_rank=1,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=self._pool(False),
                need_sort=False,
            )

    def test_constructs_one_logical_page_per_physical_page(self):
        allocator = DeepSeekV4DCPTokenToKVPoolAllocator(
            physical_size_full=1024,
            physical_size_swa=512,
            physical_page_size=256,
            dcp_size=4,
            dcp_rank=2,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=self._pool(True),
            need_sort=False,
        )
        self.assertEqual(allocator.page_size, 1024)
        self.assertEqual(allocator.size_full, 4096)
        self.assertEqual(allocator.size_swa, 2048)
        self.assertEqual(allocator.full_attn_allocator.num_pages, 4)
        self.assertEqual(allocator.swa_attn_allocator.num_pages, 2)
        self.assertEqual(allocator.full_to_swa_index_mapping.numel(), 5121)
        with self.assertRaisesRegex(NotImplementedError, "resizing"):
            allocator.resize(object())

    def test_widened_ids_cross_old_physical_range_and_localize(self):
        physical_size = 32768
        physical_page_size = 256
        dcp_size = 8
        logical_pages = 64
        allocator = DeepSeekV4DCPTokenToKVPoolAllocator(
            physical_size_full=physical_size,
            physical_size_swa=physical_size,
            physical_page_size=physical_page_size,
            dcp_size=dcp_size,
            dcp_rank=0,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=self._pool(True),
            need_sort=False,
        )

        logical_ids = allocator.full_attn_allocator.alloc(
            logical_pages * allocator.page_size
        )
        self.assertIsNotNone(logical_ids)
        self.assertGreater(int(logical_ids.max()), physical_size)
        expected_local_rows = torch.arange(
            physical_page_size,
            physical_page_size + logical_pages * physical_page_size,
        )
        for rank in range(dcp_size):
            owned = logical_ids[logical_ids % dcp_size == rank]
            local_rows = owned // dcp_size
            torch.testing.assert_close(local_rows, expected_local_rows)
            self.assertLess(int(local_rows.max()), physical_size)


if __name__ == "__main__":
    unittest.main()