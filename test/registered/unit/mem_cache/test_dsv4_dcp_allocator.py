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


if __name__ == "__main__":
    unittest.main()