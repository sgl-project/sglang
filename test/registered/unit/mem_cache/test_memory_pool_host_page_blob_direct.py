"""Unit tests for the page_blob_direct host layout."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost, MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class PageBlobDirectLayoutTest(unittest.TestCase):
    def _build_mha_pool(self) -> MHATokenToKVPoolHost:
        device_pool = SimpleNamespace(
            head_num=2,
            head_dim=3,
            layer_num=4,
            layer_shard_enabled=False,
            store_dtype=torch.float16,
            size=2,
            start_layer=0,
            end_layer=4,
            device="cpu",
        )
        return MHATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=2,
            layout="page_blob_direct",
            pin_memory=False,
            device="cpu",
        )

    def test_mha_page_blob_direct_keeps_page_blob_contiguous(self) -> None:
        pool = self._build_mha_pool()

        self.assertEqual(
            tuple(pool.kv_buffer.shape),
            (
                pool.page_num,
                2,
                pool.layer_num,
                pool.page_size,
                pool.head_num,
                pool.head_dim,
            ),
        )

        page = torch.arange(pool.get_dummy_flat_data_page().numel(), dtype=pool.dtype)
        pool.set_from_flat_data_page(0, page)

        self.assertTrue(torch.equal(pool.get_data_page(0, flat=True), page))

        first_page_indices = torch.tensor([0, 1], dtype=torch.int64)
        ptrs, sizes = pool.get_page_buffer_meta(first_page_indices)
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(sizes, [sizes[0], sizes[0]])
        self.assertEqual(ptrs[1] - ptrs[0], sizes[0])

        two_page_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        ptrs, sizes = pool.get_page_buffer_meta(two_page_indices)
        self.assertEqual(len(ptrs), 4)
        self.assertEqual(ptrs[2] - ptrs[0], 2 * sizes[0])

    def test_mla_page_blob_direct_matches_direct_page_shape(self) -> None:
        device_pool = SimpleNamespace(
            kv_lora_rank=8,
            qk_rope_head_dim=4,
            layer_num=3,
            layer_shard_enabled=False,
            store_dtype=torch.float16,
            size=2,
            start_layer=0,
            end_layer=3,
            device="cpu",
        )
        pool = MLATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=2,
            layout="page_blob_direct",
            pin_memory=False,
            device="cpu",
        )

        self.assertEqual(
            tuple(pool.kv_buffer.shape),
            (
                pool.page_num,
                pool.layer_num,
                pool.page_size,
                1,
                pool.kv_cache_dim,
            ),
        )

        page = torch.arange(pool.get_dummy_flat_data_page().numel(), dtype=pool.dtype)
        pool.set_from_flat_data_page(0, page)

        self.assertTrue(torch.equal(pool.get_data_page(0, flat=True), page))

        first_page_indices = torch.tensor([0, 1], dtype=torch.int64)
        ptrs, sizes = pool.get_page_buffer_meta(first_page_indices)
        self.assertEqual(len(ptrs), 1)
        self.assertEqual(len(sizes), 1)
        self.assertEqual(sizes[0], page.numel() * page.element_size())

        two_page_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        ptrs, sizes = pool.get_page_buffer_meta(two_page_indices)
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(ptrs[1] - ptrs[0], sizes[0])

    def test_mamba_page_blob_direct_uses_direct_page_shape(self) -> None:
        device_pool = SimpleNamespace(
            num_mamba_layers=2,
            size=3,
            device="cpu",
            mamba_cache=SimpleNamespace(
                temporal=torch.zeros((2, 3, 5), dtype=torch.float16),
                conv=[torch.zeros((2, 3, 4), dtype=torch.float16)],
            ),
        )
        pool = MambaPoolHost(
            device_pool,
            host_to_device_ratio=1.0,
            host_size=0,
            layout="page_blob_direct",
            pin_memory=False,
            device="cpu",
        )

        self.assertEqual(tuple(pool.temporal_buffer.shape), (pool.size, 2, 1, 5))
        self.assertEqual(tuple(pool.conv_buffer[0].shape), (pool.size, 2, 1, 4))

        page = torch.arange(pool.get_dummy_flat_data_page().numel(), dtype=torch.uint8)
        pool.set_from_flat_data_page(0, page)

        self.assertTrue(torch.equal(pool.get_data_page(0, flat=True), page))

        ptrs, sizes = pool.get_page_buffer_meta(torch.tensor([0], dtype=torch.int64))
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(len(sizes), 2)


if __name__ == "__main__":
    unittest.main()
