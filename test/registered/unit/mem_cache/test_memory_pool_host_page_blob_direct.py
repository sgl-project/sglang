"""Unit tests for the page_blob_direct host layout."""

# ruff: noqa: E402

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mem_cache.host_pool_test_utils import (
    install_memory_pool_host_layout_import_stubs,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

install_memory_pool_host_layout_import_stubs()

from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)


class PageBlobDirectLayoutTest(unittest.TestCase):
    def _build_mha_pool(self) -> MHATokenToKVPoolHost:
        device_pool = SimpleNamespace(
            head_num=2,
            head_dim=3,
            layer_num=4,
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

        page = torch.arange(pool.get_dummy_flat_data_page().numel(), dtype=pool.dtype)
        pool.set_from_flat_data_page(0, page)

        self.assertTrue(torch.equal(pool.get_data_page(0, flat=True), page))
        self.assertIsNone(pool.host_region_binding)

        ptrs, sizes = pool.get_page_buffer_meta(torch.tensor([0, 1], dtype=torch.int64))
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(sizes, [sizes[0], sizes[0]])
        self.assertEqual(ptrs[1] - ptrs[0], sizes[0])

    def test_tensorcast_allocator_backed_layout_is_validated_early(self) -> None:
        device_pool = SimpleNamespace(
            head_num=2,
            head_dim=3,
            layer_num=4,
            store_dtype=torch.float16,
            size=2,
            start_layer=0,
            end_layer=4,
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "page_blob_direct"):
            MHATokenToKVPoolHost(
                device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=2,
                layout="page_first_direct",
                pin_memory=False,
                device="cpu",
                allocator_type="tensorcast",
                allocator_config={
                    "host_allocator_enabled": True,
                    "daemon_address": "unix:///tmp/tensorcast.sock",
                },
            )

    def test_free_retires_resident_page_blob_direct_slot(self) -> None:
        pool = self._build_mha_pool()
        slot_tokens = pool.reserve_page_slots([0], logical_keys=["page-a"])
        pool.mark_page_get_inflight(slot_tokens)
        pool.commit_page_get_success(slot_tokens, logical_keys=["page-a"])

        pool.free(torch.tensor([0, 1], dtype=torch.int64))

        snapshot = pool.describe_page_slot(0)
        self.assertEqual(snapshot.state.value, "slot_free")
        self.assertEqual(snapshot.slot_generation, 1)

    def test_free_retires_invalid_page_blob_direct_slot(self) -> None:
        pool = self._build_mha_pool()
        slot_tokens = pool.reserve_page_slots([0], logical_keys=["page-a"])
        pool.mark_page_get_inflight(slot_tokens)
        pool.fail_page_get(slot_tokens)

        pool.free(torch.tensor([0, 1], dtype=torch.int64))

        snapshot = pool.describe_page_slot(0)
        self.assertEqual(snapshot.state.value, "slot_free")
        self.assertEqual(snapshot.slot_generation, 1)

    def test_mla_page_blob_direct_matches_existing_direct_page_shape(self) -> None:
        device_pool = SimpleNamespace(
            kv_lora_rank=8,
            qk_rope_head_dim=4,
            layer_num=3,
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

        page = torch.arange(pool.get_dummy_flat_data_page().numel(), dtype=pool.dtype)
        pool.set_from_flat_data_page(0, page)

        self.assertTrue(torch.equal(pool.get_data_page(0, flat=True), page))
        ptrs, sizes = pool.get_page_buffer_meta(torch.tensor([0, 1], dtype=torch.int64))
        self.assertEqual(len(ptrs), 1)
        self.assertEqual(len(sizes), 1)


if __name__ == "__main__":
    unittest.main()
