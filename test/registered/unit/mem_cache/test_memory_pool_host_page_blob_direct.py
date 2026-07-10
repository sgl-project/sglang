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


class RecordingHostPageSlotLifecycleManager:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.retire_calls: list[torch.Tensor] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def retire_released_page_slots(self, indices: torch.Tensor) -> None:
        self.retire_calls.append(indices.clone())


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

    def test_free_forwards_page_slot_retirement_to_attached_manager(self) -> None:
        pool = self._build_mha_pool()
        manager = RecordingHostPageSlotLifecycleManager()
        pool.attach_host_page_slot_lifecycle_manager(manager)

        indices = torch.tensor([0, 1], dtype=torch.int64)
        self.assertEqual(pool.free(indices), 2)

        self.assertEqual(len(manager.retire_calls), 1)
        self.assertTrue(torch.equal(manager.retire_calls[0], indices))

    def test_clear_resets_attached_page_slot_manager(self) -> None:
        pool = self._build_mha_pool()
        manager = RecordingHostPageSlotLifecycleManager()
        pool.attach_host_page_slot_lifecycle_manager(manager)

        pool.clear()

        self.assertEqual(manager.reset_calls, 1)

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
