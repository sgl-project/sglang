"""Unit tests for page_first_direct host layout physical-format preferences."""

# ruff: noqa: E402

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mem_cache.host_pool_test_utils import (
    install_memory_pool_host_layout_import_stubs,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

install_memory_pool_host_layout_import_stubs()

import sglang.srt.mem_cache.memory_pool_host as memory_pool_host
from sglang.srt.mem_cache.memory_pool_host import (
    HostTensorAllocator,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    PageFirstDirectMhaFormat,
    resolve_page_first_direct_mha_format,
)


class KvContiguousHostTensorAllocator(HostTensorAllocator):
    @property
    def page_first_direct_mha_format(self) -> PageFirstDirectMhaFormat:
        return PageFirstDirectMhaFormat.KV_CONTIGUOUS_PAGE_SLOT


class RecordingHostPageSlotLifecycleManager:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.retire_calls: list[torch.Tensor] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def retire_released_page_slots(self, indices: torch.Tensor) -> None:
        self.retire_calls.append(indices.clone())


class PageFirstDirectLayoutTest(unittest.TestCase):
    def _build_mha_pool(
        self,
        *,
        allocator: HostTensorAllocator | None = None,
    ) -> MHATokenToKVPoolHost:
        device_pool = SimpleNamespace(
            head_num=2,
            head_dim=3,
            layer_num=4,
            store_dtype=torch.float16,
            size=8,
            start_layer=0,
            end_layer=4,
            device="cpu",
        )
        kwargs = {
            "device_pool": device_pool,
            "host_to_device_ratio": 2.0,
            "host_size": 0,
            "page_size": 2,
            "layout": "page_first_direct",
            "pin_memory": False,
            "device": "cpu",
        }
        if allocator is None:
            return MHATokenToKVPoolHost(**kwargs)
        with patch.object(
            memory_pool_host,
            "get_allocator_from_storage",
            return_value=allocator,
        ):
            return MHATokenToKVPoolHost(**kwargs)

    def test_default_mha_page_first_direct_keeps_split_kv_planes(self) -> None:
        pool = self._build_mha_pool()

        self.assertEqual(
            tuple(pool.kv_buffer.shape),
            (
                2,
                pool.page_num,
                pool.layer_num,
                pool.page_size,
                pool.head_num,
                pool.head_dim,
            ),
        )
        self.assertEqual(
            resolve_page_first_direct_mha_format(pool.allocator),
            PageFirstDirectMhaFormat.SPLIT_KV_PLANES,
        )
        self.assertEqual(
            tuple(pool.k_buffer.shape),
            (
                pool.page_num,
                pool.layer_num,
                pool.page_size,
                pool.head_num,
                pool.head_dim,
            ),
        )
        self.assertEqual(tuple(pool.v_buffer.shape), tuple(pool.k_buffer.shape))

        page = torch.arange(pool.get_dummy_flat_data_page().numel(), dtype=pool.dtype)
        pool.set_from_flat_data_page(0, page)

        self.assertTrue(torch.equal(pool.get_data_page(0, flat=True), page))
        self.assertIsNone(pool.host_region_binding)

        ptrs, sizes = pool.get_page_buffer_meta(
            torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        )
        self.assertEqual(len(ptrs), 4)
        self.assertEqual(sizes, [sizes[0]] * 4)
        self.assertNotEqual(ptrs[1] - ptrs[0], sizes[0])
        self.assertEqual(ptrs[2] - ptrs[0], sizes[0])

    def test_tensorcast_preference_mha_page_first_direct_uses_kv_contiguous_slot(
        self,
    ) -> None:
        allocator = KvContiguousHostTensorAllocator()
        pool = self._build_mha_pool(allocator=allocator)

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
        self.assertEqual(allocator.dims, tuple(pool.kv_buffer.shape))
        self.assertEqual(
            resolve_page_first_direct_mha_format(allocator),
            PageFirstDirectMhaFormat.KV_CONTIGUOUS_PAGE_SLOT,
        )
        self.assertEqual(
            tuple(pool.k_buffer.shape),
            (
                pool.page_num,
                pool.layer_num,
                pool.page_size,
                pool.head_num,
                pool.head_dim,
            ),
        )
        self.assertEqual(tuple(pool.v_buffer.shape), tuple(pool.k_buffer.shape))

        page = torch.arange(pool.get_dummy_flat_data_page().numel(), dtype=pool.dtype)
        pool.set_from_flat_data_page(0, page)

        self.assertTrue(torch.equal(pool.get_data_page(0, flat=True), page))

        ptrs, sizes = pool.get_page_buffer_meta(
            torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        )
        self.assertEqual(len(ptrs), 4)
        self.assertEqual(sizes, [sizes[0]] * 4)
        self.assertEqual(ptrs[1] - ptrs[0], sizes[0])
        self.assertEqual(ptrs[2] - ptrs[0], 2 * sizes[0])

    def test_tensorcast_allocator_rejects_non_page_first_direct_layout(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "page_first_direct"):
            MHATokenToKVPoolHost(
                device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=2,
                layout="page_first",
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

    def test_mla_page_first_direct_matches_existing_direct_page_shape(self) -> None:
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
            layout="page_first_direct",
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
