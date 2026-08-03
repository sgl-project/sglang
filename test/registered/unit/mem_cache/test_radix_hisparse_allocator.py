import unittest

import torch

from sglang.srt.mem_cache.allocator.radix_hisparse import (
    RadixHiSparseTokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeHiSparsePool:
    def register_mapping(self, mapping):
        self.mapping = mapping


class _FakeHostPool:
    def __init__(self, size):
        self.size = size
        self.destroyed = False
        self.destroy_count = 0

    def alloc_page(self, num_pages):
        raise AssertionError("host allocation escaped the L1 view")

    def clear(self):
        raise AssertionError("host clear escaped the L1 view")

    def destroy(self):
        self.destroyed = True
        self.destroy_count += 1


class TestRadixHiSparseAllocatorSplit(unittest.TestCase):
    def setUp(self):
        self.pool = _FakeHiSparsePool()
        self.host_pool = _FakeHostPool(size=26)
        self.allocator = RadixHiSparseTokenToKVPoolAllocator(
            l0_capacity=8,
            page_size=2,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            kvcache=self.pool,
            need_sort=False,
            host_to_device_ratio=3,
            host_pool=self.host_pool,
        )
        self.l1 = self.allocator.l1_allocator
        self.l0 = self.allocator.l0_allocator

    def test_l1_and_l0_capacity_are_independent(self):
        self.assertEqual(self.allocator.l1_capacity, 24)
        self.assertEqual(self.allocator.l1_available_size(), 24)
        self.assertEqual(self.l0.available_size(), 8)
        self.assertIs(self.allocator.logical_attn_allocator, self.l1)
        self.assertIs(self.allocator.hisparse_attn_allocator, self.l0.l0_slot_allocator)

        l1_indices = self.allocator.alloc(4)
        self.assertIsNotNone(l1_indices)
        self.assertEqual(self.allocator.l1_available_size(), 20)
        self.assertEqual(self.l0.available_size(), 8)

        l0_indices = self.l0.alloc(8)
        self.assertIsNotNone(l0_indices)
        self.assertEqual(self.l0.available_size(), 0)
        self.assertEqual(self.l1.l1_available_size(), 20)

        self.l0.free(l0_indices)
        self.assertEqual(self.l0.available_size(), 8)
        self.assertEqual(self.allocator.l1_available_size(), 20)

        self.allocator.free(l1_indices)
        self.assertEqual(self.allocator.l1_available_size(), 24)
        self.assertEqual(self.l0.available_size(), 8)

    def test_l1_uses_identity_addressing(self):
        l1_indices = self.allocator.alloc(4)

        self.assertIs(self.allocator.full_kv_host_locs(l1_indices), l1_indices)
        self.assertIs(self.allocator.index_k_device_locs(l1_indices), l1_indices)
        self.assertIs(self.allocator.get_last_loc_compressed(l1_indices), l1_indices)

    def test_l0_reuses_initial_routes_without_taking_l1_ownership(self):
        l1_indices = self.allocator.alloc(4)
        initial_l0_indices = self.l0.alloc(4)
        self.l0.bind_write_locs(l1_indices, initial_l0_indices)

        self.assertTrue(
            torch.equal(self.l0.lookup_write_locs(l1_indices), initial_l0_indices)
        )
        request_l0_buffer = self.l0.acquire_request_l0_buffer(l1_indices, need_size=6)

        self.assertIsNotNone(request_l0_buffer)
        self.assertEqual(request_l0_buffer.numel(), 6)
        self.assertTrue(
            torch.equal(
                self.l0.lookup_write_locs(l1_indices),
                torch.zeros_like(l1_indices),
            )
        )
        self.assertEqual(self.l0.available_size(), 2)
        self.assertEqual(self.allocator.l1_available_size(), 20)

        self.l0.free(request_l0_buffer)
        self.assertEqual(self.l0.available_size(), 8)
        self.assertEqual(self.allocator.l1_available_size(), 20)

    def test_failed_l0_growth_preserves_write_routes(self):
        l1_indices = self.allocator.alloc(2)
        initial_l0_indices = self.l0.alloc(2)
        self.l0.bind_write_locs(l1_indices, initial_l0_indices)
        remaining_l0_indices = self.l0.alloc(6)

        request_l0_buffer = self.l0.acquire_request_l0_buffer(l1_indices, need_size=4)

        self.assertIsNone(request_l0_buffer)
        self.assertTrue(
            torch.equal(self.l0.lookup_write_locs(l1_indices), initial_l0_indices)
        )
        self.assertEqual(self.allocator.l1_available_size(), 22)

        self.l0.free(remaining_l0_indices)
        self.l0.release_write_locs(l1_indices)
        self.assertEqual(self.l0.available_size(), 8)
        self.assertEqual(self.allocator.l1_available_size(), 22)

    def test_facade_exposes_borrowed_cpu_l1_pool(self):
        pool_view = self.allocator.get_l1_host_pool()
        self.assertEqual(pool_view.storage_size, 26)
        self.assertEqual(pool_view.size, 24)
        self.assertEqual(pool_view.available_size(), 24)

        l1_indices = self.allocator.alloc(4)
        self.assertEqual(pool_view.available_size(), 20)
        self.allocator.free(l1_indices)

        with self.assertRaisesRegex(RuntimeError, "identity L1 indices"):
            pool_view.alloc_paged_token_slots()
        with self.assertRaisesRegex(AttributeError, "does not expose"):
            pool_view.alloc_page(1)
        with self.assertRaisesRegex(AttributeError, "does not expose"):
            pool_view.clear()
        with self.assertRaisesRegex(AttributeError, "does not expose"):
            pool_view.destroy()

        self.allocator.destroy()
        self.assertFalse(self.host_pool.destroyed)

    def test_resize_cannot_outgrow_identity_host_storage(self):
        too_large = type("Config", (), {"max_total_num_tokens": 10})()
        with self.assertRaisesRegex(ValueError, "cannot cover resized"):
            self.allocator.resize(too_large)

        smaller = type("Config", (), {"max_total_num_tokens": 6})()
        self.allocator.resize(smaller)
        self.assertEqual(self.allocator.l1_capacity, 18)
        self.assertEqual(self.allocator.l0_capacity, 6)
        self.assertEqual(self.allocator.get_l1_host_pool().size, 18)

    def test_owned_host_storage_is_destroyed_once(self):
        self.allocator._owns_l1_host_pool = True
        self.allocator.destroy()
        self.allocator.destroy()
        self.assertEqual(self.host_pool.destroy_count, 1)


if __name__ == "__main__":
    unittest.main()
