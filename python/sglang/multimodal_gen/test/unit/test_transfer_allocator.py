# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BuddyAllocator."""

import threading
import unittest

from sglang.multimodal_gen.runtime.disaggregation.transport.allocator import (
    BuddyAllocator,
)


class TestBuddyAllocatorInit(unittest.TestCase):
    """Test allocator initialization."""

    def test_power_of_2_pool(self):
        alloc = BuddyAllocator(pool_size=100, min_block_size=16)
        self.assertEqual(alloc.pool_size, 128)  # next power of 2
        self.assertEqual(alloc.min_block_size, 16)

    def test_exact_power_of_2(self):
        alloc = BuddyAllocator(pool_size=256, min_block_size=64)
        self.assertEqual(alloc.pool_size, 256)

    def test_invalid_min_block(self):
        with self.assertRaises(ValueError):
            BuddyAllocator(pool_size=256, min_block_size=3)

    def test_initial_state(self):
        alloc = BuddyAllocator(pool_size=1024, min_block_size=64)
        self.assertEqual(alloc.free_bytes, 1024)
        self.assertEqual(alloc.allocated_bytes, 0)
        self.assertEqual(alloc.num_allocations, 0)


class TestBuddyAllocatorAlloc(unittest.TestCase):
    """Test allocation and splitting."""

    def setUp(self):
        # 1024 bytes, min block 64
        self.alloc = BuddyAllocator(pool_size=1024, min_block_size=64)

    def test_allocate_min_size(self):
        offset = self.alloc.allocate(1)  # rounds to min_block_size
        self.assertIsNotNone(offset)
        self.assertEqual(self.alloc.allocated_bytes, 64)

    def test_allocate_exact_min(self):
        offset = self.alloc.allocate(64)
        self.assertIsNotNone(offset)
        self.assertEqual(self.alloc.allocated_bytes, 64)

    def test_allocate_rounds_up(self):
        offset = self.alloc.allocate(100)  # rounds to 128
        self.assertIsNotNone(offset)
        self.assertEqual(self.alloc.allocated_bytes, 128)

    def test_allocate_full_pool(self):
        offset = self.alloc.allocate(1024)
        self.assertIsNotNone(offset)
        self.assertEqual(self.alloc.allocated_bytes, 1024)
        self.assertEqual(self.alloc.free_bytes, 0)

    def test_allocate_exceeds_pool(self):
        offset = self.alloc.allocate(2048)
        self.assertIsNone(offset)

    def test_allocate_until_full(self):
        offsets = []
        for _ in range(16):  # 16 * 64 = 1024
            o = self.alloc.allocate(64)
            self.assertIsNotNone(o)
            offsets.append(o)
        # Pool is full
        self.assertIsNone(self.alloc.allocate(64))
        self.assertEqual(len(set(offsets)), 16)  # all unique

    def test_allocate_with_request_id(self):
        offset = self.alloc.allocate(64, request_id="req-1")
        block = self.alloc.get_block_info(offset)
        self.assertEqual(block.request_id, "req-1")
        self.assertTrue(block.allocated)

    def test_allocate_zero_raises(self):
        with self.assertRaises(ValueError):
            self.alloc.allocate(0)

    def test_splitting_creates_correct_blocks(self):
        """Allocating 64 from a 1024 pool should split: 1024→512+512→...→64+64."""
        o1 = self.alloc.allocate(64)
        self.assertEqual(o1, 0)
        self.assertEqual(self.alloc.allocated_bytes, 64)

        # Second 64-byte alloc should get offset 64 (buddy of first block)
        o2 = self.alloc.allocate(64)
        self.assertEqual(o2, 64)


class TestBuddyAllocatorFree(unittest.TestCase):
    """Test free and coalescing."""

    def setUp(self):
        self.alloc = BuddyAllocator(pool_size=256, min_block_size=64)

    def test_free_basic(self):
        offset = self.alloc.allocate(64)
        self.assertTrue(self.alloc.free(offset))
        self.assertEqual(self.alloc.free_bytes, 256)
        self.assertEqual(self.alloc.num_allocations, 0)

    def test_free_invalid_offset(self):
        self.assertFalse(self.alloc.free(9999))

    def test_free_already_free(self):
        offset = self.alloc.allocate(64)
        self.assertTrue(self.alloc.free(offset))
        self.assertFalse(self.alloc.free(offset))

    def test_coalesce_buddies(self):
        """Two adjacent 64-byte blocks should coalesce to 128 when both freed."""
        o1 = self.alloc.allocate(64)  # 0
        o2 = self.alloc.allocate(64)  # 64

        self.alloc.free(o1)
        self.alloc.free(o2)

        # Should be able to allocate 128 now (coalesced)
        o3 = self.alloc.allocate(128)
        self.assertIsNotNone(o3)

    def test_full_coalesce(self):
        """Allocate 4x64, free all, should coalesce back to one 256 block."""
        offsets = [self.alloc.allocate(64) for _ in range(4)]
        for o in offsets:
            self.alloc.free(o)

        # Full pool available as single block
        o = self.alloc.allocate(256)
        self.assertIsNotNone(o)
        self.assertEqual(self.alloc.allocated_bytes, 256)

    def test_partial_coalesce(self):
        """Free non-adjacent blocks — no coalesce."""
        o1 = self.alloc.allocate(64)  # 0
        o2 = self.alloc.allocate(64)  # 64
        o3 = self.alloc.allocate(64)  # 128
        o4 = self.alloc.allocate(64)  # 192

        self.alloc.free(o1)  # free 0
        self.alloc.free(o3)  # free 128

        # Can't allocate 128 — blocks are not adjacent buddies at order 1
        # o1 (0) buddies with o2 (64), o3 (128) buddies with o4 (192)
        # Since o2 and o4 are still allocated, no coalescing happens
        stats = self.alloc.get_stats()
        self.assertEqual(stats["num_allocations"], 2)


class TestBuddyAllocatorSlotCount(unittest.TestCase):
    """Test free slot counting."""

    def test_count_empty_pool(self):
        alloc = BuddyAllocator(pool_size=256, min_block_size=64)
        self.assertEqual(alloc.count_free_slots(64), 4)

    def test_count_after_alloc(self):
        alloc = BuddyAllocator(pool_size=256, min_block_size=64)
        alloc.allocate(64)
        self.assertEqual(alloc.count_free_slots(64), 3)

    def test_count_larger_slots(self):
        alloc = BuddyAllocator(pool_size=256, min_block_size=64)
        self.assertEqual(alloc.count_free_slots(128), 2)

    def test_can_allocate(self):
        alloc = BuddyAllocator(pool_size=256, min_block_size=64)
        self.assertTrue(alloc.can_allocate(64))
        self.assertTrue(alloc.can_allocate(256))
        self.assertFalse(alloc.can_allocate(512))


class TestBuddyAllocatorThreadSafety(unittest.TestCase):
    """Test concurrent allocation/deallocation."""

    def test_concurrent_alloc_free(self):
        alloc = BuddyAllocator(pool_size=1 << 20, min_block_size=1024)
        results = []
        errors = []

        def worker():
            try:
                for _ in range(50):
                    o = alloc.allocate(
                        1024, request_id=f"t-{threading.current_thread().name}"
                    )
                    if o is not None:
                        results.append(o)
                        alloc.free(o)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # After all frees, pool should be fully available
        self.assertEqual(alloc.free_bytes, alloc.pool_size)


class TestBuddyAllocatorRealisticSizes(unittest.TestCase):
    """Test with realistic diffusion tensor sizes."""

    def test_encoder_denoiser_slots(self):
        """Encoder→Denoiser: ~60MB per request. Pool for 4 concurrent requests."""
        pool_size = 256 << 20  # 256 MiB
        alloc = BuddyAllocator(pool_size=pool_size, min_block_size=1 << 20)

        # Allocate 4 x 64MiB slots
        request_size = 64 << 20
        offsets = []
        for i in range(4):
            o = alloc.allocate(request_size, request_id=f"req-{i}")
            self.assertIsNotNone(o, f"Failed to allocate slot {i}")
            offsets.append(o)

        # Pool should be full (4 x 64MiB = 256MiB)
        self.assertIsNone(alloc.allocate(request_size))

        # Free one, allocate again
        alloc.free(offsets[0])
        o = alloc.allocate(request_size, request_id="req-new")
        self.assertIsNotNone(o)


if __name__ == "__main__":
    unittest.main()
