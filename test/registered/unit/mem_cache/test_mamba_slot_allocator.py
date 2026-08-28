import unittest

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMambaSlotAllocator(unittest.TestCase):
    def test_schedulable_count_includes_group_preallocation(self):
        allocator = MambaSlotAllocator(size=5, device="cpu")

        allocator.alloc_group_begin(3)
        self.assertEqual(allocator.available_size(), 2)
        self.assertEqual(allocator.schedulable_available_size(), 5)

        self.assertIsNotNone(allocator.alloc(1))
        self.assertEqual(allocator.available_size(), 2)
        self.assertEqual(allocator.schedulable_available_size(), 4)

        allocator.alloc_group_end()
        self.assertEqual(allocator.available_size(), 4)
        self.assertEqual(allocator.schedulable_available_size(), 4)

    def test_clear_returns_group_preallocation_to_logical_capacity(self):
        allocator = MambaSlotAllocator(size=5, device="cpu")
        allocator.alloc_group_begin(3)

        allocator.clear()

        self.assertEqual(allocator.available_size(), 5)
        self.assertEqual(allocator.schedulable_available_size(), 5)


if __name__ == "__main__":
    unittest.main()
