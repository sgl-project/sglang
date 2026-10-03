"""Regression tests for debug checks in the page-size-one KV allocator."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_allocator(*, need_sort: bool) -> TokenToKVPoolAllocator:
    with envs.SGLANG_DEBUG_MEMORY_POOL.override(True):
        return TokenToKVPoolAllocator(
            size=8,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=need_sort,
        )


class TestTokenAllocatorDebugMode(CustomTestCase):
    def test_rejects_double_free_in_each_free_list(self):
        """Debug mode must reject a slot already returned to either free list."""
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = _make_allocator(need_sort=need_sort)
                released = allocator.alloc(8)[:4]

                allocator.free(released)
                with self.assertRaises(AssertionError):
                    allocator.free(released)

    def test_rejects_double_free_when_group_is_flushed(self):
        """Deferred frees must be checked together when their group is flushed."""
        allocator = _make_allocator(need_sort=False)
        released = allocator.alloc(8)[:4]

        allocator.free_group_begin()
        allocator.free(released)
        allocator.free(released)
        with self.assertRaises(AssertionError):
            allocator.free_group_end()


if __name__ == "__main__":
    unittest.main()
