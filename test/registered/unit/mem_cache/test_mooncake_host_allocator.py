"""Unit tests for MooncakeHostTensorAllocator allocation failures."""

import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeHostTensorAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeHostTensorAllocator(CustomTestCase):
    def _make_allocator(self, alloc_return):
        # Bypass __init__: it constructs the real MooncakeHostMemAllocator,
        # which requires the mooncake package and a mappable host segment.
        allocator = MooncakeHostTensorAllocator.__new__(MooncakeHostTensorAllocator)
        allocator.allocator = mock.Mock(alloc=mock.Mock(return_value=alloc_return))
        allocator.ptr = None
        return allocator

    def test_null_pointer_from_allocator_raises(self):
        for alloc_return in (0, None):
            with self.subTest(alloc_return=alloc_return):
                allocator = self._make_allocator(alloc_return)

                with self.assertRaises(RuntimeError) as ctx:
                    allocator.allocate((1024, 1024), dtype=torch.uint8)

                message = str(ctx.exception)
                self.assertIn("1048576 bytes", message)
                self.assertIn("0.001 GB", message)
                self.assertIn("mooncake host memory", message)


if __name__ == "__main__":
    unittest.main()
