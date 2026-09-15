import unittest

import torch

from sglang.srt.utils.common import device_memory_reserved, empty_device_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoAllocatorHooks:
    pass


class _WithAllocatorHooks:
    def __init__(self, reserved: int) -> None:
        self._reserved = reserved
        self.empty_cache_calls = 0

    def memory_reserved(self) -> int:
        return self._reserved

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class TestDeviceAllocatorProbes(CustomTestCase):
    """Guards the device-module probes the scheduler logs a reclaim from.

    A device module without the hooks must report 0 / False rather than raise,
    and must not let the caller claim a reclaim that never happened.
    """

    def test_memory_reserved_returns_zero_without_the_hook(self):
        self.assertEqual(device_memory_reserved(_NoAllocatorHooks()), 0)

    def test_memory_reserved_forwards_to_the_hook(self):
        self.assertEqual(device_memory_reserved(_WithAllocatorHooks(4096)), 4096)

    def test_empty_cache_reports_whether_it_ran(self):
        absent = _NoAllocatorHooks()
        self.assertFalse(empty_device_cache(absent))

        present = _WithAllocatorHooks(0)
        self.assertTrue(empty_device_cache(present))
        self.assertEqual(present.empty_cache_calls, 1)

    def test_cpu_device_module_has_no_memory_reserved(self):
        # torch.cpu exposes no allocator introspection, so the probe has to
        # tolerate its absence rather than assume a cuda-shaped module.
        self.assertEqual(device_memory_reserved(torch.get_device_module("cpu")), 0)


if __name__ == "__main__":
    unittest.main()
