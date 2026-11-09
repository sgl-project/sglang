"""CPU tests: the default decode retraction backup only picks host_pool where
the HiCache host pool can allocate its pinned buffers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.test.test_utils import CustomTestCase


class TestDeviceCanBackHostPool(CustomTestCase):
    def test_devices_with_a_host_allocator(self):
        for device in ("npu", "musa", "xpu", "npu:0", torch.device("xpu", 1)):
            with self.subTest(device=device):
                self.assertTrue(kv_cache_builder.device_can_back_host_pool(device))

    def test_cuda_registers_only_on_cuda_or_rocm(self):
        with patch.object(kv_cache_builder, "is_cuda", return_value=True):
            self.assertTrue(kv_cache_builder.device_can_back_host_pool("cuda"))
            self.assertTrue(kv_cache_builder.device_can_back_host_pool("cuda:3"))
        with (
            patch.object(kv_cache_builder, "is_cuda", return_value=False),
            patch.object(kv_cache_builder, "is_hip", return_value=False),
        ):
            self.assertFalse(kv_cache_builder.device_can_back_host_pool("cuda"))

    def test_cpu_and_out_of_tree_devices_fall_back(self):
        for device in ("cpu", "neuron", "my_oot_device:0"):
            with self.subTest(device=device):
                self.assertFalse(kv_cache_builder.device_can_back_host_pool(device))
        self.assertNotIn("neuron", ALLOC_MEMORY_FUNCS, "query must not insert keys")


if __name__ == "__main__":
    unittest.main()
