"""Pointer tables built on real XPU memory must round-trip (#35047).

Level Zero / SYCL USM hands out addresses with the top bit set, which an int64
table cannot hold. The spoofed-address cases that run anywhere live in
test/registered/unit/memory/test_ptr_table.py.
"""

from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")

import unittest

import torch

from sglang.kernels.ops.memory.ptr_table import make_ptr_table
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU not available")
class TestPtrTableOnDeviceMemory(CustomTestCase):
    def test_real_device_pointers_round_trip(self):
        ptrs = [
            torch.zeros(1024, device="xpu", dtype=torch.bfloat16).data_ptr()
            for _ in range(2)
        ]
        table = make_ptr_table(ptrs, device="xpu")
        self.assertEqual(table.dtype, torch.int64)
        self.assertEqual(table.device.type, "xpu")
        self.assertEqual(table.view(torch.uint64).cpu().tolist(), ptrs)


if __name__ == "__main__":
    unittest.main()
