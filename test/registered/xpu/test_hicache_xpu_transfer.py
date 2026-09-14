import unittest

import torch

from sglang.srt.hardware_backend.xpu.hicache import backup_to_host, load_to_device
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")

XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
class TestHiCacheXPUTransfer(unittest.TestCase):
    def _round_trip(self, shape, dtype):
        host_source = torch.randn(shape, dtype=dtype, pin_memory=True)
        device = torch.zeros(shape, dtype=dtype, device="xpu")
        host_result = torch.zeros(shape, dtype=dtype, pin_memory=True)
        host_indices = torch.tensor([5, 1, 3], dtype=torch.int64)
        device_indices = torch.tensor([0, 4, 2], dtype=torch.int64, device="xpu")

        load_to_device(
            host_tensors=[host_source],
            device_tensors=[device],
            host_indices=host_indices,
            device_indices=device_indices,
        )
        backup_to_host(
            device_tensors=[device],
            host_tensors=[host_result],
            device_indices=device_indices,
            host_indices=host_indices,
        )
        torch.xpu.synchronize()
        torch.testing.assert_close(
            host_result.index_select(0, host_indices),
            host_source.index_select(0, host_indices),
        )

    def test_mha_shaped_round_trip(self):
        self._round_trip((8, 4, 16), torch.float16)

    def test_mla_shaped_round_trip(self):
        self._round_trip((8, 576), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
