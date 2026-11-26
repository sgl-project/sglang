import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3, suite="extra-a-test-npu")

from sglang.srt.mem_cache.mamba_slot_fused import npu_clear_slots_fast
from sglang.srt.utils import is_npu


class TestNPUMambaClearSlots(CustomTestCase):
    def test_empty_indices_noop(self):
        buf = torch.ones((2, 4, 8), dtype=torch.bfloat16)
        idx = torch.empty((0,), dtype=torch.int64)
        npu_clear_slots_fast(buf, idx)
        self.assertEqual(buf.min().item(), 1.0)

    @unittest.skipUnless(is_npu(), "Requires Ascend NPU hardware")
    def test_clear_slots_exact_bf16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float32):
            buf = torch.ones((8, 16, 48, 64), device="npu", dtype=dtype)
            indices = torch.tensor([0, 3, 7, 15], device="npu", dtype=torch.int64)
            npu_clear_slots_fast(buf, indices)
            torch.npu.synchronize()

            self.assertEqual(buf[:, indices].abs().max().item(), 0.0)
            untouched = [i for i in range(16) if i not in (0, 3, 7, 15)]
            self.assertEqual(buf[:, untouched].min().item(), 1.0)
            self.assertEqual(buf[:, untouched].max().item(), 1.0)

    @unittest.skipUnless(is_npu(), "Requires Ascend NPU hardware")
    def test_clear_slots_int64_large_pool_no_overflow(self):
        # 48 * 64 * 48 * 128 * 128 = 2.41B elements > INT32_MAX (2.14B).
        # Verifies int64 offset arithmetic avoids int32 overflow in Triton.
        temporal = torch.ones(
            (48, 64, 48, 128, 128), device="npu", dtype=torch.bfloat16
        )
        indices = torch.tensor([0, 31, 63], device="npu", dtype=torch.int64)
        npu_clear_slots_fast(temporal, indices)
        torch.npu.synchronize()

        self.assertEqual(temporal[:, indices].abs().max().item(), 0.0)
        self.assertEqual(temporal[:, 1].min().item(), 1.0)
        self.assertEqual(temporal[:, 62].min().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
