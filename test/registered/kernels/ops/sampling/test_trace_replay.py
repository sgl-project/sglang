import unittest

import torch

from sglang.kernels.ops.sampling import trace_replay
from sglang.srt.utils.device import is_npu
from sglang.test.ci.ci_register import register_cuda_ci, register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=7, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: compile and exercise the irregular masked gather on Triton-Ascend.
register_npu_ci(est_time=15, suite="base-b-test-1-npu-a3")


class TestTraceReplay(CustomTestCase):
    def setUp(self):
        super().setUp()
        if is_npu():
            self.device = "npu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.skipTest("requires Triton on CUDA, ROCm or Ascend")

    def test_decode_replaces_active_rows_and_preserves_sampling(self):
        table = torch.arange(36, dtype=torch.int32).reshape(6, 6)
        slots = torch.tensor([3, 0, 1, 2, 4, 5, 6] * 5)
        lengths = torch.tensor([4, 0, 2, 3, 7, 1, 1] * 5)
        steps = torch.tensor([3, 0, 2, -1, 6, 0, 0] * 5)
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                sampled = torch.full((35, 1), 99, dtype=dtype, device=self.device)
                result = trace_replay(
                    *(t.to(self.device) for t in (table, slots, lengths, steps)),
                    sampled,
                )
                self.assertEqual(result.shape, sampled.shape)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(
                    result.flatten().tolist(), [21, 99, 99, 99, 99, 30, 99] * 5
                )
                self.assertTrue((sampled == 99).all().item())


if __name__ == "__main__":
    unittest.main()
