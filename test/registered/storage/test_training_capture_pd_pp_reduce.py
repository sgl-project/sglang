"""Two prefill pipeline stages hand one complete sample to an unpartitioned D."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestReducedPipelinePDCapture(PDCaptureRuntimeBase):
    def test_pp2_to_pp1_eager(self):
        self.exercise(replay=False, prefill_pp=2)

    def test_pp2_to_pp1_graph(self):
        self.exercise(replay=True, prefill_pp=2)


if __name__ == "__main__":
    unittest.main()
