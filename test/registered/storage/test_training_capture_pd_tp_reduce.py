"""Two prefill ranks hand consistent teacher rows to one decode owner."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestReducedDecodePDCapture(PDCaptureRuntimeBase):
    def test_tp2_to_tp1_eager(self):
        self.exercise(replay=False, prefill_tp=2, decode_tp=1)

    def test_tp2_to_tp1_graph_overlap(self):
        self.exercise(replay=True, prefill_tp=2, decode_tp=1)


if __name__ == "__main__":
    unittest.main()
