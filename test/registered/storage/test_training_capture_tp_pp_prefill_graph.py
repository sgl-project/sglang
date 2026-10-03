"""Combined TP/PP prefill graphs with exact all-owner Store snapshot parity."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_prefill_runtime import PrefillCaptureRuntimeBase

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="4-gpu-h100")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
class TestCombinedPrefillGraphCapture(PrefillCaptureRuntimeBase):
    tp_size = 2
    pp_size = 2

    def test_ar_full(self):
        self.exercise_prefill("full", overlap=False)

    def test_ar_breakable(self):
        self.exercise_prefill("breakable", overlap=False)

    def test_ar_piecewise(self):
        self.exercise_prefill("tc_piecewise", overlap=False)

    def test_dspark_full(self):
        self.exercise_prefill("full", overlap=False, target_kv=True)

    def test_dspark_breakable(self):
        self.exercise_prefill("breakable", overlap=False, target_kv=True)

    def test_dspark_piecewise(self):
        self.exercise_prefill("tc_piecewise", overlap=False, target_kv=True)


if __name__ == "__main__":
    unittest.main()
