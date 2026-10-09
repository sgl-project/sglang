"""Two-rank prefill graphs, overlapping KV drafts and sharded Store reads."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_prefill_runtime import PrefillCaptureRuntimeBase

register_cuda_ci(est_time=900, stage="base-b", runner_config="2-gpu")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestTensorPrefillGraphCapture(PrefillCaptureRuntimeBase):
    tp_size = 2

    def test_ar_full(self):
        self.exercise_prefill("full")

    def test_ar_breakable(self):
        self.exercise_prefill("breakable")

    def test_ar_piecewise(self):
        self.exercise_prefill("tc_piecewise")

    def test_dspark_full(self):
        self.exercise_prefill("full", target_kv=True)

    def test_dspark_breakable(self):
        self.exercise_prefill("breakable", target_kv=True)

    def test_dspark_piecewise(self):
        self.exercise_prefill("tc_piecewise", target_kv=True)


if __name__ == "__main__":
    unittest.main()
