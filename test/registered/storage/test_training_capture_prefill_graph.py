"""Real prefill graph capture, padding, prefix reuse and post-exit Store reads."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_prefill_runtime import PrefillCaptureRuntimeBase

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-small")


class TestPrefillGraphCapture(PrefillCaptureRuntimeBase):
    def test_full_synchronous(self):
        self.exercise_prefill("full", overlap=False)

    def test_full_overlap(self):
        self.exercise_prefill("full")

    def test_breakable_overlap(self):
        self.exercise_prefill("breakable")

    def test_torch_compile_piecewise_overlap(self):
        self.exercise_prefill("tc_piecewise")


if __name__ == "__main__":
    unittest.main()
