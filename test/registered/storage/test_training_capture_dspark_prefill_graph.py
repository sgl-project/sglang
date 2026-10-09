"""KV-input DSpark prefill/verify graphs and exact post-exit Store readback."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_prefill_runtime import PrefillCaptureRuntimeBase

register_cuda_ci(est_time=900, stage="base-b", runner_config="1-gpu-small")


class TestTargetKVPrefillGraphCapture(PrefillCaptureRuntimeBase):
    def test_full_synchronous(self):
        self.exercise_prefill("full", overlap=False, target_kv=True)

    def test_full_overlap(self):
        self.exercise_prefill("full", target_kv=True)

    def test_breakable_overlap(self):
        self.exercise_prefill("breakable", target_kv=True)

    def test_torch_compile_piecewise_overlap(self):
        self.exercise_prefill("tc_piecewise", target_kv=True)


if __name__ == "__main__":
    unittest.main()
