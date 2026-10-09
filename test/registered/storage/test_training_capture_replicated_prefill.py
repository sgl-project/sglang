"""Real Qwen2.5 TP4 prefill graphs with two replicated logical KV heads."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_prefill_runtime import PrefillCaptureRuntimeBase
from sglang.test.training_capture_replicated_runtime import ReplicatedKVAssertions

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="4-gpu-h100")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
class TestReplicatedPrefillCapture(ReplicatedKVAssertions, PrefillCaptureRuntimeBase):
    tp_size = 4
    compare_capture_off = True

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
