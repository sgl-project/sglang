"""Single-GPU native mixed prefill/decode collection through Mooncake."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_mixed_runtime import MixedCaptureRuntimeBase

register_cuda_ci(est_time=1000, stage="extra-a", runner_config="1-gpu-small")


class TestMixedCapture(MixedCaptureRuntimeBase):
    def test_synchronous_eager(self):
        self.exercise_mixed(overlap=False)

    def test_overlap_eager(self):
        self.exercise_mixed(overlap=True)

    def test_synchronous_decode_graph(self):
        self.exercise_mixed(overlap=False, decode="full")

    def test_overlap_decode_graph(self):
        self.exercise_mixed(overlap=True, decode="full")

    def test_breakable_prefill(self):
        self.exercise_mixed(overlap=True, prefill="breakable", decode="full")

    def test_full_prefill(self):
        self.exercise_mixed(overlap=True, prefill="full", decode="full")

    def test_piecewise_prefill(self):
        self.exercise_mixed(overlap=True, prefill="tc_piecewise", decode="full")


if __name__ == "__main__":
    unittest.main()
