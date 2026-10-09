"""Native mixed collection with tensor or pipeline parallel serving."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_mixed_runtime import MixedCaptureRuntimeBase

register_cuda_ci(est_time=1600, stage="extra-a", runner_config="2-gpu-large")


class TestMixedTP(MixedCaptureRuntimeBase):
    tp_size = 2

    def test_synchronous_eager(self):
        self.exercise_mixed(overlap=False)

    def test_overlap_eager(self):
        self.exercise_mixed(overlap=True)

    def test_overlap_decode_graph(self):
        self.exercise_mixed(overlap=True, decode="full")

    def test_breakable_prefill(self):
        self.exercise_mixed(overlap=True, prefill="breakable", decode="full")

    def test_full_prefill(self):
        self.exercise_mixed(overlap=True, prefill="full", decode="full")

    def test_piecewise_prefill(self):
        self.exercise_mixed(overlap=True, prefill="tc_piecewise", decode="full")


class TestMixedPP(MixedCaptureRuntimeBase):
    pp_size = 2

    def test_synchronous_eager(self):
        self.exercise_mixed(overlap=False)

    def test_full_prefill(self):
        self.exercise_mixed(overlap=False, prefill="full", decode="full")

    def test_piecewise_prefill(self):
        self.exercise_mixed(overlap=False, prefill="tc_piecewise", decode="full")


if __name__ == "__main__":
    unittest.main()
