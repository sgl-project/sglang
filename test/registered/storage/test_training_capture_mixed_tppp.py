"""Native mixed collection with simultaneous TP2 and PP2 serving."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_mixed_runtime import MixedCaptureRuntimeBase

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="4-gpu-h100")


class TestMixedTPPP(MixedCaptureRuntimeBase):
    tp_size = 2
    pp_size = 2

    def test_synchronous_eager(self):
        self.exercise_mixed(overlap=False)

    def test_synchronous_decode_graph(self):
        self.exercise_mixed(overlap=False, decode="full")

    def test_breakable_prefill(self):
        self.exercise_mixed(overlap=False, prefill="breakable", decode="full")

    def test_full_prefill(self):
        self.exercise_mixed(overlap=False, prefill="full", decode="full")

    def test_piecewise_prefill(self):
        self.exercise_mixed(overlap=False, prefill="tc_piecewise", decode="full")


if __name__ == "__main__":
    unittest.main()
