"""Actual P-only and D-only weight replacement with independent producer lifetimes."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_weight_runtime import PDCaptureWeightRuntimeBase

register_cuda_ci(est_time=1200, stage="base-b", runner_config="1-gpu-small")


class TestPDCaptureWeights(PDCaptureWeightRuntimeBase):
    def test_prefill_eager(self):
        self.exercise_pd_weights(role="prefill", replay=False)

    def test_prefill_graph(self):
        self.exercise_pd_weights(role="prefill", replay=True)

    def test_decode_eager(self):
        self.exercise_pd_weights(role="decode", replay=False)

    def test_decode_graph(self):
        self.exercise_pd_weights(role="decode", replay=True)


if __name__ == "__main__":
    unittest.main()
