"""Actual AR KV exhaustion, capture retirement, slot reuse and fresh admission."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_pressure_runtime import ARPressureCaptureRuntimeBase

register_cuda_ci(est_time=400, stage="base-b", runner_config="1-gpu-small")


class TestARPressureCapture(ARPressureCaptureRuntimeBase):
    def test_synchronous_eager(self):
        self.exercise_pressure(cuda_graph=False, overlap=False)

    def test_synchronous_graph(self):
        self.exercise_pressure(cuda_graph=True, overlap=False)

    def test_overlap_eager(self):
        self.exercise_pressure(cuda_graph=False, overlap=True)

    def test_overlap_graph(self):
        self.exercise_pressure(cuda_graph=True, overlap=True)


if __name__ == "__main__":
    unittest.main()
