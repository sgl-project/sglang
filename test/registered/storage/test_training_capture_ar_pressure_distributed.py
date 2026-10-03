"""Real AR KV exhaustion with rank-local capture retirement and recovery."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_pressure_runtime import ARPressureCaptureRuntimeBase

register_cuda_ci(est_time=700, stage="extra-a", runner_config="2-gpu-large")


class TestARPressureTP(ARPressureCaptureRuntimeBase):
    tp_size = 2

    def test_synchronous_eager(self):
        self.exercise_pressure(cuda_graph=False, overlap=False)

    def test_synchronous_graph(self):
        self.exercise_pressure(cuda_graph=True, overlap=False)

    def test_overlap_eager(self):
        self.exercise_pressure(cuda_graph=False, overlap=True)

    def test_overlap_graph(self):
        self.exercise_pressure(cuda_graph=True, overlap=True)


class TestARPressurePP(ARPressureCaptureRuntimeBase):
    pp_size = 2

    def test_synchronous_eager(self):
        self.exercise_pressure(cuda_graph=False, overlap=False)

    def test_synchronous_graph(self):
        self.exercise_pressure(cuda_graph=True, overlap=False)


if __name__ == "__main__":
    unittest.main()
