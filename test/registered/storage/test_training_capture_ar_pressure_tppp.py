"""AR capture survives real KV slot reuse with simultaneous TP2 and PP2."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_pressure_runtime import ARPressureCaptureRuntimeBase

register_cuda_ci(est_time=400, stage="extra-b", runner_config="4-gpu-h100")


class TestARPressureTPPP(ARPressureCaptureRuntimeBase):
    tp_size = 2
    pp_size = 2

    def test_synchronous_eager(self):
        self.exercise_pressure(cuda_graph=False, overlap=False)

    def test_synchronous_graph(self):
        self.exercise_pressure(cuda_graph=True, overlap=False)


if __name__ == "__main__":
    unittest.main()
