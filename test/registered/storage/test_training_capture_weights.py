"""Version isolation across real disk replacement and a freshly bound producer."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_weight_runtime import CaptureWeightRuntimeBase

register_cuda_ci(est_time=400, stage="base-b", runner_config="1-gpu-small")


class TestCaptureWeightReplacement(CaptureWeightRuntimeBase):
    def test_eager(self):
        self.exercise_weight_update(cuda_graph=False)

    def test_overlap_graph(self):
        self.exercise_weight_update(cuda_graph=True)


if __name__ == "__main__":
    unittest.main()
