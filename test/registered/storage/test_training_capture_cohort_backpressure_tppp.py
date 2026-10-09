"""The final PP-stage publisher controls TP2/PP2 admission during a Store stall."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_backpressure_runtime import (
    CohortBackpressureRuntimeBase,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
class TestCohortBackpressureCombined(CohortBackpressureRuntimeBase):
    tp_size = 2
    pp_size = 2

    def test_publication_stall(self):
        self.exercise_backpressure()


if __name__ == "__main__":
    unittest.main()
