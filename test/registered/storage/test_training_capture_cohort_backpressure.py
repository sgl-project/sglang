"""Real Store publication stalls must propagate through cohort admission."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_backpressure_runtime import (
    CohortBackpressureRuntimeBase,
)

register_cuda_ci(est_time=400, stage="extra-a", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestCohortBackpressureTP(CohortBackpressureRuntimeBase):
    tp_size = 2

    def test_publication_stall(self):
        self.exercise_backpressure()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestCohortBackpressurePP(CohortBackpressureRuntimeBase):
    pp_size = 2

    def test_publication_stall(self):
        self.exercise_backpressure()


if __name__ == "__main__":
    unittest.main()
