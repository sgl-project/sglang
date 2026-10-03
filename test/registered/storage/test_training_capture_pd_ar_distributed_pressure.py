"""P/D AR pressure must retire and restore the same request on every rank."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_pressure import PDCapturePressureBase

register_cuda_ci(est_time=900, stage="extra-a", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestPDARTPPressure(PDCapturePressureBase):
    draft_kind = None
    teacher_d2h_batch_tokens = 16

    def test_eager(self):
        self.exercise_pressure(replay=False, tp_size=2)

    def test_graph(self):
        self.exercise_pressure(replay=True, tp_size=2)

    def test_overlap_eager(self):
        self.exercise_pressure(replay=False, enable_overlap=True, tp_size=2)

    def test_overlap_graph(self):
        self.exercise_pressure(replay=True, enable_overlap=True, tp_size=2)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestPDARPPPressure(PDCapturePressureBase):
    draft_kind = None
    teacher_d2h_batch_tokens = 16

    def test_eager(self):
        self.exercise_pressure(replay=False, pp_size=2)

    def test_graph(self):
        self.exercise_pressure(replay=True, pp_size=2)


if __name__ == "__main__":
    unittest.main()
