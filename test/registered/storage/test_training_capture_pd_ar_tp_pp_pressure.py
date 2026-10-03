"""Combined TP2/PP2 ordinary AR P/D restore and complete Store snapshots."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_pressure import PDCapturePressureBase

register_cuda_ci(est_time=500, stage="extra-b", runner_config="4-gpu-h100")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
class TestPDARCombinedPressure(PDCapturePressureBase):
    draft_kind = None
    teacher_d2h_batch_tokens = 16

    def test_eager(self):
        self.exercise_pressure(replay=False, tp_size=2, pp_size=2)

    def test_graph(self):
        self.exercise_pressure(replay=True, tp_size=2, pp_size=2)


if __name__ == "__main__":
    unittest.main()
