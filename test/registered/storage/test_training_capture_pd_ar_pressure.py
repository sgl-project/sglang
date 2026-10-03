"""Ordinary P/D AR capture retirement under actual decode KV exhaustion."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_pressure import PDCapturePressureBase

register_cuda_ci(est_time=500, stage="base-b", runner_config="1-gpu")


class TestPDARPressure(PDCapturePressureBase):
    draft_kind = None
    teacher_d2h_batch_tokens = 16

    def test_eager(self):
        self.exercise_pressure(replay=False)

    def test_graph(self):
        self.exercise_pressure(replay=True)

    def test_overlap_eager(self):
        self.exercise_pressure(replay=False, enable_overlap=True)

    def test_overlap_graph(self):
        self.exercise_pressure(replay=True, enable_overlap=True)


if __name__ == "__main__":
    unittest.main()
