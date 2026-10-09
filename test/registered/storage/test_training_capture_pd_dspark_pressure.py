"""P/D target-KV draft pressure through ordinary and overlap scheduling."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_pressure import PDCapturePressureBase

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu")


class TestPDDSparkPressure(PDCapturePressureBase):
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
