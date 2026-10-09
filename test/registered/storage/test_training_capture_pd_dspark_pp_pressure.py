"""Two-stage P/D target-KV draft pressure and owner-local Store snapshots."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_pressure import PDCapturePressureBase

register_cuda_ci(est_time=400, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestPDPipelineDSparkPressure(PDCapturePressureBase):
    def test_eager(self):
        self.exercise_pressure(replay=False, pp_size=2)

    def test_graph(self):
        self.exercise_pressure(replay=True, pp_size=2)


if __name__ == "__main__":
    unittest.main()
