"""Single-GPU real PD capture, source parity, handoff faults and cancellation."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")


class TestPDCaptureRuntime(PDCaptureRuntimeBase):
    def test_eager_pd_handoff_and_failure_exclusion(self):
        self.exercise(replay=False)

    def test_graph_overlap_pd_handoff_and_failure_exclusion(self):
        self.exercise(replay=True)


if __name__ == "__main__":
    unittest.main()
