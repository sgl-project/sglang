"""Single-GPU real PD capture, source parity, handoff faults and cancellation."""

import argparse
import sys
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")


class TestPDCaptureRuntime(PDCaptureRuntimeBase):
    teacher_d2h_batch_tokens = 16

    def test_eager_pd_handoff_and_failure_exclusion(self):
        self.exercise(replay=False)

    def test_graph_overlap_pd_handoff_and_failure_exclusion(self):
        self.exercise(replay=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kv-export-backend", choices=("torch", "hicache"), default="torch"
    )
    parser.add_argument(
        "--teacher-topk-backend", choices=("torch", "flashinfer"), default="torch"
    )
    args, remaining = parser.parse_known_args()
    TestPDCaptureRuntime.teacher_topk_backend = args.teacher_topk_backend
    TestPDCaptureRuntime.kv_export_backend = args.kv_export_backend
    unittest.main(argv=[sys.argv[0], *remaining])
