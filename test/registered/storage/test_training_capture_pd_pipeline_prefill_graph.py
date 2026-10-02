"""Two-stage P prefill graphs, first-teacher handoff and D publication."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=1200, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestPDPipelinePrefillGraphCapture(PDCaptureRuntimeBase):
    teacher_d2h_batch_tokens = 16
    target_attention_backend = "flashinfer"
    observer_module = "sglang.test.pd_prefill_capture_server"
    validate_prefill_graph = True

    def pipeline(self, backend, *, draft=False):
        self.exercise(
            replay=True,
            prefill_backend=backend,
            prefill_pp=2,
            decode_pp=2,
            draft_kind="target_kv" if draft else None,
            prefill_draft=draft,
            enable_overlap=False,
        )

    def test_ar_full(self):
        self.pipeline("full")

    def test_ar_breakable(self):
        self.pipeline("breakable")

    def test_ar_piecewise(self):
        self.pipeline("tc_piecewise")

    def test_dspark_full(self):
        self.pipeline("full", draft=True)

    def test_dspark_breakable(self):
        self.pipeline("breakable", draft=True)

    def test_dspark_piecewise(self):
        self.pipeline("tc_piecewise", draft=True)


if __name__ == "__main__":
    unittest.main()
