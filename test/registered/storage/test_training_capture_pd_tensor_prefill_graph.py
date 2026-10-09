"""Sharded P prefill graphs and D-owned AR/DSpark Store snapshots."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=1200, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestPDTensorPrefillGraphCapture(PDCaptureRuntimeBase):
    teacher_d2h_batch_tokens = 16
    target_attention_backend = "flashinfer"
    observer_module = "sglang.test.pd_prefill_capture_server"
    validate_prefill_graph = True

    def tensor(self, backend, *, draft=False):
        self.exercise(
            replay=True,
            prefill_backend=backend,
            prefill_tp=2,
            decode_tp=2,
            draft_kind="target_kv" if draft else None,
            prefill_draft=draft,
        )

    def test_ar_full(self):
        self.tensor("full")

    def test_ar_breakable(self):
        self.tensor("breakable")

    def test_ar_piecewise(self):
        self.tensor("tc_piecewise")

    def test_dspark_full(self):
        self.tensor("full", draft=True)

    def test_dspark_breakable(self):
        self.tensor("breakable", draft=True)

    def test_dspark_piecewise(self):
        self.tensor("tc_piecewise", draft=True)


if __name__ == "__main__":
    unittest.main()
