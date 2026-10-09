"""P-side prefill graphs publish complete D-owned AR and DSpark snapshots."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=1500, stage="base-b", runner_config="1-gpu-small")


class TestPDPrefillGraphCapture(PDCaptureRuntimeBase):
    teacher_d2h_batch_tokens = 16
    target_attention_backend = "flashinfer"
    observer_module = "sglang.test.pd_prefill_capture_server"
    validate_prefill_graph = True

    def test_ar_full_synchronous(self):
        self.exercise(replay=True, prefill_backend="full", enable_overlap=False)

    def test_ar_full_overlap(self):
        self.exercise(replay=True, prefill_backend="full")

    def test_ar_breakable(self):
        self.exercise(replay=True, prefill_backend="breakable")

    def test_ar_piecewise(self):
        self.exercise(replay=True, prefill_backend="tc_piecewise")

    def test_dspark_full(self):
        self.exercise(
            replay=True,
            prefill_backend="full",
            draft_kind="target_kv",
            prefill_draft=True,
        )

    def test_dspark_full_synchronous(self):
        self.exercise(
            replay=True,
            prefill_backend="full",
            draft_kind="target_kv",
            prefill_draft=True,
            enable_overlap=False,
        )

    def test_dspark_breakable(self):
        self.exercise(
            replay=True,
            prefill_backend="breakable",
            draft_kind="target_kv",
            prefill_draft=True,
        )

    def test_dspark_piecewise(self):
        self.exercise(
            replay=True,
            prefill_backend="tc_piecewise",
            draft_kind="target_kv",
            prefill_draft=True,
        )

    def test_decode_draft_full(self):
        self.exercise(replay=True, prefill_backend="full", draft_kind="target_kv")


if __name__ == "__main__":
    unittest.main()
