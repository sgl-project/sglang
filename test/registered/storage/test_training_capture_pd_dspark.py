"""PD target-KV drafts rebuild their context from received target KV on D."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-small")


class TestPDTargetKVCapture(PDCaptureRuntimeBase):
    def test_decode_draft_eager(self):
        self.exercise(replay=False, draft_kind="target_kv")

    def test_decode_draft_graph(self):
        self.exercise(replay=True, draft_kind="target_kv")

    def test_prefill_and_decode_draft_eager(self):
        self.exercise(replay=False, draft_kind="target_kv", prefill_draft=True)

    def test_prefill_and_decode_draft_graph(self):
        self.exercise(replay=True, draft_kind="target_kv", prefill_draft=True)


if __name__ == "__main__":
    unittest.main()
