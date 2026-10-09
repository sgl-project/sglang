"""Real combined TP/PP P/D capture, including static target-KV drafts."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=1000, stage="extra-b", runner_config="4-gpu-h100")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
class TestCombinedParallelPDCapture(PDCaptureRuntimeBase):
    def combined(self, *, replay, draft=False, prefill_draft=False):
        self.exercise(
            replay=replay,
            prefill_tp=2,
            decode_tp=2,
            prefill_pp=2,
            decode_pp=2,
            draft_kind="target_kv" if draft else None,
            prefill_draft=prefill_draft,
        )

    def test_ar_eager(self):
        self.combined(replay=False)

    def test_ar_graph(self):
        self.combined(replay=True)

    def test_decode_draft_eager(self):
        self.combined(replay=False, draft=True)

    def test_decode_draft_graph(self):
        self.combined(replay=True, draft=True)

    def test_prefill_decode_draft_eager(self):
        self.combined(replay=False, draft=True, prefill_draft=True)

    def test_prefill_decode_draft_graph(self):
        self.combined(replay=True, draft=True, prefill_draft=True)


if __name__ == "__main__":
    unittest.main()
