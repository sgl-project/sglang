"""PD target-KV draft verification and publication across two tensor ranks."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=420, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestPDTargetKVParallelCapture(PDCaptureRuntimeBase):
    def test_tp2_eager(self):
        self.exercise(replay=False, prefill_tp=2, decode_tp=2, draft_kind="target_kv")

    def test_tp2_graph(self):
        self.exercise(
            replay=True,
            prefill_tp=2,
            decode_tp=2,
            draft_kind="target_kv",
            prefill_draft=True,
        )


if __name__ == "__main__":
    unittest.main()
