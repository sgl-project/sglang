"""Real pipeline P/D target-KV draft generation and full Store snapshots."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestPDPipelineTargetKV(PDCaptureRuntimeBase):
    def pipeline(self, *, replay, prefill_draft=False, decode_pp=2):
        self.exercise(
            replay=replay,
            prefill_pp=2,
            decode_pp=decode_pp,
            draft_kind="target_kv",
            prefill_draft=prefill_draft,
        )

    def test_decode_draft_eager(self):
        self.pipeline(replay=False)

    def test_decode_draft_graph(self):
        self.pipeline(replay=True)

    def test_prefill_decode_draft_eager(self):
        self.pipeline(replay=False, prefill_draft=True)

    def test_prefill_decode_draft_graph(self):
        self.pipeline(replay=True, prefill_draft=True)

    def test_reduced_pipeline_eager(self):
        self.pipeline(replay=False, prefill_draft=True, decode_pp=1)

    def test_reduced_pipeline_graph(self):
        self.pipeline(replay=True, prefill_draft=True, decode_pp=1)


if __name__ == "__main__":
    unittest.main()
