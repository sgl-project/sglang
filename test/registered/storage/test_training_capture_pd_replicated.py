"""P/D transfer into a TP4 decoder with canonical replicated-head publication."""

import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase
from sglang.test.training_capture_replicated_runtime import ReplicatedKVAssertions

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="4-gpu-h100")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
class TestReplicatedPDCapture(ReplicatedKVAssertions, PDCaptureRuntimeBase):
    teacher_d2h_batch_tokens = 16
    target_attention_backend = "flashinfer"

    def test_ar_eager(self):
        self.exercise(prefill_tp=4, decode_tp=4, replay=False)

    def test_ar_graph(self):
        self.exercise(prefill_tp=4, decode_tp=4, replay=True)

    def test_ar_split_to_replicated_eager(self):
        self.exercise(prefill_tp=2, decode_tp=4, replay=False)

    def test_ar_split_to_replicated_graph(self):
        self.exercise(prefill_tp=2, decode_tp=4, replay=True)

    def test_dspark_eager(self):
        self.exercise(prefill_tp=4, decode_tp=4, replay=False, draft_kind="target_kv")

    def test_dspark_graph(self):
        self.exercise(prefill_tp=4, decode_tp=4, replay=True, draft_kind="target_kv")


if __name__ == "__main__":
    unittest.main()
