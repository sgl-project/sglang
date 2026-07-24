from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4DraftExtendCudaGraph(TestCase):
    def test_padding_causal_lengths_stay_nonnegative(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.cuda_int32_kwargs = {"dtype": torch.int32, "device": "cpu"}

        seq_lens_casual, req_pool_indices = (
            backend.expand_extend_with_same_length(
                bs=1,
                qo_len=4,
                seq_lens=torch.tensor([1], dtype=torch.int32),
                req_pool_indices=torch.tensor([0], dtype=torch.int32),
            )
        )

        self.assertEqual(seq_lens_casual.tolist(), [1, 1, 1, 1])
        self.assertEqual((seq_lens_casual - 1).tolist(), [0, 0, 0, 0])
        self.assertEqual(req_pool_indices.tolist(), [0, 0, 0, 0])

    def test_read_done_order_follows_backend_capability(self):
        cases = (
            (AttentionBackend, ["event", "replay"]),
            (DeepseekV4AttnBackend, ["replay", "event"]),
        )
        for backend, expected_order in cases:
            with self.subTest(backend=backend.__name__):
                runner = object.__new__(EAGLEDraftExtendCudaGraphRunner)
                runner.draft_extend_attn_backend = backend
                call_order = []
                runner._record_war_fastpath_read_done = mock.Mock(
                    side_effect=lambda: call_order.append("event")
                )
                runner._replay_graph = mock.Mock(
                    side_effect=lambda *_args: call_order.append("replay") or "output"
                )

                output = runner._replay_graph_with_war_read_done(
                    SimpleNamespace(), SimpleNamespace()
                )

                self.assertEqual(output, "output")
                self.assertEqual(call_order, expected_order)
