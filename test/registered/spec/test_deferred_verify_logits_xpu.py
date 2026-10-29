"""Model-owned exact softcap table lifetime across warmup and capture."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, stage="stage-b", runner_config="1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestDeferredVerifyRanks(CustomTestCase):
    def processor(self, cap=30.0):
        with get_context().override_server_args(
            model_path="dummy", device="xpu", tp_size=1
        ):
            return LogitsProcessor(
                SimpleNamespace(vocab_size=262144, final_logit_softcapping=cap),
                defer_xpu_verify_logits=True,
            )

    def test_owner_lifetime_softcap_changes_and_graph_output(self):
        processor = self.processor()
        device = torch.device("xpu", torch.xpu.current_device())
        table = processor._get_verify_softcap_ranks(device)
        self.assertIs(processor._get_verify_softcap_ranks(device), table)
        self.assertNotIn("_verify_softcap_ranks", processor.state_dict())
        other = self.processor()
        self.assertIsNone(other._verify_softcap_ranks)
        self.assertIsNot(other._get_verify_softcap_ranks(device), table)
        raw = torch.empty(8, 262144, device=device, dtype=torch.bfloat16)
        output = LogitsProcessorOutput(
            next_token_logits=None,
            deferred_verify_logits=raw,
            deferred_verify_softcap=30.0,
            deferred_verify_comparison_ranks=table,
        )
        sliced = replace(output, deferred_verify_logits=raw[:3])
        self.assertEqual(sliced.deferred_verify_logits.shape[0], 3)
        self.assertIs(sliced.deferred_verify_comparison_ranks, table)
        processor.final_logit_softcapping = None
        replacement = processor._get_verify_softcap_ranks(device)
        self.assertIsNot(replacement, table)
        self.assertFalse(torch.equal(replacement, table))
        # Existing captured output keeps its original cap/table even if the
        # owner prepares another configuration later.
        self.assertIs(output.deferred_verify_comparison_ranks, table)
        self.assertEqual(output.deferred_verify_softcap, 30.0)

    def test_first_capture_does_not_construct_persistent_table(self):
        processor = self.processor()
        device = torch.device("xpu", torch.xpu.current_device())
        marker = torch.zeros(1, device=device)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            self.assertIsNone(processor._get_verify_softcap_ranks(device))
            marker.add_(1)
        self.assertIsNone(processor._verify_softcap_ranks)
        table = processor._get_verify_softcap_ranks(device)
        self.assertEqual(table.shape, (65536,))
        other_graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(other_graph):
            self.assertIs(processor._get_verify_softcap_ranks(device), table)
            marker.add_(1)
        processor.final_logit_softcapping = 10.0
        third_graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(third_graph):
            self.assertIsNone(processor._get_verify_softcap_ranks(device))
            marker.add_(1)
        self.assertIs(processor._verify_softcap_ranks, table)


if __name__ == "__main__":
    unittest.main()
