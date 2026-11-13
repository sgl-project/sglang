"""CPU regression for llguidance state across decode-side PREBUILT reentry."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from llguidance import LLTokenizer, grammar_from

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constrained.llguidance_backend import GuidanceGrammar  # noqa: E402
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (  # noqa: E402
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPrebuiltLLGuidance(unittest.TestCase):
    def test_reentry_does_not_accept_transferred_token_twice(self):
        self._check_reentry(decode_before_reentry=False)

    def test_reentry_after_decode_does_not_accept_last_output_twice(self):
        self._check_reentry(decode_before_reentry=True)

    def _check_reentry(self, decode_before_reentry):
        grammar = GuidanceGrammar(LLTokenizer("byte"), grammar_from("regex", "abc"))
        request = SimpleNamespace(
            output_ids=[ord("a")],
            swa_branching_seqlen=None,
            grammar=grammar,
            rid="retracted-grammar",
            finished=lambda: False,
            request_finish=Mock(),
        )
        batch = SimpleNamespace(
            reqs=[request],
            tree_cache=None,
            device="cpu",
            req_pool_indices=torch.tensor([0]),
            spec_algorithm=Mock(),
        )
        batch.spec_algorithm.build_disagg_draft_input.return_value = None
        future_map = Mock()

        # Exercise the real PREBUILT path; only KV bookkeeping and the relay
        # are substituted because this test has no model or transferred KV.
        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin.checkpoint_kv_cache"
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, future_map)
            request.request_finish.assert_not_called()
            if decode_before_reentry:
                # Normal decode accepts each sampled token and appends it to
                # the request before memory-pressure retraction preserves it.
                grammar.accept_token(ord("b"))
                request.output_ids.append(ord("b"))
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, future_map)

        request.request_finish.assert_not_called()
        self.assertEqual(grammar.current_token, request.output_ids[-1])
        # Reentry must preserve the matcher position, including decode progress.
        for token in "c" if decode_before_reentry else "bc":
            grammar.accept_token(ord(token))
        self.assertFalse(grammar.ll_matcher.is_error())


if __name__ == "__main__":
    unittest.main()
