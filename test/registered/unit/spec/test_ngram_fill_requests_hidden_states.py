"""Regression test for NgramVerifyInput._fill_requests tensor truth-value check.

When `return_hidden_states=True`, `logits_output.hidden_states` is a multi-row
tensor. `_fill_requests` previously gated the slice with `if logits_output.hidden_states:`,
a bare truthiness check that raises
`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.
The fix uses the `is not None` idiom already used throughout the speculative
dir (eagle_info.py, spec_utils.py). See issue #26131.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _make_verify_input(accept_indices, predict):
    # Bypass __init__: _fill_requests only reads accept_indices and predict.
    obj = object.__new__(NgramVerifyInput)
    obj.accept_indices = accept_indices
    obj.predict = predict
    return obj


def _make_batch():
    req = MagicMock()
    req.output_ids = []
    req.require_reasoning = False
    req.grammar = None
    req.finished.return_value = False
    req.spec_verify_ct = 0
    req.spec_num_correct_drafts = 0

    batch = MagicMock()
    batch.reqs = [req]
    batch.model_config.think_end_id = None
    return batch


class TestNgramFillRequestsHiddenStates(CustomTestCase):
    def test_multi_row_hidden_states_does_not_raise(self):
        # accepts indices 0 and 1; -1 terminates the row.
        verify_input = _make_verify_input(
            accept_indices=torch.tensor([[0, 1, -1]], dtype=torch.int64),
            predict=torch.tensor([100, 101, 102], dtype=torch.int64),
        )
        logits_output = LogitsProcessorOutput(
            next_token_logits=torch.randn(3, 8),
            hidden_states=torch.randn(3, 4),  # >1 element -> bare bool() would raise
        )

        verify_input._fill_requests(_make_batch(), logits_output)

        # hidden_states sliced to the two accepted indices, same as next_token_logits.
        self.assertEqual(logits_output.hidden_states.shape[0], 2)
        self.assertEqual(logits_output.next_token_logits.shape[0], 2)

    def test_none_hidden_states_is_skipped(self):
        verify_input = _make_verify_input(
            accept_indices=torch.tensor([[0, 1, -1]], dtype=torch.int64),
            predict=torch.tensor([100, 101, 102], dtype=torch.int64),
        )
        logits_output = LogitsProcessorOutput(
            next_token_logits=torch.randn(3, 8),
            hidden_states=None,
        )

        verify_input._fill_requests(_make_batch(), logits_output)

        self.assertIsNone(logits_output.hidden_states)


if __name__ == "__main__":
    unittest.main()
