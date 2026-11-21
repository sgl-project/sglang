"""Pointwise MIS readouts must skip structural tokens for empty candidates."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LMHeadHarness:
    def _get_logits(self, hidden_states, weight, metadata):
        return torch.nn.functional.linear(hidden_states, weight)


class TestMISReadouts(CustomTestCase):
    def _logprobs(self, lengths, delimiters, labels, *, include_lengths=True):
        hidden = torch.arange(sum(lengths) * 4, dtype=torch.float32).reshape(-1, 4)
        hidden = torch.sin(hidden)
        metadata = SimpleNamespace(
            extend_seq_lens_cpu=lengths if include_lengths else None,
            token_ids_logprobs=labels,
            extend_return_top_logprob=False,
            extend_token_ids_logprob=True,
            mm_input_embeds=None,
        )
        result = LogitsProcessor.compute_logprobs_for_multi_item_scoring(
            _LMHeadHarness(),
            input_ids=torch.arange(sum(lengths)),
            hidden_states=hidden,
            lm_head=torch.eye(4),
            logits_metadata=metadata,
            multi_item_delimiter_indices=[torch.tensor(d) for d in delimiters],
        )
        return result.input_token_ids_logprobs_val, torch.log_softmax(hidden, dim=-1)

    def test_empty_items_use_their_own_query_prediction(self):
        # Leading, consecutive and trailing empties around two nonempty items.
        first_delimiters = [2, 3, 6, 7, 8, 10, 11]
        first_readouts = [1, 1, 5, 1, 1, 9, 1]
        for include_lengths in (False, True):
            with self.subTest(include_lengths=include_lengths):
                actual, reference = self._logprobs(
                    [12],
                    [first_delimiters],
                    [[3, 1]],
                    include_lengths=include_lengths,
                )
                torch.testing.assert_close(
                    actual[0], reference[first_readouts][:, [3, 1]]
                )

        # A second packed request has a different prefix and only empty items.
        actual, reference = self._logprobs(
            [12, 6], [first_delimiters, [3, 4, 5]], [[3, 1], [2, 0, 3]]
        )
        self.assertEqual(len(actual), 2)
        torch.testing.assert_close(actual[0], reference[first_readouts][:, [3, 1]])
        torch.testing.assert_close(actual[1], reference[[14, 14, 14]][:, [2, 0, 3]])

    def test_empty_query_with_nonempty_items_is_unchanged(self):
        actual, reference = self._logprobs([6], [[0, 3, 5]], [[1, 2]])
        # The query-boundary row is discarded by the score result processor.
        torch.testing.assert_close(actual[0][1:], reference[[2, 4]][:, [1, 2]])


if __name__ == "__main__":
    unittest.main()
