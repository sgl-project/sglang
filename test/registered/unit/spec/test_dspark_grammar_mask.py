"""Unit coverage for DSPARK's grammar-mask geometry."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    apply_grammar_vocab_mask,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDSparkGrammarMask(unittest.TestCase):
    def test_linear_tree_mask_rows_match_verify_logits(self):
        grammar = MagicMock()
        reqs = [SimpleNamespace(grammar=grammar), SimpleNamespace(grammar=None)]
        draft_input = SimpleNamespace(grammar=None)
        verify_ids = torch.tensor([[11, 12, 13], [21, 22, 23]])
        logits = torch.zeros((6, 64))
        vocab_mask = torch.ones((6, 2), dtype=torch.int32)

        def generate_mask(
            actual_reqs,
            actual_draft_input,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            vocab_size,
        ):
            self.assertIs(actual_reqs, reqs)
            self.assertIs(actual_draft_input, draft_input)
            self.assertEqual(vocab_size, 64)
            torch.testing.assert_close(
                retrieve_next_token,
                torch.tensor([[1, 2, -1], [1, 2, -1]]),
            )
            torch.testing.assert_close(
                retrieve_next_sibling,
                torch.full((2, 3), -1, dtype=torch.int64),
            )
            torch.testing.assert_close(draft_tokens, verify_ids)
            actual_draft_input.grammar = grammar
            return vocab_mask

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.generate_token_bitmask",
            side_effect=generate_mask,
        ):
            apply_grammar_vocab_mask(
                reqs=reqs,
                draft_input=draft_input,
                verify_ids_2d=verify_ids,
                next_token_logits=logits,
                vocab_size=64,
            )

        grammar.apply_vocab_mask.assert_called_once()
        call = grammar.apply_vocab_mask.call_args
        self.assertIs(call.kwargs["logits"], logits)
        self.assertIs(call.kwargs["vocab_mask"], vocab_mask)


if __name__ == "__main__":
    unittest.main()
