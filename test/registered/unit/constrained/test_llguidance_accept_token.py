"""A rejected GuidanceGrammar.accept_token must leave the matcher usable:
traverse_tree's prune-on-reject keeps verifying siblings and rolling back on it.
"""

import unittest

import torch
from llguidance import LLTokenizer, grammar_from

from sglang.srt.constrained.llguidance_backend import GuidanceGrammar
from sglang.srt.speculative.spec_utils import traverse_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGuidanceAcceptTokenReject(CustomTestCase):
    def setUp(self):
        self.tokenizer = LLTokenizer("byte")
        self.grammar = GuidanceGrammar(
            llguidance_tokenizer=self.tokenizer,
            serialized_grammar=grammar_from("regex", "ab|ac"),
        )

    def _tok(self, text):
        ids = self.tokenizer.tokenize_str(text)
        self.assertEqual(len(ids), 1)
        return ids[0]

    def test_rejected_token_leaves_matcher_at_current_state(self):
        grammar = self.grammar
        grammar.accept_token(self._tok("a"))
        with self.assertRaises(ValueError):
            grammar.accept_token(self._tok("x"))
        self.assertFalse(grammar.ll_matcher.is_error())
        # Still parked after "a": both continuations stay reachable.
        grammar.accept_token(self._tok("b"))
        grammar.rollback(1)
        grammar.accept_token(self._tok("c"))

    def test_traverse_tree_survives_reject_below_root(self):
        # Verify tree: root -> "a" -> {"x", "b"}. Keep every mask row all-allow so
        # the matcher, not the parent row, is what rejects "x" (a mask/matcher
        # desync). The parent "a" then rolls back after its children; before the
        # fix that rollback raised out of traverse_tree, and the valid sibling
        # "b" was pruned too.
        grammar = self.grammar
        grammar.fill_vocab_mask = lambda vocab_mask, idx: None
        a, x, b = self._tok("a"), self._tok("x"), self._tok("b")
        retrieve_next_token = torch.tensor([1, 2, -1, -1], dtype=torch.int32)
        retrieve_next_sibling = torch.tensor([-1, -1, 3, -1], dtype=torch.int32)
        draft_tokens = torch.tensor([0, a, x, b], dtype=torch.int64)
        bitmask = grammar.allocate_vocab_mask(self.tokenizer.vocab_size, 4, "cpu")
        bitmask.fill_(-1)

        traverse_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar,
            bitmask,
        )

        self.assertFalse(grammar.ll_matcher.is_error())
        # "x" is masked off its parent row; its valid sibling "b" is not.
        self.assertEqual(int(bitmask[1, x // 32] & (1 << (x % 32))), 0)
        self.assertNotEqual(int(bitmask[1, b // 32] & (1 << (b % 32))), 0)


if __name__ == "__main__":
    unittest.main()
