"""Regression test for spec_utils.traverse_tree calling xgrammar with tensors.

xgrammar 0.2.0 tightened its FFI binding and rejects 0-d tensors where Python
ints are expected. The dfs in traverse_tree recurses with `retrieve_next_token[curr]`
and reads `draft_tokens[curr]`, both of which return 0-d tensors and must be
explicitly cast before being handed to the grammar matcher.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.constrained.utils import is_packed_bitmask_allowed_token
from sglang.srt.speculative.spec_utils import traverse_tree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestIsVocabMaskAllowedToken(unittest.TestCase):
    def test_outlines_dense_bool_mask(self):
        from sglang.srt.constrained.outlines_backend import OutlinesGrammar

        # Outlines uses dense bool masks: False means allowed, True means masked.
        vocab_mask = torch.tensor([True, False, True, False], dtype=torch.bool)

        self.assertTrue(
            OutlinesGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 1, vocab_size=4
            )
        )
        self.assertFalse(
            OutlinesGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 0, vocab_size=4
            )
        )
        self.assertFalse(
            OutlinesGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 4, vocab_size=4
            )
        )
        self.assertFalse(
            OutlinesGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 3, vocab_size=3
            )
        )

    def test_xgrammar_packed_bitmask(self):
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar

        vocab_mask = torch.zeros(2, dtype=torch.int32)
        vocab_mask[0] = 1 << 3
        vocab_mask[1] = 1 << 3

        self.assertTrue(
            XGrammarGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 3, vocab_size=64
            )
        )
        self.assertFalse(
            XGrammarGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 4, vocab_size=64
            )
        )
        self.assertTrue(
            XGrammarGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 35, vocab_size=64
            )
        )
        self.assertFalse(
            XGrammarGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 64, vocab_size=64
            )
        )

    def test_llguidance_packed_bitmask(self):
        from sglang.srt.constrained.llguidance_backend import GuidanceGrammar

        vocab_mask = torch.zeros(2, dtype=torch.int32)
        vocab_mask[0] = 1 << 7
        vocab_mask[1] = 1 << 2

        self.assertTrue(
            GuidanceGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 7, vocab_size=64
            )
        )
        self.assertFalse(
            GuidanceGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 8, vocab_size=64
            )
        )
        self.assertTrue(
            GuidanceGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 34, vocab_size=64
            )
        )
        self.assertFalse(
            GuidanceGrammar.is_vocab_mask_allowed_token(
                None, vocab_mask, 34, vocab_size=34
            )
        )

    def test_reasoner_delegates_to_inner_grammar(self):
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarObject,
        )

        inner = MagicMock()
        inner.is_vocab_mask_allowed_token.return_value = False
        obj = ReasonerGrammarObject(inner, think_end_id=99)
        vocab_mask = torch.zeros(1, dtype=torch.int32)

        self.assertFalse(obj.is_vocab_mask_allowed_token(vocab_mask, 0, vocab_size=1))
        inner.is_vocab_mask_allowed_token.assert_called_once_with(
            vocab_mask, 0, vocab_size=1
        )


class TestTraverseTreePassesIntsToGrammar(unittest.TestCase):
    def _record_grammar(self):
        """A grammar mock that records every call argument and rejects torch tensors."""
        grammar = MagicMock()
        grammar.is_terminated.return_value = False
        accept_calls = []
        fill_calls = []
        allowed_calls = []

        def record_accept(token):
            if isinstance(token, torch.Tensor):
                raise TypeError(f"accept_token got torch.Tensor: {token!r}")
            accept_calls.append(token)

        def record_fill(bitmask, idx):
            if isinstance(idx, torch.Tensor):
                raise TypeError(f"fill_vocab_mask got torch.Tensor idx: {idx!r}")
            fill_calls.append(idx)

        def record_is_vocab_mask_allowed_token(vocab_mask, token_id, vocab_size=None):
            if isinstance(token_id, torch.Tensor):
                raise TypeError(
                    "is_vocab_mask_allowed_token got torch.Tensor token_id: "
                    f"{token_id!r}"
                )
            allowed_calls.append((token_id, vocab_size))
            return is_packed_bitmask_allowed_token(vocab_mask, token_id, vocab_size)

        grammar.accept_token.side_effect = record_accept
        grammar.fill_vocab_mask.side_effect = record_fill
        grammar.is_vocab_mask_allowed_token.side_effect = (
            record_is_vocab_mask_allowed_token
        )
        grammar.rollback.return_value = None
        return grammar, accept_calls, fill_calls, allowed_calls

    def test_branching_tree_passes_ints(self):
        # Binary tree exercises both child recursion and sibling recursion:
        #   0 ─┬─ 1
        #      └─ 2 ─── 3
        retrieve_next_token = torch.tensor([1, -1, 3, -1], dtype=torch.int32)
        retrieve_next_sibling = torch.tensor([-1, 2, -1, -1], dtype=torch.int32)
        draft_tokens = torch.tensor([100, 11, 22, 33], dtype=torch.int64)
        # all bits set: every draft token passes the parent's bitmask check
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)

        grammar, accept_calls, fill_calls, allowed_calls = self._record_grammar()
        traverse_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar,
            bitmask,
        )

        self.assertEqual(set(accept_calls), {11, 22, 33})
        self.assertEqual(set(fill_calls), {0, 1, 2, 3})
        self.assertEqual(set(allowed_calls), {(11, None), (22, None), (33, None)})
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for token, vocab_size in allowed_calls:
            self.assertIsInstance(token, int)
            self.assertIsNone(vocab_size)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)


if __name__ == "__main__":
    unittest.main()
