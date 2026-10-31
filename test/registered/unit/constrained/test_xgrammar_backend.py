"""
Unit tests for sglang.srt.constrained.xgrammar_backend.

Test Coverage:
- XGrammarGrammar.rollback: drops exactly the last k accepted tokens,
  k=0 is a no-op, truncation happens in place (regression for #31711 —
  the old slice-copy was O(output_len) per call on the EAGLE spec-decode
  hot path and `[:-0]` cleared the whole token history).
- Real matcher coverage: empty/full rollback, accept/rollback/accept cycles,
  and oversized rollback rejection without changing either history.

Usage:
    python -m pytest test_xgrammar_backend.py -v
"""

import unittest

import xgrammar as xgr

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(2.0, "base-a-test-cpu")


def _make_grammar(tokens):
    """Build a real digit grammar without a downloaded tokenizer or model."""
    tokenizer_info = xgr.TokenizerInfo(
        [str(i) for i in range(10)] + ["<eos>"], stop_token_ids=[10]
    )
    ctx = xgr.GrammarCompiler(tokenizer_info, max_threads=1).compile_grammar(
        "root ::= [0-9]*"
    )
    grammar = XGrammarGrammar(
        matcher=xgr.GrammarMatcher(ctx),
        vocab_size=11,
        ctx=ctx,
        override_stop_tokens=None,
        key_string="test",
    )
    for token in tokens:
        grammar.accept_token(token)
    return grammar


class TestXGrammarGrammarRollback(CustomTestCase):
    """Test XGrammarGrammar.rollback token-history bookkeeping (#31711)."""

    def test_rollback_drops_last_k_tokens(self):
        grammar = _make_grammar([1, 2, 3, 4, 5])
        grammar.rollback(2)
        self.assertEqual(grammar.accepted_tokens, [1, 2, 3])

    def test_rollback_zero_on_empty_history(self):
        grammar = _make_grammar([])
        original = grammar.accepted_tokens
        grammar.rollback(0)
        self.assertIs(grammar.accepted_tokens, original)
        self.assertEqual(original, [])

    def test_rollback_zero_is_noop(self):
        """rollback(0) must keep the history: `[:-0]` used to clear it."""
        grammar = _make_grammar([1, 2, 3])
        grammar.rollback(0)
        self.assertEqual(grammar.accepted_tokens, [1, 2, 3])

    def test_rollback_truncates_in_place(self):
        """The spec-decode tree traversal calls rollback(1) per draft-tree
        node; the history must be truncated in place, not slice-copied."""
        grammar = _make_grammar([1, 2, 3, 4])
        tokens_before = grammar.accepted_tokens
        grammar.rollback(1)
        self.assertIs(grammar.accepted_tokens, tokens_before)
        self.assertEqual(tokens_before, [1, 2, 3])

    def test_rollback_all_tokens_keeps_list_identity(self):
        grammar = _make_grammar([1, 2, 3])
        original = grammar.accepted_tokens
        grammar.rollback(3)
        self.assertIs(grammar.accepted_tokens, original)
        self.assertEqual(original, [])

    def test_accept_rollback_accept_cycle(self):
        grammar = _make_grammar([1, 2, 3, 4, 5])
        grammar.rollback(2)
        grammar.accept_token(6)
        grammar.accept_token(7)
        self.assertEqual(grammar.accepted_tokens, [1, 2, 3, 6, 7])
        grammar.rollback(5)
        self.assertEqual(grammar.accepted_tokens, [])
        with self.assertRaisesRegex(RuntimeError, "Intended to rollback"):
            grammar.rollback(1)

    def test_oversized_rollback_preserves_history(self):
        grammar = _make_grammar([1, 2, 3])
        original = grammar.accepted_tokens
        with self.assertRaisesRegex(RuntimeError, "Intended to rollback"):
            grammar.rollback(4)
        self.assertIs(grammar.accepted_tokens, original)
        self.assertEqual(original, [1, 2, 3])
        # A rejected rollback must also leave the native matcher usable.
        grammar.rollback(3)
        self.assertEqual(grammar.accepted_tokens, [])


if __name__ == "__main__":
    unittest.main()
