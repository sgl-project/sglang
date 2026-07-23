"""
Unit tests for sglang.srt.constrained.xgrammar_backend.

Test Coverage:
- XGrammarGrammar.rollback: drops exactly the last k accepted tokens,
  k=0 is a no-op, truncation happens in place (regression for #31711 —
  the old slice-copy was O(output_len) per call on the EAGLE spec-decode
  hot path and `[:-0]` cleared the whole token history).

Usage:
    python -m pytest test_xgrammar_backend.py -v
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "base-a-test-cpu")


def _make_grammar(tokens):
    """Build an XGrammarGrammar with a mocked matcher and accept `tokens`."""
    matcher = MagicMock()
    matcher.is_terminated.return_value = False
    matcher.accept_token.return_value = True
    grammar = XGrammarGrammar(
        matcher=matcher,
        vocab_size=32000,
        ctx=MagicMock(),
        override_stop_tokens=None,
        key_string="test",
    )
    for token in tokens:
        grammar.accept_token(token)
    return grammar


class TestXGrammarGrammarRollback(unittest.TestCase):
    """Test XGrammarGrammar.rollback token-history bookkeeping (#31711)."""

    def test_rollback_drops_last_k_tokens(self):
        grammar = _make_grammar([1, 2, 3, 4, 5])
        grammar.rollback(2)
        self.assertEqual(grammar.accepted_tokens, [1, 2, 3])
        grammar.matcher.rollback.assert_called_once_with(2)

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


if __name__ == "__main__":
    unittest.main()
