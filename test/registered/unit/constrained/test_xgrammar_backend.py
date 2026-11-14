"""Unit tests for the XGrammar constrained-decoding backend."""

import unittest
from unittest.mock import MagicMock, patch

from xgrammar import TokenizerInfo

from sglang.srt.constrained import xgrammar_backend
from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _EmptyTokenizer:
    def init_xgrammar(self):
        return TokenizerInfo([]), None


class TestXGrammarGrammarBackend(unittest.TestCase):
    def test_dispatch_ebnf_uses_compile_grammar(self):
        # Bypass tokenizer initialization: this case covers dispatch order only.
        backend = XGrammarGrammarBackend.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()
        source = 'root ::= "yes" | "no"'
        compiled = MagicMock()
        grammar = MagicMock(spec=BaseGrammarObject)
        backend.grammar_compiler.compile_grammar.return_value = compiled

        with patch.object(
            backend, "_from_context", return_value=grammar
        ) as from_context:
            result = backend.dispatch_ebnf(source)

        self.assertIs(result, grammar)
        backend.grammar_compiler.compile_grammar.assert_called_once_with(source)
        backend.grammar_compiler.compile_lark.assert_not_called()
        args = from_context.call_args.args
        self.assertEqual(args[:2], (compiled, source))
        self.assertEqual(args[2].dispatch_type, "ebnf")

    def test_dispatch_ebnf_compiles_valid_grammar_sources(self):
        backend = self._make_real_backend()
        cases = (
            ('root ::= "yes" | "no"', "ebnf"),
            ('start: "yes" | "no"', "lark"),
        )
        for source, dispatch_type in cases:
            with self.subTest(dispatch_type=dispatch_type):
                grammar = backend.dispatch_ebnf(source)
                self.assertNotIsInstance(grammar, InvalidGrammarObject)
                self.assertEqual(grammar.grammar_stats.dispatch_type, dispatch_type)

    def test_dispatch_invalid_grammar_reports_both_errors(self):
        backend = self._make_real_backend()

        result = backend.dispatch_ebnf("not valid")

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("Failed to compile grammar as EBNF (", result.error_message)
        self.assertIn("EBNF parser error", result.error_message)
        self.assertIn("Lark error", result.error_message)
        recovered = backend.dispatch_ebnf('start: "yes" | "no"')
        self.assertNotIsInstance(recovered, InvalidGrammarObject)

    def test_dispatch_ebnf_without_lark_support_keeps_ebnf_error(self):
        """xgrammar without compile_lark must report the EBNF error, not AttributeError."""
        backend = self._make_real_backend()

        with patch.object(xgrammar_backend, "_XGRAMMAR_SUPPORTS_LARK", False):
            result = backend.dispatch_ebnf('start: "yes" | "no"')

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("EBNF lexer error", result.error_message)
        self.assertNotIn("Lark", result.error_message)

    def _make_real_backend(self):
        backend = XGrammarGrammarBackend(_EmptyTokenizer(), vocab_size=0)
        self.addCleanup(backend.executor.shutdown, wait=True)
        return backend


if __name__ == "__main__":
    unittest.main()
