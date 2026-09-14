"""Unit tests for the XGrammar constrained-decoding backend."""

import unittest
from unittest.mock import MagicMock, patch

from xgrammar import TokenizerInfo

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestXGrammarGrammarBackend(unittest.TestCase):
    def setUp(self):
        # Bypass tokenizer initialization: these tests cover dispatch only.
        self.backend = XGrammarGrammarBackend.__new__(XGrammarGrammarBackend)
        self.backend.grammar_compiler = MagicMock()

    def test_dispatch_ebnf_uses_compile_grammar(self):
        source = 'root ::= "yes" | "no"'
        compiled = MagicMock()
        grammar = MagicMock(spec=BaseGrammarObject)
        self.backend.grammar_compiler.compile_grammar.return_value = compiled

        with patch.object(
            self.backend, "_from_context", return_value=grammar
        ) as from_context:
            result = self.backend.dispatch_ebnf(source)

        self.assertIs(result, grammar)
        self.backend.grammar_compiler.compile_grammar.assert_called_once_with(source)
        self.backend.grammar_compiler.compile_lark.assert_not_called()
        args = from_context.call_args.args
        self.assertEqual(args[:2], (compiled, source))
        self.assertEqual(args[2].dispatch_type, "ebnf")

    def test_dispatch_ebnf_falls_back_to_lark(self):
        source = 'start: "yes" | "no"'
        compiled = MagicMock()
        grammar = MagicMock(spec=BaseGrammarObject)
        self.backend.grammar_compiler.compile_grammar.side_effect = RuntimeError(
            "invalid EBNF grammar"
        )
        self.backend.grammar_compiler.compile_lark.return_value = compiled

        with patch.object(
            self.backend, "_from_context", return_value=grammar
        ) as from_context:
            result = self.backend.dispatch_ebnf(source)

        self.assertIs(result, grammar)
        self.backend.grammar_compiler.compile_grammar.assert_called_once_with(source)
        self.backend.grammar_compiler.compile_lark.assert_called_once_with(source)
        args = from_context.call_args.args
        self.assertEqual(args[:2], (compiled, source))
        self.assertEqual(args[2].dispatch_type, "lark")

    def test_dispatch_ebnf_compiles_valid_grammar_sources(self):
        class EmptyTokenizer:
            def init_xgrammar(self):
                return TokenizerInfo([]), None

        backend = XGrammarGrammarBackend(EmptyTokenizer(), vocab_size=0)
        self.addCleanup(backend.executor.shutdown, wait=True)

        cases = (
            ('root ::= "yes" | "no"', "ebnf"),
            ('start: "yes" | "no"', "lark"),
        )
        for source, dispatch_type in cases:
            with self.subTest(dispatch_type=dispatch_type):
                grammar = backend.dispatch_ebnf(source)
                self.assertNotIsInstance(grammar, InvalidGrammarObject)
                self.assertEqual(grammar.grammar_stats.dispatch_type, dispatch_type)

    def test_dispatch_invalid_grammar_returns_both_errors(self):
        self.backend.grammar_compiler.compile_grammar.side_effect = RuntimeError(
            "invalid EBNF grammar"
        )
        self.backend.grammar_compiler.compile_lark.side_effect = RuntimeError(
            "invalid Lark grammar"
        )

        result = self.backend.dispatch_ebnf("not valid")

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertEqual(
            result.error_message,
            "Failed to compile grammar as EBNF (invalid EBNF grammar) "
            "or Lark (invalid Lark grammar)",
        )


if __name__ == "__main__":
    unittest.main()
