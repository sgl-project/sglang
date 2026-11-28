"""Unit tests for the XGrammar constrained-decoding backend."""

import json
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


# A tiny vocabulary that still spells JSON, so compiled grammars have a real
# first-token mask to inspect.
_SMALL_VOCAB = ["{", "}", '"', "a", "1", "[", "]", ":", ",", " ", "<eos>"]


class _SmallTokenizer:
    def init_xgrammar(self):
        eos_id = len(_SMALL_VOCAB) - 1
        return TokenizerInfo(_SMALL_VOCAB, stop_token_ids=[eos_id]), None


class TestXGrammarGrammarBackend(unittest.TestCase):
    def test_dispatch_ebnf_uses_compile_grammar(self):
        # Bypass tokenizer initialization: this case covers dispatch order only.
        backend = XGrammarGrammarBackend.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()
        backend.override_stop_tokens = None
        source = 'root ::= "yes" | "no"'
        compiled = MagicMock()
        grammar = MagicMock(spec=BaseGrammarObject)
        backend.grammar_compiler.compile_grammar.return_value = compiled

        with (
            patch.object(xgrammar_backend, "_admits_no_token", return_value=False),
            patch.object(
                backend, "_from_context", return_value=grammar
            ) as from_context,
        ):
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

    def test_dispatch_rejects_grammars_that_admit_no_token(self):
        """Unsatisfiable grammars must be rejected up front (HTTP 400), not
        aborted mid-generation once the matcher rejects every token (HTTP 500)."""
        backend = self._make_small_backend()
        self_referencing_schema = json.dumps(
            {"$defs": {"n": {"$ref": "#/$defs/n"}}, "$ref": "#/$defs/n"}
        )
        cases = (
            ("json", backend.dispatch_json, self_referencing_schema),
            ("ebnf", backend.dispatch_ebnf, 'root ::= root "a"'),
            ("regex", backend.dispatch_regex, "[^\\x00-\\U0010FFFF]"),
        )
        for kind, dispatch, source in cases:
            with self.subTest(kind=kind):
                self.assertIsInstance(dispatch(source), InvalidGrammarObject)

    def test_dispatch_rejects_compiled_grammar_that_admits_no_token(self):
        """Covers the guard itself, whichever error XGrammar raises at compile."""
        backend = self._make_small_backend()
        empty = backend.grammar_compiler.compile_grammar('root ::= root "a"')

        with patch.object(
            backend.grammar_compiler, "compile_json_schema", return_value=empty
        ):
            result = backend.dispatch_json('{"type": "object"}')

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("unsatisfiable", result.error_message)

    def test_dispatch_keeps_satisfiable_grammars(self):
        backend = self._make_small_backend()
        recursive_schema = json.dumps(
            {
                "$defs": {
                    "n": {
                        "type": "object",
                        "properties": {
                            "c": {"type": "array", "items": {"$ref": "#/$defs/n"}}
                        },
                    }
                },
                "$ref": "#/$defs/n",
            }
        )
        cases = (
            ("json", backend.dispatch_json, recursive_schema),
            ("json_any", backend.dispatch_json, "$$ANY$$"),
            ("ebnf", backend.dispatch_ebnf, 'root ::= "a" | "1"'),
            ("regex", backend.dispatch_regex, "a+"),
        )
        for kind, dispatch, source in cases:
            with self.subTest(kind=kind):
                self.assertNotIsInstance(dispatch(source), InvalidGrammarObject)

    def test_guard_skips_backends_without_a_vocabulary(self):
        backend = self._make_real_backend()

        result = backend.dispatch_json('{"type": "object"}')

        self.assertNotIsInstance(result, InvalidGrammarObject)

    def _make_real_backend(self):
        backend = XGrammarGrammarBackend(_EmptyTokenizer(), vocab_size=0)
        self.addCleanup(backend.executor.shutdown, wait=True)
        return backend

    def _make_small_backend(self):
        backend = XGrammarGrammarBackend(
            _SmallTokenizer(), vocab_size=len(_SMALL_VOCAB)
        )
        self.addCleanup(backend.executor.shutdown, wait=True)
        return backend


if __name__ == "__main__":
    unittest.main()
