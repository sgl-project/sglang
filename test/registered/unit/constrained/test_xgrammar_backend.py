"""Unit tests for XGrammarGrammarBackend error logging — no server, no model.

Regression test for a log/response explosion: when grammar compilation fails,
the dispatch_* methods log the offending grammar via `{key_string=}`. The
`key_string` is the RAW user-supplied grammar/schema/regex taken straight from
the request (`response_format` / `regex` / `ebnf` / `json_schema`), bounded only
by the HTTP body size — there is no app-level cap. Without truncation, a single
invalid multi-MB grammar is echoed verbatim into a `logger.error` line (and a
malicious client can repeat it), bloating the logs.

These tests exercise the REAL dispatch_* code that the server invokes per
request; only the grammar compiler is stubbed to force the failure branch.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.test_utils import CustomTestCase

XGRAMMAR_LOGGER = "sglang.srt.constrained.xgrammar_backend"

# A single ERROR line must never carry the whole user grammar. The fix truncates
# the echoed grammar to a small bound; this generous ceiling proves a ~50KB
# payload was cut while staying agnostic to the exact limit chosen.
MAX_LOG_CHARS = 2000

# (dispatch method, the grammar_compiler method it calls, expected log marker)
DISPATCH_CASES = [
    ("dispatch_json", "compile_json_schema", "invalid json_schema"),
    ("dispatch_ebnf", "compile_grammar", "invalid ebnf"),
    ("dispatch_regex", "compile_regex", "invalid regex"),
]


def _backend_with_failing_compiler(compiler_method: str) -> XGrammarGrammarBackend:
    # Build a real backend instance without __init__ (which needs a tokenizer and
    # builds a GrammarCompiler), then stub only what dispatch_* touches.
    backend = object.__new__(XGrammarGrammarBackend)
    backend.grammar_compiler = MagicMock()
    getattr(backend.grammar_compiler, compiler_method).side_effect = RuntimeError(
        "compile failed"
    )
    backend.any_whitespace = False
    return backend


class TestXGrammarErrorLoggingIsBounded(CustomTestCase):
    def test_invalid_grammar_log_does_not_dump_full_payload(self):
        payload = "A" * 50_000  # stand-in for a multi-MB user grammar

        for method_name, compiler_method, marker in DISPATCH_CASES:
            with self.subTest(dispatch=method_name):
                backend = _backend_with_failing_compiler(compiler_method)
                dispatch = getattr(backend, method_name)

                with self.assertLogs(XGRAMMAR_LOGGER, level="ERROR") as cm:
                    result = dispatch(payload)

                # Failure path returns a placeholder object, it does not raise.
                self.assertIsInstance(result, InvalidGrammarObject)

                log_output = "\n".join(cm.output)
                # The error stays useful (still says which grammar kind failed)...
                self.assertIn(marker, log_output)
                # ...but the full user grammar must not be echoed verbatim.
                self.assertNotIn(payload, log_output)
                self.assertLess(len(log_output), MAX_LOG_CHARS)


if __name__ == "__main__":
    unittest.main()
