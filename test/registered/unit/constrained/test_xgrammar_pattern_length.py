"""Unit tests for xgrammar pattern and string-length schema combinations."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammarBackend,
    has_xgrammar_unsupported_pattern_length_combination,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestXGrammarPatternLengthCombination(unittest.TestCase):
    def test_rejects_pattern_with_min_length(self):
        schema = {
            "type": "string",
            "pattern": "^[a-z]+$",
            "minLength": 5,
        }

        self.assertTrue(has_xgrammar_unsupported_pattern_length_combination(schema))

    def test_rejects_pattern_with_max_length(self):
        schema = {
            "type": "string",
            "pattern": "^[a-z]+$",
            "maxLength": 5,
        }

        self.assertTrue(has_xgrammar_unsupported_pattern_length_combination(schema))

    def test_rejects_nested_combination(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^[a-z]+$",
                        "minLength": 5,
                    },
                }
            },
        }

        self.assertTrue(has_xgrammar_unsupported_pattern_length_combination(schema))

    def test_allows_each_keyword_individually(self):
        self.assertFalse(
            has_xgrammar_unsupported_pattern_length_combination(
                {"type": "string", "pattern": "^[a-z]+$"}
            )
        )
        self.assertFalse(
            has_xgrammar_unsupported_pattern_length_combination(
                {"type": "string", "minLength": 5}
            )
        )
        self.assertFalse(
            has_xgrammar_unsupported_pattern_length_combination(
                {"type": "string", "maxLength": 5}
            )
        )

    def test_does_not_confuse_property_names_with_schema_keywords(self):
        schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "minLength": 5},
            },
        }

        self.assertFalse(has_xgrammar_unsupported_pattern_length_combination(schema))

    def test_dispatch_returns_invalid_grammar_without_compiling(self):
        backend = object.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()

        result = backend.dispatch_json(
            '{"type":"string","pattern":"^[a-z]+$","minLength":5}'
        )

        self.assertIsInstance(result, InvalidGrammarObject)
        backend.grammar_compiler.compile_json_schema.assert_not_called()


if __name__ == "__main__":
    unittest.main()
