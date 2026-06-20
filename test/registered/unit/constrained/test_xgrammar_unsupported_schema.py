"""Unit tests for xgrammar's unsupported-JSON-feature detection."""

import unittest

from sglang.srt.constrained.xgrammar_backend import (
    has_xgrammar_unsupported_json_features,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


class TestHasXGrammarUnsupportedJsonFeatures(unittest.TestCase):
    def test_supported_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name"],
        }
        self.assertFalse(has_xgrammar_unsupported_json_features(schema))

    def test_unsupported_keywords(self):
        cases = [
            {"type": "integer", "multipleOf": 2},
            {"type": "number", "multipleOf": 0.5},
            # Nullable field: "type" as a list must still be detected.
            {"type": ["integer", "null"], "multipleOf": 2},
            {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
            {
                "type": "array",
                "items": {"type": "integer"},
                "contains": {"type": "integer"},
            },
            {"type": "array", "items": {"type": "integer"}, "maxContains": 2},
            {"type": "object", "patternProperties": {"^x": {"type": "string"}}},
            {"type": "object", "propertyNames": {"pattern": "^x"}},
        ]
        for schema in cases:
            with self.subTest(schema=schema):
                self.assertTrue(has_xgrammar_unsupported_json_features(schema))

    def test_unsupported_feature_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer", "multipleOf": 3},
                }
            },
        }
        self.assertTrue(has_xgrammar_unsupported_json_features(schema))


if __name__ == "__main__":
    unittest.main()
