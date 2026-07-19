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
            {"multipleOf": 2},
            {"type": "integer", "multipleOf": 2},
            {"type": "number", "multipleOf": 0.5},
            # Nullable field: "type" as a list must still be detected.
            {"type": ["integer", "null"], "multipleOf": 2},
            {"uniqueItems": True},
            {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
            {"contains": {"type": "integer"}},
            {
                "type": "array",
                "items": {"type": "integer"},
                "contains": {"type": "integer"},
            },
            {"minContains": 1},
            {"maxContains": 2},
            {"type": "array", "items": {"type": "integer"}, "maxContains": 2},
            {"patternProperties": {"^x": {"type": "string"}}},
            {"type": "object", "patternProperties": {"^x": {"type": "string"}}},
            {"propertyNames": {"pattern": "^x"}},
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

    def test_property_names_that_match_unsupported_keywords_are_allowed(self):
        schema = {
            "type": "object",
            "properties": {
                "multipleOf": {"type": "number"},
                "uniqueItems": {"type": "boolean"},
                "contains": {"type": "string"},
                "patternProperties": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        }
        self.assertFalse(has_xgrammar_unsupported_json_features(schema))


if __name__ == "__main__":
    unittest.main()
