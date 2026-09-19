"""Unit tests for backend-specific JSON Schema prevalidation."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.json_schema_validation import (
    JSONSchemaDepthExceeded,
    JSONSchemaStateExplosion,
    UnsupportedJSONSchemaFeature,
    validate_outlines_json_schema,
    validate_schema_bounds,
    validate_xgrammar_json_schema,
)
from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


class TestXGrammarJSONSchemaValidation(unittest.TestCase):
    def test_accepts_supported_schema(self):
        validate_xgrammar_json_schema(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "created": {"type": "string", "format": "date-time"},
                    "score": {"type": "number", "minimum": 0},
                },
                "required": ["name"],
                "additionalProperties": False,
            }
        )

    def test_rejects_every_silently_ignored_keyword(self):
        for keyword in (
            "contains",
            "dependentRequired",
            "dependentSchemas",
            "else",
            "if",
            "maxContains",
            "minContains",
            "multipleOf",
            "not",
            "then",
            "uniqueItems",
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(
                    UnsupportedJSONSchemaFeature, rf"keyword\(s\) {keyword}"
                ):
                    validate_xgrammar_json_schema({keyword: {}})

    def test_rejects_unknown_format(self):
        with self.assertRaisesRegex(
            UnsupportedJSONSchemaFeature, "format 'regex' is not implemented"
        ):
            validate_xgrammar_json_schema({"type": "string", "format": "regex"})

    def test_accepts_non_string_format(self):
        validate_xgrammar_json_schema({"type": "integer", "format": "int64"})
        validate_xgrammar_json_schema({"type": "number", "format": "double"})
        validate_xgrammar_json_schema({"type": ["integer", "null"], "format": "int64"})

    def test_checks_format_when_schema_can_describe_string(self):
        for schema in (
            {"format": "markdown"},
            {"type": ["string", "null"], "format": "markdown"},
        ):
            with self.subTest(schema=schema):
                with self.assertRaisesRegex(
                    UnsupportedJSONSchemaFeature,
                    "format 'markdown' is not implemented",
                ):
                    validate_xgrammar_json_schema(schema)

    def test_rejects_lossy_string_constraint_combinations(self):
        cases = (
            {"type": "string", "pattern": "^[a-z]+$", "minLength": 2},
            {"type": "string", "format": "uuid", "maxLength": 36},
            {"type": "string", "format": "uuid", "pattern": "^a"},
        )
        for schema in cases:
            with self.subTest(schema=schema):
                with self.assertRaisesRegex(
                    UnsupportedJSONSchemaFeature, "cannot be enforced together"
                ):
                    validate_xgrammar_json_schema(schema)

    def test_checks_nested_schema_without_walking_instance_data(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {"type": "array", "items": {"multipleOf": 2}},
            },
            "const": {"multipleOf": 2},
            "default": {"multipleOf": 2},
            "enum": [{"multipleOf": 2}],
            "examples": [{"multipleOf": 2}],
        }
        with self.assertRaisesRegex(
            UnsupportedJSONSchemaFeature, "#/properties/nested/items"
        ):
            validate_xgrammar_json_schema(schema)

        validate_xgrammar_json_schema(
            {
                "const": {"multipleOf": 2},
                "default": {"multipleOf": 2},
                "enum": [{"multipleOf": 2}],
                "examples": [{"multipleOf": 2}],
            }
        )


class TestOutlinesJSONSchemaValidation(unittest.TestCase):
    def test_accepts_supported_schema(self):
        validate_outlines_json_schema(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-z]+$"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            }
        )

    def test_rejects_constraints_outlines_silently_weakens(self):
        cases = (
            {"type": "number", "minimum": 0},
            {"type": "array", "uniqueItems": True},
            {"type": "object", "patternProperties": {"^x": {}}},
            {"allOf": [{"type": "string"}, {"minLength": 2}]},
            {"oneOf": [{"type": "number"}, {"minimum": 0}]},
        )
        for schema in cases:
            with self.subTest(schema=schema):
                with self.assertRaisesRegex(
                    UnsupportedJSONSchemaFeature, "not supported by outlines"
                ):
                    validate_outlines_json_schema(schema)

    def test_rejects_constraints_shadowed_by_outlines_dispatch_order(self):
        cases = (
            {
                "type": "string",
                "pattern": "^[a-z]+$",
                "minLength": 2,
            },
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": True,
            },
            {
                "type": "array",
                "prefixItems": [{"type": "string"}],
                "minItems": 2,
            },
            {"type": "object", "required": ["name"]},
        )
        for schema in cases:
            with self.subTest(schema=schema):
                with self.assertRaises(UnsupportedJSONSchemaFeature):
                    validate_outlines_json_schema(schema)

    def test_accepts_non_string_format_and_string_keywords(self):
        validate_outlines_json_schema(
            {
                "type": "integer",
                "format": "int64",
                "pattern": "ignored-for-integers",
                "minLength": 1,
            }
        )

    def test_escapes_nested_json_pointer(self):
        with self.assertRaisesRegex(
            UnsupportedJSONSchemaFeature, r"#/properties/a~1b~0c"
        ):
            validate_outlines_json_schema(
                {"properties": {"a/b~c": {"type": "integer", "multipleOf": 2}}}
            )


class TestBackendJSONSchemaPrevalidation(unittest.TestCase):
    def test_xgrammar_rejects_before_compilation(self):
        backend = object.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()

        result = backend.dispatch_json(
            '{"type":"string","pattern":"^[a-z]+$","minLength":2}'
        )

        self.assertIsInstance(result, InvalidGrammarObject)
        backend.grammar_compiler.compile_json_schema.assert_not_called()

    def test_outlines_rejects_before_regex_compilation(self):
        backend = object.__new__(OutlinesGrammarBackend)
        backend._compile_regex = MagicMock()

        result = backend.dispatch_json('{"type":"number","minimum":0}')

        self.assertIsInstance(result, InvalidGrammarObject)
        backend._compile_regex.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TestSchemaBoundsValidation(unittest.TestCase):
    """Tests for schema depth and state explosion validation."""

    def test_validate_schema_bounds_accepts_valid_schema(self):
        """A normal schema should pass validation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        validate_schema_bounds(schema)

    def test_validate_schema_bounds_rejects_excessive_depth(self):
        """Schema with nesting depth > 16 should be rejected."""
        schema = {"type": "object"}
        current = schema
        for i in range(17):  # 17 levels = depth 17 > 16
            current["properties"] = {"nested": {"type": "object"}}
            current = current["properties"]["nested"]

        with self.assertRaises(JSONSchemaDepthExceeded):
            validate_schema_bounds(schema)

    def test_validate_schema_bounds_accepts_boundary_depth(self):
        """Schema at exactly depth 16 should pass."""
        schema = {"type": "object"}
        current = schema
        for i in range(16):
            current["properties"] = {"nested": {"type": "object"}}
            current = current["properties"]["nested"]

        validate_schema_bounds(schema)

    def test_validate_schema_bounds_rejects_excessive_states(self):
        """Schema with too many properties should be rejected."""
        properties = {}
        for i in range(6000):
            properties[f"prop_{i}"] = {"type": "string"}
        schema = {
            "type": "object",
            "properties": properties,
        }

        with self.assertRaises(JSONSchemaStateExplosion):
            validate_schema_bounds(schema)

    def test_validate_schema_bounds_rejects_allof_explosion(self):
        """Schema with many allOf clauses should be rejected."""
        all_of = []
        for i in range(3000):
            all_of.append(
                {"type": "object", "properties": {f"x{i}": {"type": "string"}}}
            )
        schema = {
            "type": "object",
            "allOf": all_of,
        }

        with self.assertRaises(JSONSchemaStateExplosion):
            validate_schema_bounds(schema)

    def test_xgrammar_rejects_deep_schema_before_compilation(self):
        """XGrammar backend should reject deep schemas before compilation."""
        backend = object.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()

        # Create a deeply nested schema
        schema = {"type": "object"}
        current = schema
        for i in range(17):
            current["properties"] = {"nested": {"type": "object"}}
            current = current["properties"]["nested"]

        import json

        result = backend.dispatch_json(json.dumps(schema))

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("nesting depth exceeds", result.error_message)
        backend.grammar_compiler.compile_json_schema.assert_not_called()

    def test_xgrammar_rejects_state_explosion_before_compilation(self):
        """XGrammar backend should reject state explosion before compilation."""
        backend = object.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()

        properties = {}
        for i in range(6000):
            properties[f"prop_{i}"] = {"type": "string"}
        schema = {"type": "object", "properties": properties}

        import json

        result = backend.dispatch_json(json.dumps(schema))

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("DFA states", result.error_message)
        backend.grammar_compiler.compile_json_schema.assert_not_called()

    def test_outlines_rejects_deep_schema_before_compilation(self):
        """Outlines backend should reject deep schemas before regex compilation."""
        backend = object.__new__(OutlinesGrammarBackend)
        backend._compile_regex = MagicMock()

        schema = {"type": "object"}
        current = schema
        for i in range(17):
            current["properties"] = {"nested": {"type": "object"}}
            current = current["properties"]["nested"]

        import json

        result = backend.dispatch_json(json.dumps(schema))

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("nesting depth exceeds", result.error_message)
        backend._compile_regex.assert_not_called()

    def test_outlines_rejects_state_explosion_before_compilation(self):
        """Outlines backend should reject state explosion before regex compilation."""
        backend = object.__new__(OutlinesGrammarBackend)
        backend._compile_regex = MagicMock()

        properties = {}
        for i in range(6000):
            properties[f"prop_{i}"] = {"type": "string"}
        schema = {"type": "object", "properties": properties}

        import json

        result = backend.dispatch_json(json.dumps(schema))

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("DFA states", result.error_message)
        backend._compile_regex.assert_not_called()
