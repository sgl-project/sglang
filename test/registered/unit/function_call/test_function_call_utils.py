"""Unit tests for function_call/utils.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.utils import (
    _find_common_prefix,
    _get_tool_schema,
    _get_tool_schema_defs,
    _is_complete_json,
    get_json_schema_constraint,
    infer_type_from_json_schema,
)
from sglang.test.test_utils import CustomTestCase


def _make_tool(name, parameters=None):
    return Tool(function=Function(name=name, parameters=parameters))


class TestFindCommonPrefix(CustomTestCase):

    def test_identical_strings(self):
        self.assertEqual(_find_common_prefix("abc", "abc"), "abc")

    def test_no_common_prefix(self):
        self.assertEqual(_find_common_prefix("abc", "xyz"), "")

    def test_partial_prefix(self):
        self.assertEqual(_find_common_prefix("abcdef", "abcxyz"), "abc")

    def test_one_empty(self):
        self.assertEqual(_find_common_prefix("abc", ""), "")

    def test_both_empty(self):
        self.assertEqual(_find_common_prefix("", ""), "")

    def test_first_shorter(self):
        self.assertEqual(_find_common_prefix("ab", "abcdef"), "ab")

    def test_second_shorter(self):
        self.assertEqual(_find_common_prefix("abcdef", "ab"), "ab")


class TestIsCompleteJson(CustomTestCase):

    def test_valid_object(self):
        self.assertTrue(_is_complete_json('{"key": "value"}'))

    def test_valid_array(self):
        self.assertTrue(_is_complete_json("[1, 2, 3]"))

    def test_valid_string(self):
        self.assertTrue(_is_complete_json('"hello"'))

    def test_incomplete_object(self):
        self.assertFalse(_is_complete_json('{"key":'))

    def test_incomplete_array(self):
        self.assertFalse(_is_complete_json("[1, 2,"))

    def test_empty_string(self):
        self.assertFalse(_is_complete_json(""))

    def test_nested_complete(self):
        self.assertTrue(_is_complete_json('{"a": {"b": [1, 2]}}'))

    def test_nested_incomplete(self):
        self.assertFalse(_is_complete_json('{"a": {"b": [1, 2'))


class TestInferTypeFromJsonSchema(CustomTestCase):

    def test_direct_type_string(self):
        self.assertEqual(infer_type_from_json_schema({"type": "string"}), "string")

    def test_direct_type_integer(self):
        self.assertEqual(infer_type_from_json_schema({"type": "integer"}), "integer")

    def test_type_array_filters_null(self):
        """type: ["string", "null"] should return "string", skipping null."""
        self.assertEqual(
            infer_type_from_json_schema({"type": ["string", "null"]}), "string"
        )

    def test_type_array_only_null_defaults_to_string(self):
        self.assertEqual(infer_type_from_json_schema({"type": ["null"]}), "string")

    def test_anyof_uniform_types(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "string"}]}
        self.assertEqual(infer_type_from_json_schema(schema), "string")

    def test_anyof_mixed_types_prefers_string(self):
        schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
        self.assertEqual(infer_type_from_json_schema(schema), "string")

    def test_anyof_without_string_returns_first(self):
        schema = {"anyOf": [{"type": "integer"}, {"type": "number"}]}
        self.assertEqual(infer_type_from_json_schema(schema), "integer")

    def test_oneof_handled_same_as_anyof(self):
        schema = {"oneOf": [{"type": "boolean"}, {"type": "boolean"}]}
        self.assertEqual(infer_type_from_json_schema(schema), "boolean")

    def test_enum_all_strings(self):
        self.assertEqual(
            infer_type_from_json_schema({"enum": ["a", "b", "c"]}), "string"
        )

    def test_enum_all_integers(self):
        self.assertEqual(infer_type_from_json_schema({"enum": [1, 2, 3]}), "integer")

    def test_enum_mixed_returns_string(self):
        self.assertEqual(
            infer_type_from_json_schema({"enum": [1, "a", True]}), "string"
        )

    def test_enum_empty(self):
        self.assertEqual(infer_type_from_json_schema({"enum": []}), "string")

    def test_enum_with_none_value(self):
        self.assertEqual(infer_type_from_json_schema({"enum": [None]}), "null")

    def test_enum_with_list_values(self):
        self.assertEqual(
            infer_type_from_json_schema({"enum": [[1, 2], [3, 4]]}), "array"
        )

    def test_enum_with_dict_values(self):
        self.assertEqual(infer_type_from_json_schema({"enum": [{"a": 1}]}), "object")

    def test_allof_returns_non_string_type(self):
        """allOf skips string sub-schemas and returns the first non-string type."""
        schema = {"allOf": [{"type": "string"}, {"type": "object"}]}
        self.assertEqual(infer_type_from_json_schema(schema), "object")

    def test_allof_all_string_returns_string(self):
        schema = {"allOf": [{"type": "string"}]}
        self.assertEqual(infer_type_from_json_schema(schema), "string")

    def test_properties_implies_object(self):
        schema = {"properties": {"name": {"type": "string"}}}
        self.assertEqual(infer_type_from_json_schema(schema), "object")

    def test_items_implies_array(self):
        schema = {"items": {"type": "string"}}
        self.assertEqual(infer_type_from_json_schema(schema), "array")

    def test_non_dict_returns_none(self):
        self.assertIsNone(infer_type_from_json_schema("not a dict"))

    def test_empty_dict_returns_none(self):
        self.assertIsNone(infer_type_from_json_schema({}))


class TestGetToolSchema(CustomTestCase):

    def test_with_parameters(self):
        tool = _make_tool(
            "weather",
            {"type": "object", "properties": {"city": {"type": "string"}}},
        )
        schema = _get_tool_schema(tool)
        self.assertEqual(schema["properties"]["name"]["enum"], ["weather"])
        self.assertEqual(
            schema["properties"]["parameters"]["properties"]["city"]["type"], "string"
        )
        self.assertEqual(schema["required"], ["name", "parameters"])

    def test_none_parameters_gets_empty_object(self):
        """When tool.function.parameters is None, schema should substitute
        an empty object so downstream JSON schema validation still works."""
        tool = _make_tool("ping", None)
        schema = _get_tool_schema(tool)
        self.assertEqual(
            schema["properties"]["parameters"],
            {"type": "object", "properties": {}},
        )


class TestGetToolSchemaDefs(CustomTestCase):

    def test_no_defs(self):
        tools = [_make_tool("a", {"type": "object"})]
        self.assertEqual(_get_tool_schema_defs(tools), {})

    def test_identical_defs_merged(self):
        shared_def = {"type": "string"}
        tools = [
            _make_tool("a", {"$defs": {"MyType": shared_def}}),
            _make_tool("b", {"$defs": {"MyType": shared_def}}),
        ]
        result = _get_tool_schema_defs(tools)
        self.assertEqual(result, {"MyType": shared_def})

    def test_conflicting_defs_raise(self):
        tools = [
            _make_tool("a", {"$defs": {"MyType": {"type": "string"}}}),
            _make_tool("b", {"$defs": {"MyType": {"type": "integer"}}}),
        ]
        with self.assertRaises(ValueError):
            _get_tool_schema_defs(tools)

    def test_none_parameters_skipped(self):
        tools = [_make_tool("a", None)]
        self.assertEqual(_get_tool_schema_defs(tools), {})


class TestGetJsonSchemaConstraint(CustomTestCase):

    def test_specific_tool_choice(self):
        tools = [
            _make_tool(
                "weather",
                {"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ]
        choice = ToolChoice(function=ToolChoiceFuncName(name="weather"))
        result = get_json_schema_constraint(tools, choice)
        self.assertEqual(result["type"], "array")
        self.assertEqual(result["minItems"], 1)
        # parallel_tool_calls defaults to True, so maxItems is not set (#20208)
        self.assertNotIn("maxItems", result)

    def test_specific_tool_choice_not_found_returns_none(self):
        tools = [_make_tool("weather", {"type": "object"})]
        choice = ToolChoice(function=ToolChoiceFuncName(name="nonexistent"))
        self.assertIsNone(get_json_schema_constraint(tools, choice))

    def test_required_lists_all_tools(self):
        tools = [
            _make_tool("a", {"type": "object"}),
            _make_tool("b", {"type": "object"}),
        ]
        result = get_json_schema_constraint(tools, "required")
        self.assertEqual(result["type"], "array")
        self.assertEqual(result["minItems"], 1)
        self.assertEqual(len(result["items"]["anyOf"]), 2)

    def test_required_includes_defs(self):
        tools = [_make_tool("a", {"$defs": {"T": {"type": "string"}}})]
        result = get_json_schema_constraint(tools, "required")
        self.assertIn("T", result["$defs"])

    def test_auto_returns_none(self):
        self.assertIsNone(
            get_json_schema_constraint([_make_tool("a", {"type": "object"})], "auto")
        )


if __name__ == "__main__":
    unittest.main()
