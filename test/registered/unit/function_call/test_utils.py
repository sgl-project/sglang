"""
Tests for utility functions in sglang.srt.function_call.utils
"""

import unittest

import partial_json_parser
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.utils import (
    _find_common_prefix,
    _get_tool_schema,
    _is_complete_json,
    _partial_json_loads,
    _get_tool_schema_defs,
    infer_type_from_json_schema,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "stage-a-test-cpu")


class TestFindCommonPrefix(unittest.TestCase):
    """Test _find_common_prefix utility function"""

    def test_no_common_prefix(self):
        """Test strings with no common prefix"""
        self.assertEqual(_find_common_prefix("abc", "xyz"), "")

    def test_full_common_prefix(self):
        """Test when one string is a complete prefix of the other"""
        self.assertEqual(_find_common_prefix("hello", "hello"), "hello")
        self.assertEqual(_find_common_prefix("hello", "hellooooo"), "hello")

    def test_partial_common_prefix(self):
        """Test partial common prefix"""
        self.assertEqual(_find_common_prefix("hello world", "hello there"), "hello ")
        self.assertEqual(_find_common_prefix("prefix", "preheat"), "pre")

    def test_empty_strings(self):
        """Test with empty strings"""
        self.assertEqual(_find_common_prefix("", ""), "")
        self.assertEqual(_find_common_prefix("", "abc"), "")

    def test_single_character(self):
        """Test single character strings"""
        self.assertEqual(_find_common_prefix("a", "a"), "a")
        self.assertEqual(_find_common_prefix("a", "b"), "")

    def test_case_sensitive(self):
        """Test that comparison is case sensitive"""
        self.assertEqual(_find_common_prefix("Hello", "hello"), "")
        self.assertEqual(_find_common_prefix("ABC", "ABCDEF"), "ABC")


class TestPartialJsonLoads(unittest.TestCase):
    """Test _partial_json_loads utility function"""

    def test_complete_object(self):
        """Test parsing complete JSON object"""
        result, consumed = _partial_json_loads('{"key": "value"}', Allow.ALL)
        self.assertEqual(result, {"key": "value"})
        self.assertEqual(consumed, len('{"key": "value"}'))

    def test_complete_array(self):
        """Test parsing complete JSON array"""
        result, consumed = _partial_json_loads('[1, 2, 3]', Allow.ALL)
        self.assertEqual(result, [1, 2, 3])

    def test_partial_string(self):
        """Test parsing partial JSON string"""
        result, consumed = _partial_json_loads('"hello wo', Allow.STR)
        self.assertEqual(result, "hello wo")

    def test_partial_object(self):
        """Test parsing partial JSON object"""
        result, consumed = _partial_json_loads('{"key":', Allow.OBJ)
        self.assertEqual(result, {"key": None})

    def test_partial_array(self):
        """Test parsing partial JSON array"""
        result, consumed = _partial_json_loads('[1, 2,', Allow.ARR)
        self.assertEqual(result, [1, 2])

    def test_nested_partial(self):
        """Test parsing nested partial JSON"""
        result, consumed = _partial_json_loads('{"name": "John", "age": 25}', Allow.ALL)
        self.assertEqual(result, {"name": "John", "age": 25})

    def test_extra_data_handling(self):
        """Test handling of extra data after valid JSON"""
        result, consumed = _partial_json_loads('{"a":1}"extra"', Allow.ALL)
        self.assertEqual(result, {"a": 1})
        self.assertEqual(consumed, 7)

    def test_invalid_json_raises(self):
        """Test that invalid JSON raises appropriate exception"""
        with self.assertRaises(Exception):
            _partial_json_loads("not json at all", Allow.ALL)

    def test_allow_none_flag(self):
        """Test with Allow.NONE flag for strict parsing"""
        with self.assertRaises(Exception):
            _partial_json_loads('{"key": "value"}', Allow.NONE)


class TestIsCompleteJson(unittest.TestCase):
    """Test _is_complete_json utility function"""

    def test_valid_object(self):
        """Test valid JSON object returns True"""
        self.assertTrue(_is_complete_json('{"key": "value"}'))
        self.assertTrue(_is_complete_json("{}"))
        self.assertTrue(_is_complete_json('{"a": 1, "b": [1, 2, 3]}'))

    def test_valid_array(self):
        """Test valid JSON array returns True"""
        self.assertTrue(_is_complete_json("[1, 2, 3]"))
        self.assertTrue(_is_complete_json("[]"))

    def test_valid_primitives(self):
        """Test valid JSON primitives return True"""
        self.assertTrue(_is_complete_json('"string"'))
        self.assertTrue(_is_complete_json("123"))
        self.assertTrue(_is_complete_json("true"))
        self.assertTrue(_is_complete_json("null"))

    def test_invalid_object_returns_false(self):
        """Test invalid JSON object returns False"""
        self.assertFalse(_is_complete_json('{"key": "value"'))  # missing closing
        self.assertFalse(_is_complete_json('{"key": }'))  # invalid value
        self.assertFalse(_is_complete_json("{key: value}"))  # unquoted key

    def test_invalid_array_returns_false(self):
        """Test invalid JSON array returns False"""
        self.assertFalse(_is_complete_json("[1, 2, 3"))  # missing closing
        self.assertFalse(_is_complete_json("[1, 2,]"))  # trailing comma invalid

    def test_invalid_string_returns_false(self):
        """Test invalid JSON string returns False"""
        self.assertFalse(_is_complete_json('"incomplete string'))
        self.assertFalse(_is_complete_json("not json"))

    def test_empty_string_returns_false(self):
        """Test empty string returns False"""
        self.assertFalse(_is_complete_json(""))

    def test_whitespace_only_returns_false(self):
        """Test whitespace-only string returns False"""
        self.assertFalse(_is_complete_json("   "))

    def test_nested_json(self):
        """Test nested JSON structures"""
        nested = '{"a": {"b": {"c": [1, 2, {"d": true}]}}}'
        self.assertTrue(_is_complete_json(nested))


class TestGetToolSchema(unittest.TestCase):
    """Test _get_tool_schema utility function"""

    def test_basic_tool_schema(self):
        """Test getting schema for a basic tool"""
        tool = Tool(
            type="function",
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                },
            ),
        )
        schema = _get_tool_schema(tool)

        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        self.assertEqual(schema["properties"]["name"]["enum"], ["get_weather"])
        self.assertIn("parameters", schema["properties"])
        self.assertIn("name", schema["required"])
        self.assertIn("parameters", schema["required"])

    def test_tool_with_complex_parameters(self):
        """Test tool with complex parameter schema"""
        tool = Tool(
            type="function",
            function=Function(
                name="search",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["query"],
                },
            ),
        )
        schema = _get_tool_schema(tool)

        self.assertEqual(schema["properties"]["name"]["enum"], ["search"])
        params = schema["properties"]["parameters"]
        self.assertEqual(params["properties"]["query"]["type"], "string")
        self.assertEqual(params["properties"]["limit"]["type"], "integer")

    def test_tool_with_null_parameters(self):
        """Test tool with null parameters"""
        tool = Tool(
            type="function",
            function=Function(
                name="no_params_tool",
                parameters=None,
            ),
        )
        schema = _get_tool_schema(tool)

        self.assertEqual(
            schema["properties"]["parameters"],
            {"type": "object", "properties": {}},
        )

    def test_tool_with_refs(self):
        """Test tool with $ref in parameters"""
        tool = Tool(
            type="function",
            function=Function(
                name="ref_tool",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"$ref": "#/$defs/DataType"},
                    },
                    "$defs": {
                        "DataType": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        }
                    },
                },
            ),
        )
        schema = _get_tool_schema(tool)

        self.assertEqual(schema["properties"]["name"]["enum"], ["ref_tool"])
        self.assertIn("$ref", schema["properties"]["parameters"]["properties"]["data"])


class TestInferTypeFromJsonSchema(unittest.TestCase):
    """Test infer_type_from_json_schema utility function"""

    def test_direct_string_type(self):
        """Test inferring string type from direct type field"""
        self.assertEqual(infer_type_from_json_schema({"type": "string"}), "string")

    def test_direct_number_type(self):
        """Test inferring number type from direct type field"""
        self.assertEqual(infer_type_from_json_schema({"type": "number"}), "number")

    def test_direct_integer_type(self):
        """Test inferring integer type from direct type field"""
        self.assertEqual(infer_type_from_json_schema({"type": "integer"}), "integer")

    def test_direct_boolean_type(self):
        """Test inferring boolean type from direct type field"""
        self.assertEqual(infer_type_from_json_schema({"type": "boolean"}), "boolean")

    def test_direct_object_type(self):
        """Test inferring object type from direct type field"""
        self.assertEqual(infer_type_from_json_schema({"type": "object"}), "object")

    def test_direct_array_type(self):
        """Test inferring array type from direct type field"""
        self.assertEqual(infer_type_from_json_schema({"type": "array"}), "array")

    def test_type_array_returns_first_non_null(self):
        """Test that type arrays return first non-null type"""
        self.assertEqual(
            infer_type_from_json_schema({"type": ["string", "null"]}), "string"
        )
        self.assertEqual(
            infer_type_from_json_schema({"type": ["number", "string"]}), "number"
        )
        self.assertEqual(
            infer_type_from_json_schema({"type": [None, "string"]}), "string"
        )

    def test_type_array_all_null_defaults_to_string(self):
        """Test that type array with only null defaults to string"""
        self.assertEqual(
            infer_type_from_json_schema({"type": [None, None]}), "string"
        )

    def test_any_of_with_single_type(self):
        """Test anyOf with single type"""
        self.assertEqual(
            infer_type_from_json_schema({"anyOf": [{"type": "string"}]}), "string"
        )

    def test_any_of_with_multiple_same_type(self):
        """Test anyOf with multiple schemas of same type"""
        self.assertEqual(
            infer_type_from_json_schema(
                {"anyOf": [{"type": "string"}, {"type": "string"}]}
            ),
            "string",
        )

    def test_any_of_with_different_types_prioritizes_string(self):
        """Test anyOf with different types prioritizes string"""
        self.assertEqual(
            infer_type_from_json_schema(
                {"anyOf": [{"type": "number"}, {"type": "string"}]}
            ),
            "string",
        )

    def test_one_of_with_single_type(self):
        """Test oneOf with single type"""
        self.assertEqual(
            infer_type_from_json_schema({"oneOf": [{"type": "boolean"}]}), "boolean"
        )

    def test_one_of_with_different_types(self):
        """Test oneOf with different types"""
        self.assertEqual(
            infer_type_from_json_schema(
                {"oneOf": [{"type": "integer"}, {"type": "number"}]}
            ),
            "integer",
        )

    def test_enum_with_string_values(self):
        """Test inferring string type from string enum"""
        self.assertEqual(
            infer_type_from_json_schema({"enum": ["a", "b", "c"]}), "string"
        )

    def test_enum_with_integer_values(self):
        """Test inferring integer type from integer enum"""
        self.assertEqual(infer_type_from_json_schema({"enum": [1, 2, 3]}), "integer")

    def test_enum_with_mixed_values_returns_string(self):
        """Test that mixed enum values returns string"""
        self.assertEqual(
            infer_type_from_json_schema({"enum": ["a", 1, True]}), "string"
        )

    def test_enum_with_boolean_values(self):
        """Test inferring boolean type from boolean enum"""
        self.assertEqual(
            infer_type_from_json_schema({"enum": [True, False]}), "boolean"
        )

    def test_enum_empty_returns_string(self):
        """Test that empty enum returns string"""
        self.assertEqual(infer_type_from_json_schema({"enum": []}), "string")

    def test_properties_infers_object(self):
        """Test that properties infers object type"""
        self.assertEqual(
            infer_type_from_json_schema({"properties": {"name": {"type": "string"}}}),
            "object",
        )

    def test_items_infers_array(self):
        """Test that items infers array type"""
        self.assertEqual(
            infer_type_from_json_schema({"items": {"type": "string"}}), "array"
        )

    def test_all_of_with_non_string_type(self):
        """Test allOf with non-string type returns that type"""
        self.assertEqual(
            infer_type_from_json_schema(
                {"allOf": [{"type": "object"}, {"type": "string"}]}
            ),
            "string",
        )

    def test_all_of_only_string(self):
        """Test allOf with only string types returns string"""
        self.assertEqual(
            infer_type_from_json_schema(
                {"allOf": [{"type": "string"}, {"type": "string"}]}
            ),
            "string",
        )

    def test_non_dict_input_returns_none(self):
        """Test that non-dict input returns None"""
        self.assertIsNone(infer_type_from_json_schema("string"))
        self.assertIsNone(infer_type_from_json_schema(123))
        self.assertIsNone(infer_type_from_json_schema(None))
        self.assertIsNone(infer_type_from_json_schema(["array"]))

    def test_empty_dict_returns_none(self):
        """Test that empty dict returns None"""
        self.assertIsNone(infer_type_from_json_schema({}))

    def test_nested_any_of(self):
        """Test nested anyOf structures"""
        self.assertEqual(
            infer_type_from_json_schema(
                {
                    "anyOf": [
                        {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        {"type": "boolean"},
                    ]
                }
            ),
            "string",
        )

    def test_complex_nested_schema(self):
        """Test complex nested JSON Schema"""
        schema = {
            "anyOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            ]
        }
        result = infer_type_from_json_schema(schema)
        self.assertIn(result, ["string", "object"])


class TestGetToolSchemaDefs(unittest.TestCase):
    """Test _get_tool_schema_defs utility function"""

    def test_single_tool_with_defs(self):
        """Test getting defs from a single tool"""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="tool1",
                    parameters={
                        "type": "object",
                        "properties": {"data": {"$ref": "#/$defs/DataType"}},
                        "$defs": {
                            "DataType": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            }
                        },
                    },
                ),
            )
        ]
        defs = _get_tool_schema_defs(tools)
        self.assertIn("DataType", defs)
        self.assertEqual(defs["DataType"]["type"], "object")

    def test_multiple_tools_with_defs(self):
        """Test getting defs from multiple tools"""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="tool1",
                    parameters={
                        "$defs": {"Type1": {"type": "object", "properties": {}}}
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="tool2",
                    parameters={
                        "$defs": {"Type2": {"type": "object", "properties": {}}}
                    },
                ),
            ),
        ]
        defs = _get_tool_schema_defs(tools)
        self.assertEqual(len(defs), 2)
        self.assertIn("Type1", defs)
        self.assertIn("Type2", defs)

    def test_tool_without_defs(self):
        """Test tool without any $defs"""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="simple_tool",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                ),
            )
        ]
        defs = _get_tool_schema_defs(tools)
        self.assertEqual(len(defs), 0)

    def test_tool_with_null_parameters(self):
        """Test tool with null parameters"""
        tools = [
            Tool(
                type="function",
                function=Function(name="no_params", parameters=None),
            )
        ]
        defs = _get_tool_schema_defs(tools)
        self.assertEqual(len(defs), 0)

    def test_empty_tools_list(self):
        """Test with empty tools list"""
        defs = _get_tool_schema_defs([])
        self.assertEqual(len(defs), 0)

    def test_conflicting_defs_raises_error(self):
        """Test that conflicting defs raise ValueError"""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="tool1",
                    parameters={
                        "$defs": {
                            "ConflictingType": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            }
                        }
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="tool2",
                    parameters={
                        "$defs": {
                            "ConflictingType": {
                                "type": "object",
                                "properties": {"value": {"type": "number"}},
                            }
                        }
                    },
                ),
            ),
        ]
        with self.assertRaises(ValueError) as context:
            _get_tool_schema_defs(tools)
        self.assertIn("ConflictingType", str(context.exception))
        self.assertIn("multiple schemas", str(context.exception))

    def test_identical_defs_no_error(self):
        """Test that identical defs in different tools don't raise error"""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="tool1",
                    parameters={
                        "$defs": {
                            "SharedType": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            }
                        }
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="tool2",
                    parameters={
                        "$defs": {
                            "SharedType": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            }
                        }
                    },
                ),
            ),
        ]
        # Should not raise
        defs = _get_tool_schema_defs(tools)
        self.assertEqual(len(defs), 1)
        self.assertIn("SharedType", defs)

    def test_nested_defs(self):
        """Test tools with nested defs"""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="nested_tool",
                    parameters={
                        "$defs": {
                            "Outer": {
                                "type": "object",
                                "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                            },
                            "Inner": {"type": "object", "properties": {}},
                        }
                    },
                ),
            )
        ]
        defs = _get_tool_schema_defs(tools)
        self.assertEqual(len(defs), 2)
        self.assertIn("Outer", defs)
        self.assertIn("Inner", defs)


if __name__ == "__main__":
    unittest.main()
