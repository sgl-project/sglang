"""
Tests for JSON schema constraint functionality used by JsonArrayParser
"""

import unittest

import jsonschema

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.utils import (
    _get_tool_schema_defs,
    get_json_schema_constraint,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


class TestJsonSchemaConstraint(unittest.TestCase):
    """Test JSON schema constraint generation for tool choices"""

    def setUp(self):
        """Set up test tools"""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to get weather for",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]

    def test_required_tool_choice_schema(self):
        """Test schema generation for tool_choice='required'"""
        schema = get_json_schema_constraint(self.tools, "required")

        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["minItems"], 1)
        self.assertIn("items", schema)
        self.assertIn("anyOf", schema["items"])

        # Should have schemas for both tools
        self.assertEqual(len(schema["items"]["anyOf"]), 2)

        # Check that each tool schema is present
        tool_names = [
            item["properties"]["name"]["enum"][0] for item in schema["items"]["anyOf"]
        ]
        self.assertIn("get_weather", tool_names)
        self.assertIn("search", tool_names)

    def test_specific_tool_choice_schema(self):
        """Test schema generation for specific tool choice"""
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="get_weather")
        )
        schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["minItems"], 1)
        self.assertEqual(schema["maxItems"], 1)

        # Should only have schema for the specific tool
        item_schema = schema["items"]
        self.assertEqual(item_schema["properties"]["name"]["enum"], ["get_weather"])
        self.assertIn("parameters", item_schema["properties"])

    def test_specific_tool_choice_dict_schema(self):
        """Test schema generation for specific tool choice as ToolChoice object"""
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="search")
        )
        schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["minItems"], 1)
        self.assertEqual(schema["maxItems"], 1)

        # Should only have schema for the specific tool
        item_schema = schema["items"]
        self.assertEqual(item_schema["properties"]["name"]["enum"], ["search"])
        self.assertIn("parameters", item_schema["properties"])

    def test_nonexistent_tool_choice(self):
        """Test schema generation for nonexistent tool"""
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="nonexistent")
        )
        schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNone(schema)

    def test_nonexistent_tool_choice_dict(self):
        """Test schema generation for nonexistent tool as dict"""
        tool_choice = {"type": "function", "function": {"name": "nonexistent"}}
        schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNone(schema)

    def test_auto_tool_choice_schema(self):
        """Test schema generation for tool_choice='auto'"""
        schema = get_json_schema_constraint(self.tools, "auto")

        self.assertIsNone(schema)

    def test_none_tool_choice_schema(self):
        """Test schema generation for tool_choice=None"""
        schema = get_json_schema_constraint(self.tools, None)

        self.assertIsNone(schema)

    def test_tools_with_defs(self):
        """Test schema generation with tools that have $defs"""
        tools_with_defs = [
            Tool(
                type="function",
                function=Function(
                    name="complex_tool",
                    description="Tool with complex schema",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "nested": {"$ref": "#/$defs/NestedType"},
                                },
                            },
                        },
                        "$defs": {
                            "NestedType": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_defs)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")

        schema = get_json_schema_constraint(tools_with_defs, "required")

        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertIn("$defs", schema)
        self.assertIn("NestedType", schema["$defs"])

    def test_tools_without_parameters(self):
        """Test schema generation with tools that have no parameters"""
        tools_without_params = [
            Tool(
                type="function",
                function=Function(
                    name="simple_tool",
                    description="Tool without parameters",
                    parameters=None,
                ),
            ),
        ]

        schema = get_json_schema_constraint(tools_without_params, "required")

        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        item_schema = schema["items"]["anyOf"][0]
        self.assertEqual(
            item_schema["properties"]["parameters"],
            {"type": "object", "properties": {}},
        )

    def test_conflicting_defs_raises_valueerror(self):
        """Test that conflicting tool definitions raise ValueError with proper message"""
        tools_with_conflicting_defs = [
            Tool(
                type="function",
                function=Function(
                    name="tool1",
                    description="Tool 1",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "$defs": {
                            "ConflictingType": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            },
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="tool2",
                    description="Tool 2",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "$defs": {
                            "ConflictingType": {
                                "type": "object",
                                "properties": {"value": {"type": "number"}},
                            },
                        },
                    },
                ),
            ),
        ]

        with self.assertRaises(ValueError) as context:
            _get_tool_schema_defs(tools_with_conflicting_defs)

        self.assertIn(
            "Tool definition 'ConflictingType' has multiple schemas",
            str(context.exception),
        )
        self.assertIn("which is not supported", str(context.exception))

    def test_tools_with_empty_defs(self):
        """Test tools with empty $defs objects"""
        tools_with_empty_defs = [
            Tool(
                type="function",
                function=Function(
                    name="empty_defs_tool",
                    description="Tool with empty $defs",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {"type": "string"},
                        },
                        "required": ["data"],
                        "$defs": {},
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_empty_defs)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")

        schema = get_json_schema_constraint(tools_with_empty_defs, "required")
        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        # Should not have $defs section when empty
        self.assertNotIn("$defs", schema)

    def test_tools_with_identical_defs(self):
        """Test different tools with same $defs names but identical schemas (should not raise exception)"""
        tools_with_identical_defs = [
            Tool(
                type="function",
                function=Function(
                    name="weather_tool",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"$ref": "#/$defs/Location"},
                        },
                        "required": ["location"],
                        "$defs": {
                            "Location": {
                                "type": "object",
                                "properties": {
                                    "lat": {"type": "number"},
                                    "lon": {"type": "number"},
                                },
                                "required": ["lat", "lon"],
                            },
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="address_tool",
                    description="Get address information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "address": {"$ref": "#/$defs/Location"},
                        },
                        "required": ["address"],
                        "$defs": {
                            "Location": {
                                "type": "object",
                                "properties": {
                                    "lat": {"type": "number"},
                                    "lon": {"type": "number"},
                                },
                                "required": ["lat", "lon"],
                            },
                        },
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_identical_defs)
        except ValueError as e:
            self.fail(
                f"Should not raise ValueError for identical schemas, but got: {e}"
            )

        # Also test that schema generation works
        schema = get_json_schema_constraint(tools_with_identical_defs, "required")
        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        # Verify both tools are present
        tool_names = [
            item["properties"]["name"]["enum"][0] for item in schema["items"]["anyOf"]
        ]
        self.assertIn("weather_tool", tool_names)
        self.assertIn("address_tool", tool_names)

        # Should have $defs with Location
        self.assertIn("$defs", schema)
        self.assertIn("Location", schema["$defs"])

    def test_tools_with_nested_defs(self):
        """Test tools with nested $defs"""
        tools_with_nested_defs = [
            Tool(
                type="function",
                function=Function(
                    name="complex_tool",
                    description="Tool with nested $defs",
                    parameters={
                        "type": "object",
                        "properties": {
                            "user": {"$ref": "#/$defs/User"},
                            "settings": {"$ref": "#/$defs/Settings"},
                        },
                        "required": ["user"],
                        "$defs": {
                            "User": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "profile": {"$ref": "#/$defs/Profile"},
                                },
                                "required": ["id"],
                            },
                            "Profile": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string", "format": "email"},
                                },
                                "required": ["name"],
                            },
                            "Settings": {
                                "type": "object",
                                "properties": {
                                    "theme": {
                                        "type": "string",
                                        "enum": ["light", "dark"],
                                    },
                                    "notifications": {"type": "boolean"},
                                },
                            },
                        },
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_nested_defs)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")

        schema = get_json_schema_constraint(tools_with_nested_defs, "required")
        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        # Verify all $defs are properly included
        self.assertIn("$defs", schema)
        self.assertIn("User", schema["$defs"])
        self.assertIn("Profile", schema["$defs"])
        self.assertIn("Settings", schema["$defs"])

    def test_mixed_tools_with_and_without_defs(self):
        """Test mixed tools with and without $defs"""
        mixed_tools = [
            Tool(
                type="function",
                function=Function(
                    name="simple_tool",
                    description="Simple tool without $defs",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="complex_tool",
                    description="Complex tool with $defs",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {"$ref": "#/$defs/DataType"},
                        },
                        "required": ["data"],
                        "$defs": {
                            "DataType": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                    "metadata": {"type": "object"},
                                },
                                "required": ["value"],
                            },
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="another_simple_tool",
                    description="Another simple tool",
                    parameters={
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                        },
                        "required": ["id"],
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(mixed_tools)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")

        schema = get_json_schema_constraint(mixed_tools, "required")
        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        # Should have $defs from the complex tool
        self.assertIn("$defs", schema)
        self.assertIn("DataType", schema["$defs"])

        # Should have all three tools
        tool_names = [
            item["properties"]["name"]["enum"][0] for item in schema["items"]["anyOf"]
        ]
        self.assertEqual(len(tool_names), 3)
        self.assertIn("simple_tool", tool_names)
        self.assertIn("complex_tool", tool_names)
        self.assertIn("another_simple_tool", tool_names)

    def test_tools_with_defs_but_no_refs(self):
        """Test tools with $defs but no $ref usage"""
        tools_with_unused_defs = [
            Tool(
                type="function",
                function=Function(
                    name="unused_defs_tool",
                    description="Tool with $defs but no $ref usage",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {"type": "string"},
                        },
                        "required": ["data"],
                        "$defs": {
                            "UnusedType": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_unused_defs)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")

        schema = get_json_schema_constraint(tools_with_unused_defs, "required")
        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        # Should still include $defs even if not referenced
        self.assertIn("$defs", schema)
        self.assertIn("UnusedType", schema["$defs"])


if __name__ == "__main__":
    unittest.main()
