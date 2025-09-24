# ...existing code...
"""
Tests for JSON schema constraint functionality used by JsonArrayParser
"""

import json
import unittest

import jsonschema

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.utils import get_json_schema_constraint, validate_tool_definitions


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

        schema = get_json_schema_constraint(tools_with_defs, "required")

        self.assertIsNotNone(schema)
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertIn("$defs", schema)
        self.assertIn("NestedType", schema["$defs"])

    def test_tools_with_conflicting_defs(self):
        """Test schema generation with conflicting $defs"""
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
            get_json_schema_constraint(tools_with_conflicting_defs, "required")

        self.assertIn("multiple schemas", str(context.exception))

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

    def test_json_schema_vs_ebnf_constraint_generation(self):
        """Test direct comparison between JSON schema and EBNF constraint generation"""

        # Test with specific tool choice
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="get_weather")
        )

        # Generate JSON schema constraint
        json_schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNotNone(json_schema)
        jsonschema.Draft202012Validator.check_schema(json_schema)

        # Generate EBNF constraint using FunctionCallParser
        parser = FunctionCallParser(
            self.tools, "llama3"
        )  # Use a parser that supports EBNF
        ebnf_constraint = parser.get_ebnf(tool_choice)

        # Verify JSON schema constraint
        self.assertEqual(json_schema["type"], "array")
        self.assertEqual(json_schema["minItems"], 1)
        self.assertEqual(json_schema["maxItems"], 1)

        # Verify EBNF constraint
        self.assertIsNotNone(ebnf_constraint)
        self.assertIsInstance(ebnf_constraint, str)
        self.assertIn("get_weather", ebnf_constraint)

        # Test with required tool choice
        required_json_schema = get_json_schema_constraint(self.tools, "required")

        self.assertIsNotNone(required_json_schema)
        jsonschema.Draft202012Validator.check_schema(required_json_schema)

        required_ebnf_constraint = parser.get_ebnf("required")

        # Verify required JSON schema constraint
        self.assertEqual(required_json_schema["type"], "array")
        self.assertEqual(required_json_schema["minItems"], 1)
        self.assertIn("anyOf", required_json_schema["items"])

        # Verify required EBNF constraint
        self.assertIsNotNone(required_ebnf_constraint)
        self.assertIsInstance(required_ebnf_constraint, str)

        # Both should contain references to the available tools
        tool_names = [tool.function.name for tool in self.tools]
        for tool_name in tool_names:
            self.assertIn(tool_name, required_ebnf_constraint)

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
            validate_tool_definitions(tools_with_conflicting_defs)

        self.assertIn("Tool definition 'ConflictingType' has multiple schemas", str(context.exception))
        self.assertIn("which is not supported", str(context.exception))

    def test_http_error_handling_integration(self):
        """Test that conflicting tool definitions are caught in validation and return 400 error"""
        from unittest.mock import Mock
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, Tool, Function
        
        # Create a mock tokenizer manager
        mock_tokenizer_manager = Mock()
        mock_tokenizer_manager.model_config = Mock(is_multimodal=False)
        mock_tokenizer_manager.server_args = Mock(enable_cache_report=False)
        
        # Create a mock template manager
        mock_template_manager = Mock()
        mock_template_manager.chat_template_name = None
        mock_template_manager.jinja_template_content_format = "string"
        
        # Create serving chat instance
        serving_chat = OpenAIServingChat(mock_tokenizer_manager, mock_template_manager)
        
        # Create tools with conflicting definitions
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
        
        # Create a request with conflicting tools
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            tools=tools_with_conflicting_defs,
            tool_choice="required"
        )
        
        # Test that the validation catches the conflicting definitions
        error_msg = serving_chat._validate_request(request)
        self.assertIsNotNone(error_msg)
        self.assertIn("Tool definition 'ConflictingType' has multiple schemas", error_msg)
        self.assertIn("which is not supported", error_msg)


if __name__ == "__main__":
    unittest.main()
