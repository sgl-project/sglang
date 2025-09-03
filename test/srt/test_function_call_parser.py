import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.json_detector import JSONDetector


class TestJSONDetector(unittest.TestCase):
    def setUp(self):
        # Create sample tools for testing
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
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
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
        self.detector = JSONDetector()

    def test_parse_streaming_no_json(self):
        """Test parsing text with no JSON (no tool calls)."""
        text = "This is just normal text without any tool calls."
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])
        self.assertEqual(self.detector._buffer, "")  # Buffer should be cleared

    def test_parse_streaming_complete_tool_call(self):
        """Test parsing a complete tool call in JSON format."""
        text = 'Here\'s a tool call: [{"name": "get_weather", "parameters": {"location": "New York", "unit": "celsius"}}]'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "Here's a tool call: ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            self.detector._buffer, ""
        )  # Buffer should be cleared after processing

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "New York")
        self.assertEqual(params["unit"], "celsius")

    def test_parse_streaming_text_before_tool_call(self):
        """Test parsing text that appears before a tool call."""
        text = 'This is some text before [{"name": "get_weather", "parameters": {"location": "London"}}]'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "This is some text before ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "London")

    def test_parse_streaming_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        # First chunk with opening bracket but no closing bracket
        text1 = 'Let me check the weather: [{"name": "get_weather", "parameters": {"location":'
        result1 = self.detector.parse_streaming_increment(text1, self.tools)

        self.assertEqual(result1.normal_text, "Let me check the weather: ")
        self.assertEqual(result1.calls, [])
        self.assertIn(
            '{"name": "get_weather", "parameters": {"location":',
            self.detector._buffer
        )  # Partial tool call remains in buffer

        # Second chunk completing the tool call
        text2 = '"Paris"}}]'
        result2 = self.detector.parse_streaming_increment(text2, self.tools)

        self.assertEqual(result2.normal_text, "")
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")

        # Check the parameters
        params = json.loads(result2.calls[0].parameters)
        self.assertEqual(params["location"], "Paris")
        self.assertEqual(
            self.detector._buffer, ""
        )  # Buffer should be cleared after processing

    def test_parse_streaming_json_without_text_before(self):
        """Test parsing a tool call that starts at the beginning of the text."""
        text = '[{"name": "search", "parameters": {"query": "python programming"}}]'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "python programming")

    def test_parse_streaming_text_after_tool_call(self):
        """Test parsing text that appears after a tool call."""
        # First chunk with complete tool call and some text after
        text = '[{"name": "get_weather", "parameters": {"location": "Tokyo"}}] Here\'s the forecast:'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            self.detector._buffer, " Here's the forecast:"
        )  # Text after tool call remains in buffer

        # Process the remaining text in buffer
        result2 = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result2.normal_text, " Here's the forecast:")
        self.assertEqual(result2.calls, [])
        self.assertEqual(self.detector._buffer, "")  # Buffer should be cleared

    def test_parse_streaming_multiple_tool_calls(self):
        """Test parsing multiple tool calls in sequence."""
        text = '[{"name": "get_weather", "parameters": {"location": "Berlin"}}, {"name": "search", "parameters": {"query": "restaurants"}}]'

        # First tool call
        result1 = self.detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result1.calls), 2)
        self.assertEqual(result1.calls[0].name, "get_weather")
        self.assertEqual(result1.calls[1].name, "search")
        self.assertEqual(self.detector._buffer, "")

    def test_parse_streaming_opening_bracket_only(self):
        """Test parsing text with only an opening bracket but no closing bracket."""
        text = "Let's try this: ["
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "Let's try this: ")
        self.assertEqual(result.calls, [])
        self.assertEqual(
            self.detector._buffer, "["
        )  # Opening bracket remains in buffer

    def test_parse_streaming_nested_json(self):
        """Test parsing tool calls with nested JSON in arguments."""
        # Test with complex nested JSON
        text = '[{"name": "get_weather", "parameters": {"location": "New York", "unit": "celsius", "data": [1, 2, 3]}}]'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "New York")
        self.assertEqual(params["unit"], "celsius")
        self.assertEqual(params["data"], [1, 2, 3])

    def test_parse_streaming_nested_json_dict(self):
        """Test parsing tool calls with nested dictionaries and lists."""
        # Test with nested dict and list arguments
        text = '[{"name": "search", "parameters": {"query": "test", "config": {"options": [1, 2], "nested": {"key": "value"}}}}]'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "test")
        self.assertEqual(params["config"]["options"], [1, 2])
        self.assertEqual(params["config"]["nested"]["key"], "value")

    def test_parse_streaming_multiple_tools_with_nested_json(self):
        """Test parsing multiple tool calls with nested JSON."""
        text = '[{"name": "get_weather", "parameters": {"location": "Paris", "data": [10, 20]}}, {"name": "search", "parameters": {"query": "test", "filters": ["a", "b"]}}]'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(self.detector._buffer, "")

        # Check first tool call
        params1 = json.loads(result.calls[0].parameters)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(params1["location"], "Paris")
        self.assertEqual(params1["data"], [10, 20])

        # Check second tool call
        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(result.calls[1].name, "search")
        self.assertEqual(params2["query"], "test")
        self.assertEqual(params2["filters"], ["a", "b"])

    def test_parse_streaming_partial_nested_json(self):
        """Test parsing partial tool calls with nested JSON across chunks."""
        # First chunk with nested JSON but incomplete
        text1 = 'Here\'s a call: [{"name": "get_weather", "parameters": {"location": "Tokyo", "data": [1, 2'
        result1 = self.detector.parse_streaming_increment(text1, self.tools)

        self.assertEqual(result1.normal_text, "Here's a call: ")
        self.assertEqual(result1.calls, [])
        self.assertIn(
            '{"name": "get_weather", "parameters": {"location": "Tokyo", "data": [1, 2',
            self.detector._buffer
        )

        # Second chunk completing the nested JSON
        text2 = ', 3]}}]'
        result2 = self.detector.parse_streaming_increment(text2, self.tools)

        self.assertEqual(result2.normal_text, "")
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result2.calls[0].parameters)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["data"], [1, 2, 3])

    def test_detect_and_parse_single_tool_call(self):
        """Test parsing a single tool call in a complete text."""
        text = '[{"name": "get_weather", "parameters": {"location": "Paris"}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, "")

    def test_detect_and_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls in a complete text."""
        text = '[{"name": "get_weather", "parameters": {"location": "Paris"}}, {"name": "search", "parameters": {"query": "restaurants"}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")
        self.assertEqual(result.normal_text, "")

    def test_detect_and_parse_with_text_before(self):
        """Test parsing text that has content before the tool call."""
        text = 'Here is some text before the tool call: [{"name": "get_weather", "parameters": {"location": "Paris"}}]'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Here is some text before the tool call:")
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_detect_and_parse_no_tool_calls(self):
        """Test parsing text without any tool calls."""
        text = "This is just normal text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_detect_and_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        text = '[{"name": "get_weather", "parameters": {"location": "Paris"}'  # Missing closing bracket
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_detect_and_parse_invalid_tool_name(self):
        """Test parsing JSON with invalid tool name."""
        text = '[{"name": "invalid_tool", "parameters": {"location": "Paris"}}]'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_has_tool_call(self):
        """Test detection of tool call markers."""
        self.assertTrue(self.detector.has_tool_call('[{"name": "get_weather"}]'))
        self.assertTrue(self.detector.has_tool_call('{"name": "get_weather"}'))
        self.assertFalse(self.detector.has_tool_call("No tool call here"))
        self.assertFalse(self.detector.has_tool_call(""))

    def test_single_object_format(self):
        """Test parsing single object format (not array)."""
        text = '{"name": "get_weather", "parameters": {"location": "Paris"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, "")

    def test_parameters_vs_arguments(self):
        """Test that both 'parameters' and 'arguments' fields work."""
        # Test with 'parameters'
        text1 = '[{"name": "get_weather", "parameters": {"location": "Paris"}}]'
        result1 = self.detector.detect_and_parse(text1, self.tools)
        self.assertEqual(len(result1.calls), 1)
        self.assertEqual(result1.calls[0].name, "get_weather")

        # Test with 'arguments' (should be converted to parameters internally)
        text2 = '[{"name": "get_weather", "arguments": {"location": "Paris"}}]'
        result2 = self.detector.detect_and_parse(text2, self.tools)
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")

    def test_structure_info(self):
        """Test structure info generation."""
        get_info = self.detector.structure_info()
        info = get_info("get_weather")
        
        self.assertEqual(info.begin, '{"name":"get_weather", "parameters":')
        self.assertEqual(info.end, "}")
        self.assertEqual(info.trigger, "")

    def test_supports_structural_tag(self):
        """Test that structural tag support is enabled."""
        self.assertTrue(self.detector.supports_structural_tag())

    def test_build_json_schema(self):
        """Test JSON schema generation."""
        schema = self.detector.build_json_schema(self.tools)
        self.assertIsNotNone(schema)
        self.assertIn("type", schema)
        self.assertEqual(schema["type"], "array")
        self.assertIn("items", schema)
        self.assertIn("minItems", schema)

    def test_build_ebnf(self):
        """Test EBNF generation (for backward compatibility)."""
        ebnf = self.detector.build_ebnf(self.tools)
        self.assertIsNotNone(ebnf)
        # Should contain some EBNF grammar rules
        self.assertIn("::=", ebnf)


if __name__ == "__main__":
    unittest.main()
