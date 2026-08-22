"""Unit tests for PythonicDetector — no server, no model loading."""

import json
import warnings

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestPythonicDetector(CustomTestCase):
    def setUp(self):
        # Create sample tools for testing
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
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
        self.detector = PythonicDetector()

    def test_has_tool_call_detects_marker(self):
        # Guards the pythonic ``[func(kw=v)]`` regex predicate, which
        # detect_and_parse does not exercise in isolation.
        self.assertTrue(self.detector.has_tool_call('[get_weather(location="Tokyo")]'))
        self.assertFalse(self.detector.has_tool_call("plain text only"))

    def test_invalid_escape_sequence_still_parses(self):
        """An invalid Python escape (e.g. "\\d+") must not drop the tool call.

        CPython keeps the backslash and only warns; if the warning were
        promoted to an error the whole call would fall out as normal text."""
        text = '[search(query="\\d+")]'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "\\d+")
        self.assertEqual(result.normal_text, "")
        self.assertFalse(
            any(isinstance(w.message, SyntaxWarning) for w in caught),
            [str(w.message) for w in caught],
        )

    def test_parse_streaming_no_brackets(self):
        """Test parsing text with no brackets (no tool calls)."""
        text = "This is just normal text without any tool calls."
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])
        self.assertEqual(self.detector._buffer, "")  # Buffer should be cleared

    def test_parse_streaming_complete_tool_call(self):
        """Test parsing a complete tool call."""
        text = "Here's a tool call: [get_weather(location='New York', unit='celsius')]"
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

    def test_parse_streaming_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        # First chunk with opening bracket but no closing bracket
        text1 = "Let me check the weather: [get_weather(location="
        result1 = self.detector.parse_streaming_increment(text1, self.tools)

        self.assertEqual(result1.normal_text, "Let me check the weather: ")
        self.assertEqual(result1.calls, [])
        self.assertEqual(
            self.detector._buffer, "[get_weather(location="
        )  # Partial tool call remains in buffer

        # Second chunk completing the tool call
        text2 = "'Paris')]"
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

    def test_parse_streaming_bracket_without_text_before(self):
        """Test parsing a tool call that starts at the beginning of the text."""
        text = "[search(query='python programming')]"
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
        text = "[get_weather(location='Tokyo')] Here's the forecast:"
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
        text = "[get_weather(location='Berlin')] and [search(query='restaurants')]"

        # First tool call
        result1 = self.detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result1.calls), 1)
        self.assertEqual(result1.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, " and [search(query='restaurants')]")

        # Second tool call
        result2 = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result2.normal_text, " and ")
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "search")
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

    def test_parse_streaming_nested_brackets(self):
        """Test parsing tool calls with nested brackets in arguments."""
        # Test with list argument containing nested brackets
        text = "[get_weather(location='New York', unit='celsius', data=[1, 2, 3])]"
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

    def test_parse_streaming_nested_brackets_dict(self):
        """Test parsing tool calls with nested dictionaries and lists."""
        # Test with nested dict and list arguments
        text = "[search(query='test', config={'options': [1, 2], 'nested': {'key': 'value'}})]"
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

    def test_parse_streaming_multiple_tools_with_nested_brackets(self):
        """Test parsing multiple tool calls with nested brackets."""
        text = "[get_weather(location='Paris', data=[10, 20]), search(query='test', filters=['a', 'b'])]"
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

    def test_parse_streaming_partial_nested_brackets(self):
        """Test parsing partial tool calls with nested brackets across chunks."""
        # First chunk with nested brackets but incomplete
        text1 = "Here's a call: [get_weather(location='Tokyo', data=[1, 2"
        result1 = self.detector.parse_streaming_increment(text1, self.tools)

        self.assertEqual(result1.normal_text, "Here's a call: ")
        self.assertEqual(result1.calls, [])
        self.assertEqual(
            self.detector._buffer, "[get_weather(location='Tokyo', data=[1, 2"
        )

        # Second chunk completing the nested brackets
        text2 = ", 3])]"
        result2 = self.detector.parse_streaming_increment(text2, self.tools)

        self.assertEqual(result2.normal_text, "")
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result2.calls[0].parameters)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["data"], [1, 2, 3])

    def test_parse_streaming_with_python_start_and_end_token(self):
        """Test parsing a message that starts with <|python_start|> and <|python_end|> across chunks."""
        chunks = [
            "Here's a call: ",
            "<|python_",
            "start|>[get_weather(location=",
            "'Tokyo', data=[1, 2",
            ", 3])]<|python_end|>",
        ]

        normal_text = ""
        call_name = ""
        parameters = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                normal_text += result.normal_text
            if result.calls:
                call_name += result.calls[0].name
                parameters += result.calls[0].parameters

        self.assertEqual(normal_text, "Here's a call: ")
        self.assertEqual(call_name, "get_weather")
        self.assertEqual(self.detector._buffer, "")
        self.assertEqual(
            result.normal_text, "", "Final result should have no normal text"
        )

        # Check the parameters
        params = json.loads(parameters)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["data"], [1, 2, 3])

        chunks = [
            "Here's a call: <|python_start|>[get_weather(location='Tokyo', data=[1, 2, 3])]<|python_end|>"
        ]

        normal_text = ""
        call_name = ""
        parameters = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                normal_text += result.normal_text
            if result.calls:
                call_name += result.calls[0].name
                parameters += result.calls[0].parameters

        self.assertEqual(normal_text, "Here's a call: ")
        self.assertEqual(call_name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(parameters)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["data"], [1, 2, 3])

    def test_detect_and_parse_with_python_start_and_end_token(self):
        """Test parsing a message that starts with <|python_start|> and contains a valid tool call."""
        text = "User wants to get the weather in Mars. <|python_start|>[get_weather(location='Mars', unit='celsius')]<|python_end|> In this way we will get the weather in Mars."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            result.normal_text,
            "User wants to get the weather in Mars.  In this way we will get the weather in Mars.",
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Mars")
        self.assertEqual(params["unit"], "celsius")

    def test_non_finite_argument_never_emits_invalid_json(self):
        """A 1e999 literal overflows to inf and json.dumps rendered it as
        Infinity — parameters no JSON parser accepts, delivered as a
        successful call. The call is skipped instead."""
        text = "[get_weather(location='Tokyo', unit=1e999)]"
        result = self.detector.detect_and_parse(text, self.tools)

        for call in result.calls:
            json.loads(call.parameters)
        self.assertEqual(result.calls, [])

    def test_unconvertible_argument_skips_only_that_call(self):
        """A bytes argument is an ast.Constant, so it passed value
        extraction and only failed later inside json.dumps, escaping to the
        block-level handler and dropping every parseable sibling call."""
        text = "[get_weather(location='Tokyo'), search(query=b'raw')]"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertEqual(json.loads(result.calls[0].parameters), {"location": "Tokyo"})


if __name__ == "__main__":
    import unittest

    unittest.main()
