"""Unit tests for MistralDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestMistralDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
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
        self.detector = MistralDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_json_array_format(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
        )
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_compact_format(self):
        text = '[TOOL_CALLS]get_weather[ARGS]{"city": "Beijing"}'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== JSON Array Format Tests ====================

    def test_json_array_single_tool_call(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_json_array_multiple_tool_calls(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "search", "arguments": {"query": "restaurants"}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_json_array_with_leading_text(self):
        text = 'I will check. [TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "I will check.")

    # ==================== Compact Format Tests ====================

    def test_compact_format_single_tool_call(self):
        text = '[TOOL_CALLS]get_weather[ARGS]{"city": "Beijing"}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_compact_format_with_leading_text(self):
        text = 'Let me help. [TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo"}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Let me help.")

    # ==================== No Tool Call Tests ====================

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    # ==================== Edge Cases ====================

    def test_tool_call_with_nested_json(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing", "options": {"detailed": true}}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["options"]["detailed"], True)

    def test_json_array_with_invalid_json(self):
        text = "[TOOL_CALLS] [not valid json]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "")

    # ==================== Internal Methods Tests ====================

    def test_extract_json_array(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
        )
        result = self.detector._extract_json_array(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "get_weather")

    def test_extract_json_array_nested_brackets(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"tags": ["a", "b"]}}]'
        )
        result = self.detector._extract_json_array(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(parsed[0]["arguments"]["tags"], ["a", "b"])

    def test_extract_json_array_no_marker(self):
        text = "no tool calls here"
        result = self.detector._extract_json_array(text)
        self.assertIsNone(result)

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn("[TOOL_CALLS]", info.trigger)
        self.assertEqual(info.end, "}]")

    # ==================== Streaming Tests ====================

    def test_streaming_compact_format(self):
        detector = MistralDetector()
        chunks = [
            "[TOOL_",
            "CALLS]get_weather",
            '[ARGS]{"city": "Beijing"}',
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_before_tool(self):
        detector = MistralDetector()
        result = detector.parse_streaming_increment("Let me check. ", self.tools)
        self.assertEqual(result.normal_text, "Let me check. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = MistralDetector()
        chunks = [
            "Sure! ",
            "[TOOL_CALLS]get_weather",
            '[ARGS]{"city": "Tokyo"}',
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "Sure! ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")


class TestMistralDetectorNestedContent(CustomTestCase):
    def setUp(self):
        """Set up test tools and detector for Mistral format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="make_next_step_decision",
                    description="Test function for decision making",
                    parameters={
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "description": "The next step to take",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the next step",
                            },
                        },
                        "required": ["decision", "content"],
                    },
                ),
            ),
        ]
        self.detector = MistralDetector()

    def test_detect_and_parse_with_nested_brackets_in_content(self):
        """Test parsing Mistral format with nested brackets in JSON content.

        This test case specifically addresses the issue where the regex pattern
        was incorrectly truncating JSON when it contained nested brackets like [City Name].
        """
        # This is the exact problematic text from the original test failure
        test_text = '[TOOL_CALLS] [{"name":"make_next_step_decision", "arguments":{"decision":"","content":"```\\nTOOL: Access a weather API or service\\nOBSERVATION: Retrieve the current weather data for the top 5 populated cities in the US\\nANSWER: The weather in the top 5 populated cities in the US is as follows: [City Name] - [Weather Conditions] - [Temperature]\\n```"}}]'

        result = self.detector.detect_and_parse(test_text, self.tools)

        # Verify that the parsing was successful
        self.assertEqual(len(result.calls), 1, "Should detect exactly one tool call")

        call = result.calls[0]
        self.assertEqual(
            call.name,
            "make_next_step_decision",
            "Should detect the correct function name",
        )

        # Verify that the parameters are valid JSON and contain the expected content
        params = json.loads(call.parameters)
        self.assertEqual(
            params["decision"], "", "Decision parameter should be empty string"
        )

        # The content should contain the full text including the nested brackets [City Name]
        expected_content = "```\nTOOL: Access a weather API or service\nOBSERVATION: Retrieve the current weather data for the top 5 populated cities in the US\nANSWER: The weather in the top 5 populated cities in the US is as follows: [City Name] - [Weather Conditions] - [Temperature]\n```"
        self.assertEqual(
            params["content"],
            expected_content,
            "Content should include nested brackets without truncation",
        )

        # Verify that normal text is empty (since the entire input is a tool call)
        self.assertEqual(
            result.normal_text, "", "Normal text should be empty for pure tool call"
        )

    def test_detect_and_parse_no_tool_calls(self):
        """Test parsing text without any tool calls."""
        test_text = "This is just normal text without any tool calls."

        result = self.detector.detect_and_parse(test_text, self.tools)

        self.assertEqual(len(result.calls), 0, "Should detect no tool calls")
        self.assertEqual(
            result.normal_text,
            test_text,
            "Should return the original text as normal text",
        )

    def test_detect_and_parse_with_text_before_tool_call(self):
        """Test parsing text that has content before the tool call."""
        test_text = 'Here is some text before the tool call: [TOOL_CALLS] [{"name":"make_next_step_decision", "arguments":{"decision":"ANSWER", "content":"The answer is 42"}}]'

        result = self.detector.detect_and_parse(test_text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Here is some text before the tool call:")

        call = result.calls[0]
        self.assertEqual(call.name, "make_next_step_decision")

        params = json.loads(call.parameters)
        self.assertEqual(params["decision"], "ANSWER")
        self.assertEqual(params["content"], "The answer is 42")

    def test_detect_and_parse_compact_args_format(self):
        """Test parsing compact format: [TOOL_CALLS]name[ARGS]{...}."""
        test_text = '[TOOL_CALLS]make_next_step_decision[ARGS]{"decision":"TOOL", "content":"Use weather API"}'

        result = self.detector.detect_and_parse(test_text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "make_next_step_decision")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["decision"], "TOOL")
        self.assertEqual(params["content"], "Use weather API")

    def test_streaming_compact_args_format_emits_tool_calls(self):
        """Test streaming chunks for compact format produce tool_calls items."""
        chunks = [
            "[TOOL_CALLS]make_next_step_decision[ARGS]",
            '{"decision":"TOOL", ',
            '"content":"Use weather API"}',
        ]

        emitted = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                emitted.extend(result.calls)

        # Expect two items: name chunk + full args chunk
        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[0].name, "make_next_step_decision")
        self.assertEqual(emitted[0].parameters, "")
        self.assertIsNone(emitted[1].name)
        params = json.loads(emitted[1].parameters)
        self.assertEqual(params["decision"], "TOOL")
        self.assertEqual(params["content"], "Use weather API")

if __name__ == "__main__":
    import unittest

    unittest.main()
