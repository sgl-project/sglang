"""Unit tests for GigaChat3Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestGigaChat3Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="manage_user_memory",
                    description="Create, update, or delete a user memory entry.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "content": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "action": {
                                "type": "string",
                                "enum": ["create", "update", "delete"],
                                "default": "create",
                            },
                            "id": {
                                "anyOf": [
                                    {"type": "string", "format": "uuid"},
                                    {"type": "null"},
                                ],
                                "default": None,
                            },
                        },
                    },
                ),
            ),
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
        ]
        self.detector = GigaChat3Detector()

    def test_has_tool_call_detects_both_markers(self):
        # Guards both marker forms the regex ORs together; the parser exercises
        # each via detect_and_parse but never the bare predicate.
        self.assertTrue(self.detector.has_tool_call("function call<|role_sep|>\n{}"))
        self.assertTrue(self.detector.has_tool_call("<|function_call|>{}"))
        self.assertFalse(self.detector.has_tool_call("No tool call here"))

    def test_detect_and_parse_no_tool_call(self):
        """Test parsing text without tool calls."""
        text = "How can I help you today?"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(len(result.calls), 0)

    def test_detect_and_parse_simple_tool_call(self):
        """Test parsing a simple tool call without content."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences"}}'
        result = self.detector.detect_and_parse(text, self.tools)

        # No content before tool call
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "manage_user_memory")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["action"], "create")
        self.assertEqual(params["id"], "preferences")

    def test_detect_and_parse_parameterless_tool_call(self):
        """Test parsing a tool call with empty arguments."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {}}'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "manage_user_memory")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params, {})

    def test_detect_and_parse_complex_tool_call(self):
        """Test parsing a tool call with nested objects."""
        text = """<|message_sep|>

function call<|role_sep|>
{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences", "content": {"short_answers": true, "hate_emojis": true, "english_ui": false, "russian_math_explanations": true}}}"""

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "manage_user_memory")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["action"], "create")
        self.assertEqual(params["id"], "preferences")
        self.assertIsInstance(params["content"], dict)
        self.assertEqual(params["content"]["short_answers"], True)
        self.assertEqual(params["content"]["hate_emojis"], True)

    def test_detect_and_parse_with_content_and_eos(self):
        """Test parsing tool call with content and EOS token."""
        text = 'I\'ll remember that.<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {"action": "create", "id": "test"}}</s>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "I'll remember that.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "manage_user_memory")

    def test_detect_and_parse_invalid_json(self):
        """Test parsing with invalid JSON in function call."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {invalid json}}'
        result = self.detector.detect_and_parse(text, self.tools)

        # Should return the full text as content when JSON parsing fails
        self.assertIn("function call", result.normal_text)
        self.assertEqual(len(result.calls), 0)

    def test_detect_and_parse_missing_name(self):
        """Test parsing with missing function name."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"arguments": {"action": "create"}}'
        result = self.detector.detect_and_parse(text, self.tools)

        # Should not extract tool call if name is missing
        self.assertEqual(len(result.calls), 0)

    def test_detect_and_parse_missing_arguments(self):
        """Test parsing with missing arguments field."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory"}'
        result = self.detector.detect_and_parse(text, self.tools)

        # Should not extract tool call if arguments is missing
        self.assertEqual(len(result.calls), 0)

    def test_detect_and_parse_arguments_not_dict(self):
        """Test parsing with arguments that is not a dict."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": "string_args"}'
        result = self.detector.detect_and_parse(text, self.tools)

        # Should not extract tool call if arguments is not a dict
        self.assertEqual(len(result.calls), 0)

    def test_streaming_no_tool_call(self):
        """Test streaming text without tool calls."""
        chunks = ["How ", "can ", "I ", "help ", "you?"]

        accumulated_text = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text

        self.assertEqual(accumulated_text, "How can I help you?")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_simple_tool_call(self):
        """Test streaming a simple tool call."""
        chunks = [
            "<|message_sep|>\n\n",
            "function call",
            "<|role_sep|>\n",
            '{"name": "manage_user_memory", ',
            '"arguments": {"action": "create"',
            ', "id": "preferences"}}',
        ]

        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "manage_user_memory")

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["action"], "create")
        self.assertEqual(params["id"], "preferences")

    def test_streaming_with_content_before(self):
        """Test streaming with content before tool call."""
        chunks = [
            "I'll ",
            "help ",
            "you.",
            "<|message_sep|>\n\n",
            "function call",
            "<|role_sep|>\n",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Tokyo"}}',
        ]

        accumulated_text = ""
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(accumulated_text, "I'll help you.")
        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "Tokyo")

    def test_streaming_complex_arguments(self):
        """Test streaming with complex nested arguments."""
        chunks = [
            "<|message_sep|>\n\n",
            "functi",
            "on call<|role_sep|>\n",
            '{"name": "manage_user_memory", "arguments": ',
            '{"action": "create", "id": "prefs", ',
            '"content": {"likes": ["short", "clear"], ',
            '"dislikes": ["emojis", "verbose"]}',
            "}}",
        ]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "manage_user_memory")

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["action"], "create")
        self.assertEqual(params["content"]["likes"], ["short", "clear"])
        self.assertEqual(params["content"]["dislikes"], ["emojis", "verbose"])

    def test_streaming_with_eos_token(self):
        """Test streaming with EOS token at the end."""
        chunks = [
            "<|message_sep|>\n\n",
            "function c",
            "all<|role_sep|>\n",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            "</s>",
        ]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "Paris")

    def test_streaming_incomplete_json(self):
        """Test streaming with incomplete JSON (no closing brace)."""
        chunks = [
            "<|message_sep|>\n\n",
            "fun",
            "ction call<|role_sep|>\n",
            '{"name": "get_weather", ',
            '"arguments": {"city": "London"',
            # Missing closing braces
        ]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Should have name but incomplete parameters
        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")
        self.assertTrue(tool_calls_by_index[0]["parameters"].startswith('{"city":'))

    def test_streaming_large_steps(self):
        """Test streaming with large chunks that complete in fewer steps."""
        chunks = [
            "I'll remember that.",
            "<|message_sep|>\n\nfuncti",
            "on call<|role_sep|>\n",
            '{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences", "content": {"short_answers": true, "hate_emojis": true, ',
            '"english_ui": false, "russian_math_explanations": true}',
            "}}",
        ]

        accumulated_text = ""
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(accumulated_text, "I'll remember that.")
        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "manage_user_memory")

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["action"], "create")
        self.assertEqual(params["content"]["short_answers"], True)
        self.assertEqual(params["content"]["russian_math_explanations"], True)

    def test_streaming_very_small_chunks(self):
        """Test streaming with very small chunks (character by character)."""
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'

        # Split into very small chunks (every 5 characters)
        chunk_size = 5
        chunked_text = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]
        chunks = [
            "<|message_sep|>\n\n",
            "func",
            "tion call",
            "<|role_sep|>\n",
            *chunked_text,
        ]
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "NYC")

    def test_detect_and_parse_function_call_marker_simple_tool_call(self):
        """Test parsing a simple <|function_call|> tool call (GigaChat3.1-style)."""
        text = '<|function_call|>{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences"}}'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "manage_user_memory")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["action"], "create")
        self.assertEqual(params["id"], "preferences")

    def test_streaming_function_call_marker_simple_tool_call(self):
        """Test streaming parsing of the <|function_call|> marker form."""
        chunks = [
            "I'll help you.",
            "<|function_call|>",
            '{"name": "manage_user_memory", "arguments": ',
            '{"action": "create", "id": "prefs"}}',
        ]

        accumulated_text = ""
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(accumulated_text, "I'll help you.")
        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "manage_user_memory")

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["action"], "create")
        self.assertEqual(params["id"], "prefs")


if __name__ == "__main__":
    import unittest

    unittest.main()
