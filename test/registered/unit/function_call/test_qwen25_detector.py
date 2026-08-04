"""Unit tests for Qwen25Detector — no server, no model loading.

The Qwen 2.5 / Qwen 3 tool-call format wraps each call in
``<tool_call>\\n{...}\\n</tool_call>`` blocks. This module exercises:

* ``has_tool_call`` detection on positive / negative inputs.
* ``detect_and_parse`` for single, multiple, and malformed payloads.
* ``parse_streaming_increment`` including the ``</tool_call>`` partial-token
  buffer logic that strips trailing tag fragments from ``normal_text``.
* ``structure_info`` shape used downstream by structural-tag generation.
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestQwen25Detector(CustomTestCase):
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
        self.detector = Qwen25Detector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n'
            "</tool_call>"
        )
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false_plain_text(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    def test_has_tool_call_false_only_closing_tag(self):
        # Only the closing tag — without ``<tool_call>\n`` ``has_tool_call`` must be False.
        text = "some text </tool_call> trailing"
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n'
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_normal_text_before_tool_call(self):
        text = (
            'Let me check the weather. <tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Tokyo"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        # ``detect_and_parse`` strips whitespace around the prefix segment.
        self.assertEqual(result.normal_text, "Let me check the weather.")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_parallel_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n'
            "</tool_call>\n"
            '<tool_call>\n{"name": "search", "arguments": {"query": "restaurants"}}\n'
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")
        first_args = json.loads(result.calls[0].parameters)
        second_args = json.loads(result.calls[1].parameters)
        self.assertEqual(first_args["city"], "Beijing")
        self.assertEqual(second_args["query"], "restaurants")

    def test_tool_call_with_multiple_arguments(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": '
            '{"city": "London", "unit": "celsius"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    def test_malformed_json_is_skipped(self):
        # First block has invalid JSON; second is valid. Detector should warn-and-skip
        # the broken block and still return the valid call.
        text = (
            "<tool_call>\n{not valid json}\n</tool_call>\n"
            '<tool_call>\n{"name": "search", "arguments": {"query": "books"}}\n'
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn("<tool_call>", info.begin)
        self.assertIn("</tool_call>", info.end)
        self.assertEqual(info.trigger, "<tool_call>")

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call_chunked(self):
        # Feed a single call in arbitrary chunks — accumulated calls/parameters
        # must reconstruct the original payload.
        detector = Qwen25Detector()
        chunks = [
            "<tool_",
            'call>\n{"name": "get_weather",',
            ' "arguments": {"city": "Beijing"',
            "}}\n</tool_call>",
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
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment(
            "Let me check the weather. ", self.tools
        )
        self.assertEqual(result.normal_text, "Let me check the weather. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = Qwen25Detector()
        chunks = [
            "I'll look that up. ",
            '<tool_call>\n{"name": "get_weather",',
            ' "arguments": {"city": "Tokyo", "unit": "celsius"',
            "}}\n</tool_call>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "I'll look that up. ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")
        self.assertEqual(params["unit"], "celsius")

    def test_streaming_partial_bot_token_is_buffered(self):
        """When a chunk ends with a partial ``<tool_call>`` prefix the detector
        must NOT leak the partial tag as normal_text — it has to keep buffering
        until the next chunk decides whether this is a real tool call.

        This guards the common race where the model emits ``...some text<tool_``
        in one streaming chunk and ``call>\\n{...``in the next. Without correct
        buffering the user would briefly see the ``<tool_`` fragment in their
        rendered text.
        """
        detector = Qwen25Detector()
        # Chunk 1: ends with a partial bot_token. The base class falls through
        # ``_ends_with_partial_token`` and must hold the buffer instead of
        # flushing the prefix as normal_text.
        result1 = detector.parse_streaming_increment("hello <tool_", self.tools)
        self.assertEqual(result1.normal_text, "")
        self.assertEqual(len(result1.calls), 0)

        # Chunk 2: completes the tool call. The accumulated call must be parsed
        # and the function name surfaced through the streaming interface.
        result2 = detector.parse_streaming_increment(
            'call>\n{"name": "search", "arguments": {"query": "hi"}}\n</tool_call>',
            self.tools,
        )
        names = [c.name for c in result2.calls if c.name]
        self.assertIn("search", names)

    def test_streaming_partial_bot_token_does_not_leak_prefix(self):
        """Same buffering guarantee, but for a chunk that *only* contains a
        non-call prefix that happens to end with the bot_token's leading
        characters. Even though the chunk is not a real call, no fragment of
        ``<tool`` may be returned to the user yet — the detector must wait."""
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment("I think <tool", self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_arguments_emitted_after_name(self):
        """A single tool call streamed in two chunks split mid-JSON must emit
        the function name first (with empty parameters) and then stream the
        arguments JSON in subsequent calls, matching the OpenAI-style
        incremental tool-call delta protocol."""
        detector = Qwen25Detector()
        chunks = [
            '<tool_call>\n{"name": "search",',
            ' "arguments": {"query": "x"}}\n</tool_call>',
        ]
        all_calls = []
        for chunk in chunks:
            all_calls.extend(
                detector.parse_streaming_increment(chunk, self.tools).calls
            )

        # First emitted call should carry the name with empty parameters.
        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "search")
        self.assertEqual(named[0].parameters, "")

        # Subsequent emissions should carry the arguments JSON (no name).
        arg_chunks = [c.parameters for c in all_calls if c.parameters]
        full_args = "".join(arg_chunks)
        self.assertEqual(json.loads(full_args), {"query": "x"})


if __name__ == "__main__":
    import unittest

    unittest.main()
