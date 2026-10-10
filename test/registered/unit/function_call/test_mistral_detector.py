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

    # ==================== Streaming Regression Tests ====================

    CHUNK_SIZES = (None, 16, 8, 4, 1)

    def _stream(self, text, chunk_size=None):
        """Feed text through a fresh detector in fixed-size chunks.

        Returns (normal_text, [(name, arguments_json)]) with the streamed items
        merged back per tool_index, so a stream can be compared with
        detect_and_parse and across chunk sizes. chunk_size=None sends everything
        in one increment, which is what a request with a large stream_interval
        (or several accepted speculative tokens) produces.
        """
        detector = MistralDetector()
        results = [
            detector.parse_streaming_increment(
                text[i : i + (chunk_size or len(text))], self.tools
            )
            for i in range(0, len(text), chunk_size or len(text))
        ]
        results.append(detector.finish(self.tools))

        normal_text = ""
        merged = {}
        order = []
        for result in results:
            normal_text += result.normal_text or ""
            for call in result.calls or []:
                if call.tool_index not in merged:
                    merged[call.tool_index] = {"name": None, "arguments": ""}
                    order.append(call.tool_index)
                if call.name:
                    merged[call.tool_index]["name"] = call.name
                if call.parameters:
                    merged[call.tool_index]["arguments"] += call.parameters
        return normal_text, [(merged[i]["name"], merged[i]["arguments"]) for i in order]

    def test_streaming_text_and_compact_call_in_one_chunk(self):
        """A chunk holding both preamble and a complete compact call keeps the call.

        The detector used to take a single action per increment: it emitted the
        preamble and left the rest of the buffer untouched, so the tool call was
        silently dropped whenever one chunk carried both.
        """
        text = 'Sure.\n[TOOL_CALLS]get_weather[ARGS]{"city": "Beijing"}'
        normal_text, calls = self._stream(text)
        self.assertEqual(calls, [("get_weather", '{"city": "Beijing"}')])
        self.assertEqual(normal_text, "Sure.\n")

    def test_streaming_two_compact_calls_in_one_chunk(self):
        """Two compact calls in one increment both reach the client."""
        text = (
            '[TOOL_CALLS]get_weather[ARGS]{"city": "Beijing"}'
            '[TOOL_CALLS]search[ARGS]{"query": "restaurants"}'
        )
        normal_text, calls = self._stream(text)
        self.assertEqual(
            calls,
            [
                ("get_weather", '{"city": "Beijing"}'),
                ("search", '{"query": "restaurants"}'),
            ],
        )
        self.assertEqual(normal_text, "")

    def test_streaming_compact_call_is_chunk_size_invariant(self):
        """Text following a compact call survives every chunk boundary.

        Trailing content used to be dropped or kept depending on where the chunk
        boundary fell, so the same response yielded different assistant content
        at different stream intervals.
        """
        text = 'Sure.\n[TOOL_CALLS]get_weather[ARGS]{"city": "Beijing"}\nIt is sunny.'
        outputs = {size: self._stream(text, size) for size in self.CHUNK_SIZES}
        for size, (normal_text, calls) in outputs.items():
            with self.subTest(chunk_size=size):
                self.assertEqual(calls, [("get_weather", '{"city": "Beijing"}')])
                self.assertIn("It is sunny.", normal_text)
        self.assertEqual(len({normal_text for normal_text, _ in outputs.values()}), 1)

    def test_streaming_unknown_compact_tool_leaks_no_markup(self):
        """An undefined tool name is dropped, never echoed as assistant content.

        detect_and_parse skips unknown tools, but the streaming path used to flush
        the buffer as normal text, handing raw `[TOOL_CALLS]...[ARGS]{...}` markup
        to the client.
        """
        text = 'Let me check.\n[TOOL_CALLS]get_wether[ARGS]{"city": "Beijing"}\nDone.'
        for size in self.CHUNK_SIZES:
            with self.subTest(chunk_size=size):
                normal_text, calls = self._stream(text, size)
                self.assertEqual(calls, [])
                self.assertNotIn("[TOOL_CALLS", normal_text)
                self.assertNotIn("[ARGS", normal_text)
                self.assertNotIn("get_wether", normal_text)
                self.assertIn("Let me check.", normal_text)
                self.assertIn("Done.", normal_text)

    def test_detect_and_parse_drops_truncated_compact_call(self):
        """A call cut off by the token limit is markup, not assistant content."""
        text = '[TOOL_CALLS]get_weather[ARGS]{"city": "Beijing"}[TOOL_CALLS]get_wea'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "")

    def test_streaming_json_array_two_calls(self):
        """Both array entries stream out, and no part of the array leaks as text.

        The base JSON parser sends a tool name and its arguments in separate
        increments and relies on a following increment to carry the rest, so an
        array completing in the last chunk lost its arguments. The continuation of
        an array (`, {...}]`) also carries no `[TOOL_CALLS` marker, and used to be
        flushed to the client as plain text along with the dropped second call.
        """
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}, '
            '{"name": "search", "arguments": {"query": "restaurants"}}]'
        )
        for size in self.CHUNK_SIZES:
            with self.subTest(chunk_size=size):
                normal_text, calls = self._stream(text, size)
                self.assertEqual(normal_text, "")
                self.assertEqual([name for name, _ in calls], ["get_weather", "search"])
                self.assertEqual(json.loads(calls[0][1])["city"], "Beijing")
                self.assertEqual(json.loads(calls[1][1])["query"], "restaurants")

    def test_streaming_json_array_after_preamble_in_one_chunk(self):
        """Preamble before a JSON array is delivered, not swallowed by the parser."""
        text = 'Sure.\n[TOOL_CALLS] [{"name": "search", "arguments": {"query": "sf"}}]'
        normal_text, calls = self._stream(text)
        self.assertEqual(normal_text, "Sure.\n")
        self.assertEqual(calls, [("search", '{"query": "sf"}')])


if __name__ == "__main__":
    import unittest

    unittest.main()
