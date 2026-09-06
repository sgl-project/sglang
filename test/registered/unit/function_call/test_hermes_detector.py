"""Unit tests for HermesDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestHermesDetector(CustomTestCase):
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
        self.detector = HermesDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'
            '<tool_call>{"name": "search", "arguments": {"query": "restaurants"}}</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_leading_text(self):
        text = 'I will check the weather for you. <tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, "I will check the weather for you.")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_tool_call_with_multiple_arguments(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "London", "unit": "celsius"}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    def test_malformed_json_returns_original_text(self):
        text = "<tool_call>not valid json</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertEqual(info.trigger, "<tool_call>")
        self.assertEqual(info.end, "}</tool_call>")

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        detector = HermesDetector()
        chunks = [
            "<tool_",
            'call>{"name": "get_weather",',
            ' "arguments": {"city": "Beijing"',
            "}}</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        # Verify tool name
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

        # Verify parameters
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_before_tool(self):
        detector = HermesDetector()
        result = detector.parse_streaming_increment("Hello! Let me help. ", self.tools)
        self.assertEqual(result.normal_text, "Hello! Let me help. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = HermesDetector()
        chunks = [
            "Sure, let me check. ",
            '<tool_call>{"name": "get_weather",',
            ' "arguments": {"city": "Tokyo"',
            "}}</tool_call>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "Sure, let me check. ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")

    # ==================== finish() / stream-end Tests ====================
    # These cover the finish() override added for the base-class contract:
    # "Called once when the stream ends; flush any buffered state. Detectors
    # that hold text back while waiting for a marker that can no longer
    # arrive (the stream is over) override this to release it." Before this
    # override existed, HermesDetector inherited the no-op default and
    # silently dropped whatever was still buffered.

    def test_finish_flushes_partial_marker_at_stream_end(self):
        """Generation stops (e.g. hits max_tokens) right after a chunk
        boundary makes the tail look like the start of '<tool_call>'. That
        text can never complete now that the stream is over, so finish()
        must hand it back instead of losing it."""
        detector = HermesDetector()
        chunks = ["The format looks like ", "<tool_", "call"]
        streamed = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            streamed += result.normal_text

        # Without finish(), "<tool_call" is stuck in the buffer forever.
        self.assertEqual(streamed, "The format looks like ")
        self.assertEqual(detector._buffer, "<tool_call")

        final = detector.finish(self.tools)
        self.assertEqual(final.normal_text, "<tool_call")
        self.assertEqual(len(final.calls), 0)
        self.assertEqual(
            streamed + final.normal_text, "The format looks like <tool_call"
        )
        # Buffers are drained so a reused detector instance can't leak state.
        self.assertEqual(detector._buffer, "")
        self.assertEqual(detector._normal_text_buffer, "")

    def test_finish_after_complete_tool_call_is_unchanged(self):
        """Regression: a complete, well-formed tool call must still parse
        exactly as before, and finish() afterwards must be a no-op."""
        detector = HermesDetector()
        chunks = [
            "Sure, let me check. ",
            '<tool_call>{"name": "get_weather",',
            ' "arguments": {"city": "Tokyo"',
            "}}</tool_call>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "Sure, let me check. ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(json.loads(full_params), {"city": "Tokyo"})

        final = detector.finish(self.tools)
        self.assertEqual(final.normal_text, "")
        self.assertEqual(len(final.calls), 0)

    def test_finish_drops_unterminated_tool_call_without_raising(self):
        """If generation stops mid-argument (bot_token seen, no eot_token),
        the buffered content is neither valid normal text nor a parseable
        tool call. finish() must warn and drop it, not raise or emit
        garbage as a call/normal_text."""
        detector = HermesDetector()
        chunks = ['<tool_call>{"name": "get_weather", "arguments": {"ci']
        for chunk in chunks:
            detector.parse_streaming_increment(chunk, self.tools)

        self.assertIn(detector.bot_token, detector._buffer)

        final = detector.finish(self.tools)  # must not raise
        self.assertEqual(final.normal_text, "")
        self.assertEqual(len(final.calls), 0)
        self.assertEqual(detector._buffer, "")

    def test_finish_with_no_partial_marker_is_unchanged(self):
        """Regression: a stream that ends cleanly, with nothing buffered,
        must be unaffected by the new finish() override."""
        detector = HermesDetector()
        result = detector.parse_streaming_increment(
            "The weather is nice today.", self.tools
        )
        self.assertEqual(result.normal_text, "The weather is nice today.")

        final = detector.finish(self.tools)
        self.assertEqual(final.normal_text, "")
        self.assertEqual(len(final.calls), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
