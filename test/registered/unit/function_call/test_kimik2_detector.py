"""Unit tests for KimiK2Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _wrap_section(*tool_calls: str) -> str:
    """Helper: wrap a sequence of tool_call segments in the section markers."""
    inner = "".join(tool_calls)
    return f"<|tool_calls_section_begin|>{inner}<|tool_calls_section_end|>"


def _make_call(func_name: str, idx: int, args_json: str) -> str:
    """Helper: build a single <|tool_call_begin|>...<|tool_call_end|> segment."""
    return (
        f"<|tool_call_begin|>functions.{func_name}:{idx}"
        f"<|tool_call_argument_begin|>{args_json}"
        f"<|tool_call_end|>"
    )


class TestKimiK2Detector(CustomTestCase):
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
        self.detector = KimiK2Detector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = _wrap_section(_make_call("get_weather", 0, '{"city": "Beijing"}'))
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_detect_and_parse_no_tool_call(self):
        """Plain text should pass through as normal_text with no calls."""
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_detect_and_parse_single_tool_call(self):
        """Standard single tool call in canonical KimiK2 format."""
        text = _wrap_section(_make_call("get_weather", 0, '{"city": "Beijing"}'))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        # No content before the section -> normal_text empty
        self.assertEqual(result.normal_text, "")

    def test_detect_and_parse_multiple_tool_calls(self):
        """Multiple tool calls within the same section."""
        text = _wrap_section(
            _make_call("get_weather", 0, '{"city": "Beijing"}'),
            _make_call("search", 1, '{"query": "restaurants"}'),
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        names = {c.name for c in result.calls}
        self.assertEqual(names, {"get_weather", "search"})

    def test_detect_and_parse_text_before_tool_call(self):
        """Normal text preceding the section should appear in normal_text."""
        prefix = "Sure, let me check that for you. "
        text = prefix + _wrap_section(_make_call("get_weather", 0, '{"city": "Tokyo"}'))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, prefix)

    def test_detect_and_parse_tool_call_with_multiple_arguments(self):
        text = _wrap_section(
            _make_call("get_weather", 0, '{"city": "London", "unit": "celsius"}')
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        """structure_info should embed function name and use the canonical markers."""
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("functions.get_weather:0", info.begin)
        self.assertIn("<|tool_calls_section_begin|>", info.begin)
        self.assertIn("<|tool_call_argument_begin|>", info.begin)
        self.assertEqual(info.trigger, "<|tool_calls_section_begin|>")
        self.assertIn("<|tool_call_end|>", info.end)
        self.assertIn("<|tool_calls_section_end|>", info.end)

    def test_get_structural_tag_name(self):
        self.assertEqual(self.detector.get_structural_tag_name(), "kimi")

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        """Streaming returns incremental partial calls; the accumulated stream
        across all chunks should reconstruct the full tool call."""
        detector = KimiK2Detector()
        full = _wrap_section(_make_call("get_weather", 0, '{"city": "Beijing"}'))
        # Split at boundaries that exercise the buffering logic.
        # The exact chunk boundaries should not affect the accumulated result.
        cut1 = full.index("<|tool_call_begin|>")
        cut2 = full.index("<|tool_call_argument_begin|>")
        cut3 = full.index("<|tool_call_end|>")
        chunks = [full[:cut1], full[cut1:cut2], full[cut2:cut3], full[cut3:]]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        # Verify a tool name was emitted at some point in the stream.
        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")

        # Concatenating all parameter increments should yield the full JSON.
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_only(self):
        """Streaming pure normal text should yield it back with no tool calls."""
        detector = KimiK2Detector()
        result = detector.parse_streaming_increment("Hello! Let me help. ", self.tools)
        self.assertEqual(result.normal_text, "Hello! Let me help. ")
        self.assertEqual(len(result.calls), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
