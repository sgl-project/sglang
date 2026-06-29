"""Unit tests for DeepSeekV3Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

# Special Unicode delimiter tokens used by DeepSeek V3
_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_CALLS_END = "<｜tool▁calls▁end｜>"
_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_CALL_END = "<｜tool▁call▁end｜>"
_SEP = "<｜tool▁sep｜>"


def _make_call(name: str, args_json: str) -> str:
    return f"{_CALL_BEGIN}function{_SEP}{name}\n```json\n{args_json}\n```{_CALL_END}"


def _make_tools():
    return [
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
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]


class TestDeepSeekV3DetectorHasToolCall(CustomTestCase):
    def setUp(self):
        self.detector = DeepSeekV3Detector()

    def test_has_tool_call_true(self):
        text = f"{_CALLS_BEGIN}{_make_call('get_weather', '{\"city\": \"Tokyo\"}')}{_CALLS_END}"
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false_plain_text(self):
        self.assertFalse(self.detector.has_tool_call("The weather is sunny."))

    def test_has_tool_call_false_only_end_token(self):
        self.assertFalse(self.detector.has_tool_call(_CALLS_END))


class TestDeepSeekV3DetectorDetectAndParse(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = DeepSeekV3Detector()

    def test_single_tool_call(self):
        args = '{"city": "Tokyo"}'
        text = f"{_CALLS_BEGIN}{_make_call('get_weather', args)}{_CALLS_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Tokyo")
        self.assertEqual(result.normal_text, "")

    def test_single_tool_call_with_multiple_args(self):
        args = '{"city": "London", "unit": "celsius"}'
        text = f"{_CALLS_BEGIN}{_make_call('get_weather', args)}{_CALLS_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "London")
        self.assertEqual(params["unit"], "celsius")

    def test_multiple_tool_calls(self):
        call1 = _make_call("get_weather", '{"city": "Beijing"}')
        call2 = _make_call("search", '{"query": "restaurants"}')
        text = f"{_CALLS_BEGIN}{call1}\n{call2}{_CALLS_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_leading_text(self):
        args = '{"city": "Paris"}'
        text = f"I will look that up. {_CALLS_BEGIN}{_make_call('get_weather', args)}{_CALLS_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertIn("I will look that up.", result.normal_text)

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_search_tool_call(self):
        args = '{"query": "best coffee shops"}'
        text = f"{_CALLS_BEGIN}{_make_call('search', args)}{_CALLS_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "best coffee shops")


class TestDeepSeekV3DetectorStructureInfo(CustomTestCase):
    def setUp(self):
        self.detector = DeepSeekV3Detector()

    def test_structure_info_contains_function_name(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)

    def test_structure_info_trigger(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertEqual(info.trigger, _CALLS_BEGIN)

    def test_structure_info_end_contains_call_end(self):
        info_func = self.detector.structure_info()
        info = info_func("my_func")
        self.assertIn(_CALL_END, info.end)
        self.assertIn(_CALLS_END, info.end)


class TestDeepSeekV3DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_streaming_normal_text_before_tool(self):
        detector = DeepSeekV3Detector()
        result = detector.parse_streaming_increment(
            "Plain text with no tool call.", self.tools
        )
        self.assertEqual(result.normal_text, "Plain text with no tool call.")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_single_tool_call(self):
        detector = DeepSeekV3Detector()
        inner = _make_call("get_weather", '{"city": "Beijing"}')
        full_call = f"{_CALLS_BEGIN}{inner}{_CALLS_END}"

        # Split into 3 chunks so the streaming parser can emit the name on the
        # first match and then the argument diff on the second match:
        #   chunk 1 — header + inner call up to (not including) _CALL_END → name sent
        #   chunk 2 — _CALL_END                                            → args sent
        #   chunk 3 — _CALLS_END                                           → cleanup
        split = full_call.index(_CALL_END)
        chunks = [full_call[:split], _CALL_END, _CALLS_END]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_accumulates_arguments(self):
        detector = DeepSeekV3Detector()
        full_call = (
            f"{_CALLS_BEGIN}{_make_call('get_weather', '{\"city\": \"Shanghai\"}')}{_CALLS_END}"
        )
        all_calls = []
        # Use chunk size larger than the longest delimiter token (≥ 20 chars) so
        # the streaming buffer is not cleared before the token is fully received.
        chunk_size = 25
        for i in range(0, len(full_call), chunk_size):
            result = detector.parse_streaming_increment(
                full_call[i : i + chunk_size], self.tools
            )
            all_calls.extend(result.calls)

        named = [c for c in all_calls if c.name]
        self.assertGreater(len(named), 0)
        self.assertEqual(named[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Shanghai")


if __name__ == "__main__":
    import unittest

    unittest.main()
