"""Unit tests for Qwen25Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


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


class TestQwen25DetectorHasToolCall(CustomTestCase):
    def setUp(self):
        self.detector = Qwen25Detector()

    def test_has_tool_call_true(self):
        text = '<tool_call>\n{"name":"get_weather","arguments":{"city":"Tokyo"}}\n</tool_call>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false_plain_text(self):
        self.assertFalse(self.detector.has_tool_call("The weather is sunny."))

    def test_has_tool_call_false_tag_without_newline(self):
        # Must include the newline after the tag to match bot_token
        self.assertFalse(self.detector.has_tool_call("<tool_call>no-newline"))

    def test_has_tool_call_partial_tag(self):
        self.assertFalse(self.detector.has_tool_call("<tool_ca"))

    def test_has_tool_call_empty(self):
        self.assertFalse(self.detector.has_tool_call(""))


class TestQwen25DetectorDetectAndParse(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = Qwen25Detector()

    def _make_call(self, name, args):
        return f'<tool_call>\n{{"name":"{name}","arguments":{json.dumps(args)}}}\n</tool_call>'

    def test_single_tool_call(self):
        text = self._make_call("get_weather", {"city": "Beijing"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Beijing"})
        self.assertEqual(result.normal_text, "")

    def test_single_tool_call_with_multiple_args(self):
        text = self._make_call("get_weather", {"city": "London", "unit": "celsius"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    def test_multiple_tool_calls(self):
        text = self._make_call("get_weather", {"city": "Tokyo"}) + self._make_call(
            "search", {"query": "restaurants"}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_leading_text(self):
        text = "Sure, let me check that for you.\n" + self._make_call(
            "get_weather", {"city": "Paris"}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        # Normal text is stripped
        self.assertIn("Sure", result.normal_text)

    def test_no_tool_call(self):
        text = "The weather in Tokyo is sunny today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_malformed_json_skipped(self):
        # Invalid JSON inside the tag should be logged and skipped, not raise
        text = "<tool_call>\nnot valid json\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_unknown_tool_name_skipped_by_default(self):
        # A function not in the tools list should be silently dropped
        text = '<tool_call>\n{"name":"nonexistent","arguments":{}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_empty_text(self):
        result = self.detector.detect_and_parse("", self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "")

    def test_tool_call_index_set_correctly(self):
        # tool_index should map to position in tools list
        text = self._make_call("search", {"query": "hello"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        # "search" is at index 1 in _make_tools()
        self.assertEqual(result.calls[0].tool_index, 1)


class TestQwen25DetectorStructureInfo(CustomTestCase):
    def setUp(self):
        self.detector = Qwen25Detector()

    def test_structure_info_trigger(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertEqual(info.trigger, "<tool_call>")

    def test_structure_info_begin_contains_name(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn("<tool_call>", info.begin)

    def test_structure_info_end(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertTrue(info.end.endswith("</tool_call>"))

    def test_structure_info_different_names(self):
        info_func = self.detector.structure_info()
        info_weather = info_func("get_weather")
        info_search = info_func("search")
        self.assertIn("get_weather", info_weather.begin)
        self.assertIn("search", info_search.begin)
        self.assertNotEqual(info_weather.begin, info_search.begin)


class TestQwen25DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_streaming_complete_in_one_chunk(self):
        detector = Qwen25Detector()
        text = '<tool_call>\n{"name":"get_weather","arguments":{"city":"Seoul"}}\n</tool_call>'
        all_calls = []
        for chunk in [text]:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
        calls_with_name = [c for c in all_calls if c.name]
        self.assertEqual(len(calls_with_name), 1)
        self.assertEqual(calls_with_name[0].name, "get_weather")

    def test_streaming_normal_text_passthrough(self):
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment(
            "Hello! How can I help? ", self.tools
        )
        self.assertEqual(result.normal_text, "Hello! How can I help? ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_split_across_chunks(self):
        detector = Qwen25Detector()
        chunks = [
            "<tool_call>\n",
            '{"name":"get_weather",',
            '"arguments":{"city":"Berlin"}}',
            "\n</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
        calls_with_name = [c for c in all_calls if c.name]
        self.assertTrue(len(calls_with_name) >= 1)
        self.assertEqual(calls_with_name[0].name, "get_weather")

    def test_streaming_normal_text_before_tool(self):
        detector = Qwen25Detector()
        normal_chunks = ["Sure, ", "let me ", "check. "]
        for chunk in normal_chunks:
            r = detector.parse_streaming_increment(chunk, self.tools)
            # No tool calls yet
            self.assertEqual(len(r.calls), 0)

    def test_streaming_multiple_detectors_independent(self):
        # Each Qwen25Detector instance has its own state
        d1 = Qwen25Detector()
        d2 = Qwen25Detector()
        text = '<tool_call>\n{"name":"get_weather","arguments":{"city":"NYC"}}\n</tool_call>'
        r1 = d1.parse_streaming_increment(text, self.tools)
        r2 = d2.parse_streaming_increment(text, self.tools)
        # Both should independently parse the same call
        calls1 = [c for c in r1.calls if c.name]
        calls2 = [c for c in r2.calls if c.name]
        self.assertEqual(len(calls1), len(calls2))


if __name__ == "__main__":
    import unittest

    unittest.main()
