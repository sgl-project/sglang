"""Unit tests for Xing4_0Detector - no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.xing4_0_detector import Xing4_0Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"


def _tag_call(name: str, arguments: list[tuple[str, str]]) -> str:
    pairs = "".join(
        f"<param_key>{key}</param_key><param_value>{value}</param_value>"
        for key, value in arguments
    )
    return f"{TOOL_START}{name}{pairs}{TOOL_END}"


def _json_call(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{TOOL_START}{payload}{TOOL_END}"


def _raw_call(payload: str) -> str:
    return f"{TOOL_START}{payload}{TOOL_END}"


def _tools() -> list[Tool]:
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_time",
                parameters={
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                },
            ),
        ),
    ]


def _tools_without_time() -> list[Tool]:
    return [tool for tool in _tools() if tool.function.name != "get_time"]


def _stream(text: str, tools: list[Tool], chunked: bool = False):
    """Feed text through the detector, returning (content, calls)."""
    detector = Xing4_0Detector()
    chunks = list(text) if chunked else [text]
    content, calls = [], []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        content.append(result.normal_text)
        calls.extend(result.calls)
    return "".join(content), calls


class TestXing4_0DetectorNonStream(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    def test_parser_is_registered(self):
        self.assertIs(FunctionCallParser.ToolCallParserEnum["xing4_0"], Xing4_0Detector)

    def test_tag_format_parses_arguments_and_declared_types(self):
        text = _tag_call("get_weather", [("city", "Paris"), ("count", "42")])

        result = Xing4_0Detector().detect_and_parse(text, self.tools)

        self.assertEqual([call.name for call in result.calls], ["get_weather"])
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {
                "city": "Paris",
                "count": 42,
            },
        )

    def test_declared_string_argument_is_not_coerced(self):
        text = _tag_call("get_weather", [("city", "123")])

        result = Xing4_0Detector().detect_and_parse(text, self.tools)

        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "123"})

    def test_json_format_is_supported(self):
        text = _json_call("get_time", {"timezone": "UTC"})

        result = Xing4_0Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.calls[0].name, "get_time")
        self.assertEqual(json.loads(result.calls[0].parameters), {"timezone": "UTC"})

    def test_text_without_tool_call_is_returned_unchanged(self):
        result = Xing4_0Detector().detect_and_parse("hello", self.tools)

        self.assertEqual(result.normal_text, "hello")
        self.assertEqual(result.calls, [])

    def test_unknown_tool_dropped_by_default(self):
        text = _tag_call("get_time", [("timezone", "UTC")])

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            result = Xing4_0Detector().detect_and_parse(text, _tools_without_time())

        self.assertEqual(result.calls, [])
        self.assertNotIn(TOOL_START, result.normal_text)

    def test_unknown_tool_forwarded_when_enabled(self):
        text = _json_call("get_time", {"timezone": "UTC"})

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = Xing4_0Detector().detect_and_parse(text, _tools_without_time())

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_time")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(json.loads(result.calls[0].parameters), {"timezone": "UTC"})

    def test_empty_name_is_skipped_even_when_forwarding(self):
        text = _json_call("", {})

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = Xing4_0Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])

    def test_declared_name_json_format_is_supported(self):
        text = _raw_call('get_weather{"city": "Paris", "count": 2}')

        result = Xing4_0Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Paris", "count": 2}
        )

    def test_undeclared_name_json_format_dropped_by_default(self):
        text = _raw_call('get_time{"timezone": "UTC"}')

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            result = Xing4_0Detector().detect_and_parse(text, _tools_without_time())

        self.assertEqual(result.calls, [])
        self.assertNotIn(TOOL_START, result.normal_text)

    def test_undeclared_name_json_format_forwarded_when_enabled(self):
        text = _raw_call('get_time{"timezone": "UTC"}')

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = Xing4_0Detector().detect_and_parse(text, _tools_without_time())

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_time")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(json.loads(result.calls[0].parameters), {"timezone": "UTC"})

    def test_bare_name_json_format_still_parses_without_arguments(self):
        text = _raw_call("get_time")

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = Xing4_0Detector().detect_and_parse(text, _tools_without_time())

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_time")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_invalid_name_is_skipped_and_keeps_other_calls(self):
        payloads = [
            '{"name": 123, "arguments": {}}',
            '{"name": ["x"], "arguments": {}}',
            '{"name": "   ", "arguments": {}}',
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                text = _tag_call("get_weather", [("city", "Paris")]) + _raw_call(
                    payload
                )

                with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
                    result = Xing4_0Detector().detect_and_parse(
                        text, _tools_without_time()
                    )

                self.assertEqual([call.name for call in result.calls], ["get_weather"])
                self.assertEqual(
                    json.loads(result.calls[0].parameters), {"city": "Paris"}
                )

    def test_repeated_call_gets_sequential_indices(self):
        text = _tag_call("get_weather", [("city", "Paris")]) + _tag_call(
            "get_weather", [("city", "London")]
        )

        result = Xing4_0Detector().detect_and_parse(text, self.tools)

        self.assertEqual([call.tool_index for call in result.calls], [0, 1])
        self.assertEqual(
            [json.loads(call.parameters)["city"] for call in result.calls],
            ["Paris", "London"],
        )

    def test_dropped_unknown_call_does_not_shift_known_indices(self):
        text = _tag_call("get_time", [("timezone", "UTC")]) + _tag_call(
            "get_weather", [("city", "Paris")]
        )

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            dropped = Xing4_0Detector().detect_and_parse(text, _tools_without_time())
        self.assertEqual(
            [(call.name, call.tool_index) for call in dropped.calls],
            [("get_weather", 0)],
        )

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            forwarded = Xing4_0Detector().detect_and_parse(text, _tools_without_time())
        self.assertEqual(
            [(call.name, call.tool_index) for call in forwarded.calls],
            [("get_time", 0), ("get_weather", 1)],
        )

    def test_structure_info_targets_json_format(self):
        info = Xing4_0Detector().structure_info()("get_weather")

        self.assertEqual(info.trigger, TOOL_START)
        self.assertTrue(info.begin.startswith(TOOL_START))
        self.assertIn('"name": "get_weather"', info.begin)
        self.assertTrue(info.end.endswith(TOOL_END))


class TestXing4_0DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    def test_dropped_call_does_not_leak_markup(self):
        text = "before" + _tag_call("get_time", [("timezone", "UTC")]) + "after"

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            content, calls = _stream(text, _tools_without_time())

        self.assertEqual(content, "beforeafter")
        self.assertEqual(calls, [])

    def test_dropped_call_does_not_block_later_call(self):
        text = _tag_call("get_time", [("timezone", "UTC")]) + _tag_call(
            "get_weather", [("city", "Paris")]
        )

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            content, calls = _stream(text, _tools_without_time())

        self.assertEqual(content, "")
        self.assertEqual(
            [(call.name, call.tool_index) for call in calls], [("get_weather", 0)]
        )

    def test_forwarded_unknown_call_streams_with_arguments(self):
        text = "before" + _json_call("get_time", {"timezone": "UTC"}) + "after"

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            content, calls = _stream(text, _tools_without_time())

        self.assertEqual(content, "beforeafter")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_time")
        self.assertEqual(calls[0].tool_index, 0)
        self.assertEqual(json.loads(calls[0].parameters), {"timezone": "UTC"})

    def test_forwarded_unknown_name_json_call_streams_with_arguments(self):
        text = "before" + _raw_call('get_time{"timezone": "UTC"}') + "after"

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            content, calls = _stream(text, _tools_without_time())

        self.assertEqual(content, "beforeafter")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_time")
        self.assertEqual(calls[0].tool_index, 0)
        self.assertEqual(json.loads(calls[0].parameters), {"timezone": "UTC"})

    def test_repeated_call_streams_distinct_indices(self):
        text = _tag_call("get_weather", [("city", "Paris")]) + _tag_call(
            "get_weather", [("city", "London")]
        )

        content, calls = _stream(text, self.tools)

        self.assertEqual(content, "")
        self.assertEqual([call.tool_index for call in calls], [0, 1])
        self.assertEqual(
            [json.loads(call.parameters)["city"] for call in calls],
            ["Paris", "London"],
        )

    def test_character_chunks_match_single_chunk(self):
        text = (
            "before"
            + _tag_call("get_weather", [("city", "Paris")])
            + _tag_call("get_weather", [("city", "London")])
            + "after"
        )

        plain_content, plain_calls = _stream(text, self.tools)
        chunked_content, chunked_calls = _stream(text, self.tools, chunked=True)

        self.assertEqual(chunked_content, plain_content)
        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in chunked_calls],
            [(call.tool_index, call.name, call.parameters) for call in plain_calls],
        )

    def test_partial_start_token_is_buffered(self):
        detector = Xing4_0Detector()

        first = detector.parse_streaming_increment("hello<tool_", self.tools)
        self.assertEqual(first.normal_text, "hello")

        second = detector.parse_streaming_increment(
            "call>" + _tag_call("get_weather", [("city", "Paris")])[len(TOOL_START) :],
            self.tools,
        )
        self.assertEqual([call.name for call in second.calls], ["get_weather"])


if __name__ == "__main__":
    unittest.main()
