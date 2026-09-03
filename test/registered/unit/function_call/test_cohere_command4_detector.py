"""Unit tests for CohereCommand4Detector - no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


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
                        "city": {"type": "string"},
                        "unit": {"type": "string"},
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
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ),
    ]


class TestCohereCommand4Detector(CustomTestCase):
    def setUp(self):
        self.detector = CohereCommand4Detector()
        self.tools = _make_tools()

    def test_has_tool_call(self):
        self.assertTrue(self.detector.has_tool_call("<|START_ACTION|>[]"))
        self.assertFalse(self.detector.has_tool_call("plain response"))

    def test_normalize_calls_accepts_dict_and_drops_tool_call_id(self):
        normalized = self.detector._normalize_calls(
            {
                "tool_call_id": "call-0",
                "tool_name": "search",
                "parameters": {"query": "sglang"},
            }
        )

        self.assertEqual(
            normalized, [{"name": "search", "parameters": {"query": "sglang"}}]
        )

    def test_normalize_calls_filters_non_dict_items(self):
        normalized = self.detector._normalize_calls(
            [
                "ignore me",
                {"tool_name": "get_weather", "parameters": {"city": "Paris"}},
                42,
            ]
        )

        self.assertEqual(
            normalized,
            [{"name": "get_weather", "parameters": {"city": "Paris"}}],
        )

    def test_normalize_calls_rejects_non_sequence_shape(self):
        self.assertEqual(self.detector._normalize_calls("not a call"), [])

    def test_no_tool_call_returns_normal_text(self):
        result = self.detector.detect_and_parse("No tools needed.", self.tools)

        self.assertEqual(result.normal_text, "No tools needed.")
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_single_cohere_tool_call(self):
        text = (
            "I will check."
            '<|START_ACTION|>[{"tool_call_id":"0","tool_name":"get_weather",'
            '"parameters":{"city":"Paris","unit":"celsius"}}]<|END_ACTION|>'
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "I will check.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"city": "Paris", "unit": "celsius"},
        )

    def test_detect_and_parse_multiple_tool_calls(self):
        text = (
            '<|START_ACTION|>[{"tool_name":"get_weather",'
            '"parameters":{"city":"Tokyo"}},'
            '{"name":"search","parameters":{"query":"Tokyo weather"}}]'
            "<|END_ACTION|>"
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            [call.name for call in result.calls], ["get_weather", "search"]
        )
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})
        self.assertEqual(
            json.loads(result.calls[1].parameters), {"query": "Tokyo weather"}
        )

    def test_detect_and_parse_accepts_single_object_body(self):
        text = (
            '<|START_ACTION|>{"tool_name":"search",'
            '"parameters":{"query":"SGLang docs"}}<|END_ACTION|>'
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"query": "SGLang docs"}
        )

    def test_detect_and_parse_skips_unknown_tool(self):
        text = (
            '<|START_ACTION|>[{"tool_name":"unknown",'
            '"parameters":{"value":1}}]<|END_ACTION|>'
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_malformed_json_returns_surrounding_text(self):
        text = "prefix<|START_ACTION|>not-json<|END_ACTION|>"

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "prefix")
        self.assertEqual(result.calls, [])

    def test_streaming_plain_text_without_tool_call(self):
        result = self.detector.parse_streaming_increment("plain text", self.tools)

        self.assertEqual(result.normal_text, "plain text")
        self.assertEqual(result.calls, [])

    def test_streaming_buffers_partial_start_token_and_emits_complete_call(self):
        chunks = [
            "Intro <|STAR",
            'T_ACTION|>[{"tool_call_id":"0","tool_name":"get_weather",',
            '"parameters":{"city":"Berlin"}}]<|END_ACTION|>tail',
        ]

        normal_text = []
        calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            normal_text.append(result.normal_text)
            calls.extend(result.calls)

        flush = self.detector.parse_streaming_increment("", self.tools)
        normal_text.append(flush.normal_text)
        calls.extend(flush.calls)

        self.assertEqual("".join(normal_text), "Intro tail")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Berlin"})

    def test_streaming_waits_for_end_action(self):
        result = self.detector.parse_streaming_increment(
            '<|START_ACTION|>[{"tool_name":"search",'
            '"parameters":{"query":"sglang"}}',
            self.tools,
        )

        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_structure_info(self):
        info = self.detector.structure_info()("search")

        self.assertIn('"tool_name": "search"', info.begin)
        self.assertIn('"parameters": ', info.begin)
        self.assertEqual(info.end, "}]<|END_ACTION|>")
        self.assertEqual(info.trigger, "<|START_ACTION|>")
        self.assertFalse(self.detector.supports_structural_tag())
