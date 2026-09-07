"""Unit tests for Apertus2509Detector - no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestApertus2509Detector(CustomTestCase):
    @staticmethod
    def _collect_streamed_tool_calls(calls):
        """Accumulate streaming ToolCallItems (name + arg-JSON fragments) by tool_index."""
        grouped = {}
        for call in calls:
            entry = grouped.setdefault(call.tool_index, {"name": "", "parameters": ""})
            if call.name:
                entry["name"] += call.name
            if call.parameters:
                entry["parameters"] += call.parameters
        return [grouped[i] for i in sorted(grouped)]

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="now",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]
        self.detector = Apertus2509Detector()

    # ==================== detect_and_parse Tests ====================

    def test_nonstream_parses_multiple_calls_and_preserves_normal_text(self):
        text = (
            "before"
            + '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}, {"now": {}}]'
            + "<|tools_suffix|>"
            + "after"
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "beforeafter")
        self.assertEqual([call.tool_index for call in result.calls], [0, 1])
        self.assertEqual([call.name for call in result.calls], ["get_weather", "now"])
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})
        self.assertEqual(json.loads(result.calls[1].parameters), {})

    def test_no_tool_call(self):
        text = "The weather is nice today."

        self.assertFalse(self.detector.has_tool_call(text))

        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_incomplete_block_falls_back_to_plaintext(self):
        text = '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}]'

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_has_tool_call_true_but_detect_and_parse_requires_suffix(self):
        prefix_only = "<|tools_prefix|>["

        self.assertTrue(self.detector.has_tool_call(prefix_only))

        result = self.detector.detect_and_parse(prefix_only, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, prefix_only)

    def test_malformed_and_null_args_items_are_skipped_or_defaulted(self):
        text = (
            '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}, {}, {"now": null}]'
            + "<|tools_suffix|>"
        )

        # Non-streaming: _parse_apertus_call_list / _apertus_obj_to_call.
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual([call.name for call in result.calls], ["get_weather", "now"])
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})
        self.assertEqual(json.loads(result.calls[1].parameters), {})

        # Streaming: same malformed (`{}`) and null-args (`{"now": null}`) items.
        streaming_result = Apertus2509Detector().parse_streaming_increment(
            text, self.tools
        )
        collected = self._collect_streamed_tool_calls(streaming_result.calls)
        self.assertEqual([c["name"] for c in collected], ["get_weather", "now"])
        self.assertEqual(json.loads(collected[0]["parameters"]), {"city": "Paris"})
        self.assertEqual(json.loads(collected[1]["parameters"]), {})

    def test_unknown_tool_dropped_by_default_and_forwarded_with_override(self):
        text = '<|tools_prefix|>[{"missing_tool": {"value": 1}}]<|tools_suffix|>'

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            result = self.detector.detect_and_parse(text, self.tools)
            self.assertEqual(result.calls, [])

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = self.detector.detect_and_parse(text, self.tools)
            self.assertEqual(len(result.calls), 1)
            self.assertEqual(result.calls[0].name, "missing_tool")
            self.assertEqual(json.loads(result.calls[0].parameters), {"value": 1})

    # ==================== Streaming Tests ====================

    def test_streaming_name_arrives_before_params(self):
        text = '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}]<|tools_suffix|>'

        result = self.detector.parse_streaming_increment(text, self.tools)

        name_indices = [i for i, c in enumerate(result.calls) if c.name]
        param_indices = [i for i, c in enumerate(result.calls) if c.parameters]
        self.assertTrue(name_indices, "expected a name delta")
        self.assertTrue(param_indices, "expected a params delta")
        self.assertLess(min(name_indices), min(param_indices))
        self.assertEqual(result.calls[0].parameters, "")
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "Paris"})

    def test_streaming_leading_text_and_full_marker_arrive_in_one_chunk(self):
        text = (
            "hello "
            + '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}]'
            + "<|tools_suffix|>"
        )

        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "hello ")
        collected = self._collect_streamed_tool_calls(result.calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_weather")
        self.assertEqual(json.loads(collected[0]["parameters"]), {"city": "Paris"})

    def test_streaming_invalid_json_with_suffix_present_is_flushed_as_text(self):
        text = "<|tools_prefix|>[invalid json <|tools_suffix|>"

        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_streaming_second_block_found_after_flushing_first_block(self):
        block1 = '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}]<|tools_suffix|>'
        block2 = '<|tools_prefix|>[{"now": {}}]<|tools_suffix|>'

        # Sub-case: the second block's full marker is already present in the
        # leftover buffer right after the first block is flushed.
        detector_a = Apertus2509Detector()
        r1 = detector_a.parse_streaming_increment(
            block1 + "middle" + block2, self.tools
        )
        self.assertEqual(r1.normal_text, "middle")
        collected1 = self._collect_streamed_tool_calls(r1.calls)
        self.assertEqual(collected1[0]["name"], "get_weather")

        r2 = detector_a.parse_streaming_increment("", self.tools)
        self.assertEqual(r2.normal_text, "")
        collected2 = self._collect_streamed_tool_calls(r2.calls)
        self.assertEqual(collected2[0]["name"], "now")

        # Sub-case: only a partial marker for the second block is in the
        # leftover buffer, split across the chunk boundary.
        detector_b = Apertus2509Detector()
        r1b = detector_b.parse_streaming_increment(
            block1 + "middle" + "<|tools_", self.tools
        )
        self.assertEqual(r1b.normal_text, "middle")
        collected1b = self._collect_streamed_tool_calls(r1b.calls)
        self.assertEqual(collected1b[0]["name"], "get_weather")

        r2b = detector_b.parse_streaming_increment(
            'prefix|>[{"now": {}}]<|tools_suffix|>', self.tools
        )
        self.assertEqual(r2b.normal_text, "")
        collected2b = self._collect_streamed_tool_calls(r2b.calls)
        self.assertEqual(collected2b[0]["name"], "now")

    def test_streaming_whitespace_between_json_and_suffix_is_skipped(self):
        text = '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}]  <|tools_suffix|>'

        result = self.detector.parse_streaming_increment(text, self.tools)

        collected = self._collect_streamed_tool_calls(result.calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_weather")
        self.assertEqual(json.loads(collected[0]["parameters"]), {"city": "Paris"})

    def test_streaming_partial_bot_token_buffered_across_chunks(self):
        r1 = self.detector.parse_streaming_increment("hi <|tools_", self.tools)
        self.assertEqual(r1.normal_text, "hi ")
        self.assertEqual(r1.calls, [])

        r2 = self.detector.parse_streaming_increment(
            'prefix|>[{"get_weather": {"city": "Paris"}}]<|tools_suffix|>', self.tools
        )
        self.assertEqual(r2.normal_text, "")
        collected = self._collect_streamed_tool_calls(r2.calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_weather")
        self.assertEqual(json.loads(collected[0]["parameters"]), {"city": "Paris"})

    def test_streaming_unknown_tool_dropped_by_default_and_forwarded_with_override(
        self,
    ):
        text = '<|tools_prefix|>[{"missing_tool": {"value": 1}}]<|tools_suffix|>'

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            result = Apertus2509Detector().parse_streaming_increment(text, self.tools)
            self.assertEqual(result.calls, [])

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = Apertus2509Detector().parse_streaming_increment(text, self.tools)
            collected = self._collect_streamed_tool_calls(result.calls)
            self.assertEqual(len(collected), 1)
            self.assertEqual(collected[0]["name"], "missing_tool")
            self.assertEqual(json.loads(collected[0]["parameters"]), {"value": 1})

    def test_streaming_char_by_char_matches_nonstream_result(self):
        text = (
            "before"
            + '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}, {"now": {}}]'
            + "<|tools_suffix|>"
            + "after"
        )
        streaming_detector = Apertus2509Detector()
        normal_parts = []
        calls = []
        for ch in text:
            result = streaming_detector.parse_streaming_increment(ch, self.tools)
            normal_parts.append(result.normal_text)
            calls.extend(result.calls)

        expected = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual("".join(normal_parts), expected.normal_text)
        collected = self._collect_streamed_tool_calls(calls)
        self.assertEqual(
            [c["name"] for c in collected], [call.name for call in expected.calls]
        )
        self.assertEqual(
            [json.loads(c["parameters"]) for c in collected],
            [json.loads(call.parameters) for call in expected.calls],
        )

    # ==================== structure_info Tests ====================

    def test_structure_info_matches_the_markers_the_parser_scans_for(self):
        info = self.detector.structure_info()("get_weather")

        self.assertEqual(info.trigger, "<|tools_prefix|>")
        self.assertTrue(self.detector.bot.startswith(info.trigger))
        self.assertEqual(info.begin, '<|tools_prefix|>[{"get_weather": ')
        self.assertEqual(info.end, "}]<|tools_suffix|>")

        # The begin/end markers must themselves form a block detect_and_parse accepts.
        text = info.begin + '{"city": "Paris"}' + info.end
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})


if __name__ == "__main__":
    unittest.main()
