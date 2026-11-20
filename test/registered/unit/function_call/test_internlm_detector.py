"""Unit tests for InternlmDetector streaming — no server, no model loading.

Covers the three chunking shapes required for the streaming tool-call release:
one whole text in a single chunk, two calls in one chunk, and one character
per chunk. For every split width the concatenated streaming output must match
the one-shot parse.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

BOT_TOKEN = "<|action_start|> <|plugin|>"
EOT_TOKEN = "<|action_end|>"


def _call(name: str, **parameters) -> str:
    payload = json.dumps({"name": name, "parameters": parameters}, ensure_ascii=False)
    return f"{BOT_TOKEN}\n{payload}{EOT_TOKEN}"


def _chunks(text: str, width: int) -> list:
    return [text[index : index + width] for index in range(0, len(text), width)]


class TestInternlmDetectorStreaming(CustomTestCase):
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
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
        ]

    def _stream(self, text: str, width: int):
        """Feed ``text`` in fixed-width chunks and return the accumulated result."""
        detector = InternlmDetector()
        normal_text = ""
        calls = []
        for chunk in _chunks(text, width):
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            calls.extend(result.calls)
        return normal_text, calls, detector

    def _assert_matches_one_shot(self, text: str):
        expected = InternlmDetector().detect_and_parse(text, self.tools)
        expected_calls = [(call.name, call.parameters) for call in expected.calls]

        for width in range(1, len(text) + 1):
            with self.subTest(width=width):
                normal_text, calls, detector = self._stream(text, width)

                self.assertEqual(normal_text.strip(), expected.normal_text.strip())
                self.assertNotIn(BOT_TOKEN, normal_text)
                self.assertNotIn(EOT_TOKEN, normal_text)
                self.assertEqual(
                    [(call.name, call.parameters) for call in calls], expected_calls
                )
                # Streaming reports call order, one index per call.
                self.assertEqual(
                    [call.tool_index for call in calls], list(range(len(calls)))
                )
                self.assertEqual(
                    detector.streamed_args_for_tool,
                    [call.parameters for call in calls],
                )
                self.assertEqual(
                    [entry["name"] for entry in detector.prev_tool_call_arr],
                    [call.name for call in calls],
                )

    def test_single_call_matches_one_shot_at_every_chunk_width(self):
        self._assert_matches_one_shot(_call("get_weather", city="SF", unit="celsius"))

    def test_two_calls_match_one_shot_at_every_chunk_width(self):
        text = (
            _call("get_weather", city="SF", unit="celsius")
            + "\n"
            + _call("get_time", city="SF")
        )

        self._assert_matches_one_shot(text)

    def test_preamble_before_two_calls_matches_one_shot_at_every_chunk_width(self):
        text = (
            "Let me check the weather and the time.\n"
            + _call("get_weather", city="SF", unit="celsius")
            + "\n"
            + _call("get_time", city="SF")
        )

        self._assert_matches_one_shot(text)

    def test_two_calls_in_one_chunk_are_emitted_together(self):
        text = (
            _call("get_weather", city="SF", unit="celsius")
            + "\n"
            + _call("get_time", city="SF")
        )

        result = InternlmDetector().parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text.strip(), "")
        self.assertEqual(
            [(call.tool_index, call.name) for call in result.calls],
            [(0, "get_weather"), (1, "get_time")],
        )
        self.assertEqual(
            [json.loads(call.parameters) for call in result.calls],
            [{"city": "SF", "unit": "celsius"}, {"city": "SF"}],
        )

    def test_two_calls_split_into_separate_chunks_keep_sequential_indices(self):
        detector = InternlmDetector()

        first = detector.parse_streaming_increment(
            _call("get_weather", city="SF"), self.tools
        )
        second = detector.parse_streaming_increment(
            _call("get_time", city="SF"), self.tools
        )

        self.assertEqual(
            [(c.tool_index, c.name) for c in first.calls], [(0, "get_weather")]
        )
        self.assertEqual(
            [(c.tool_index, c.name) for c in second.calls], [(1, "get_time")]
        )
        self.assertEqual(
            detector.streamed_args_for_tool, ['{"city": "SF"}', '{"city": "SF"}']
        )

    def test_one_character_per_chunk_streams_complete_calls(self):
        text = (
            _call("get_weather", city="SF", unit="celsius")
            + "\n"
            + _call("get_time", city="Tokyo")
        )

        normal_text, calls, _ = self._stream(text, 1)

        self.assertEqual(normal_text.strip(), "")
        self.assertEqual(
            [
                (call.tool_index, call.name, json.loads(call.parameters))
                for call in calls
            ],
            [
                (0, "get_weather", {"city": "SF", "unit": "celsius"}),
                (1, "get_time", {"city": "Tokyo"}),
            ],
        )

    def test_partial_start_marker_after_call_is_not_dropped(self):
        detector = InternlmDetector()
        text = _call("get_weather", city="SF") + "Checking again <|action_st"

        normal_text = ""
        calls = []
        for chunk in _chunks(text, 5):
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            calls.extend(result.calls)

        self.assertEqual(len(calls), 1)
        self.assertEqual(normal_text, "Checking again ")
        self.assertNotIn("<|action_st", normal_text)

    def test_finish_flushes_buffered_prose_without_leaking_markers(self):
        detector = InternlmDetector()

        streamed = detector.parse_streaming_increment(
            "Checking <|action_st", self.tools
        )
        flushed = detector.finish(self.tools)

        self.assertEqual(streamed.normal_text + flushed.normal_text, "Checking ")
        self.assertEqual(streamed.calls + flushed.calls, [])

    def test_unknown_tool_is_skipped_in_streaming(self):
        text = _call("delete_everything", city="SF") + _call("get_weather", city="SF")

        normal_text, calls, detector = self._stream(text, 7)

        self.assertEqual(normal_text, "")
        self.assertEqual(
            [(call.tool_index, call.name) for call in calls], [(0, "get_weather")]
        )
        self.assertEqual(detector.streamed_args_for_tool, ['{"city": "SF"}'])

    def test_function_call_parser_streams_two_calls_and_flushes_at_end(self):
        text = (
            "Checking.\n"
            + _call("get_weather", city="SF", unit="celsius")
            + "\n"
            + _call("get_time", city="SF")
        )
        parser = FunctionCallParser(self.tools, "interns1")

        normal_text = ""
        calls = []
        for chunk in _chunks(text, 3):
            chunk_normal, chunk_calls = parser.parse_stream_chunk(chunk)
            normal_text += chunk_normal
            calls.extend(chunk_calls)
        end_normal, end_calls = parser.parse_stream_end()
        normal_text += end_normal
        calls.extend(end_calls)

        self.assertEqual(normal_text.strip(), "Checking.")
        self.assertEqual(
            [
                (call.tool_index, call.name, json.loads(call.parameters))
                for call in calls
            ],
            [
                (0, "get_weather", {"city": "SF", "unit": "celsius"}),
                (1, "get_time", {"city": "SF"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
