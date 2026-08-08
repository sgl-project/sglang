"""
Unit tests for the DeepSeek V3.2 / V4 DSML tool-call detectors.

Focus: streaming preamble preservation. Assistant prose that arrives in the
same streaming delta as the DSML tool-call opener must be emitted as normal
text instead of being silently dropped. Speculative decoding makes
multi-token deltas common, so the prose/opener boundary is exercised at every
possible chunk split point. The load-bearing invariant is that the total
normal_text emitted across a stream equals the normal_text returned by the
non-streaming ``detect_and_parse`` for the same input.

DeepSeek-V4 frequently opens a tool-call section with a bare
``<｜DSML｜invoke`` (no enclosing wrapper token), so preamble preservation is
tested for every DSML marker form, not just ``bot_token``.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

INVOKE_END = "</｜DSML｜invoke>"
PREAMBLE = "Let me look up the weather in Beijing for you."


def _invoke(city):
    return (
        '<｜DSML｜invoke name="get_weather">'
        f'<｜DSML｜parameter name="city" string="true">{city}</｜DSML｜parameter>'
        + INVOKE_END
    )


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
                    },
                    "required": ["city"],
                },
            ),
        ),
    ]


def _collect_streamed_tool_calls(all_calls):
    tools = {}
    for c in all_calls:
        idx = c.tool_index
        if idx not in tools:
            tools[idx] = {"name": c.name or "", "parameters": c.parameters or ""}
        else:
            if c.name:
                tools[idx]["name"] += c.name
            if c.parameters:
                tools[idx]["parameters"] += c.parameters
    return [tools[i] for i in sorted(tools.keys())]


class _StreamingPreambleTestsMixin:
    """Shared streaming-preamble tests, run against both DSML detectors."""

    detector_class = None

    def setUp(self):
        self.tools = _make_tools()
        detector = self.detector_class()
        self.bot_token = detector.bot_token
        self.eot_token = detector.eot_token

    def _wrapped_call(self, *cities):
        return (
            self.bot_token + "".join(_invoke(city) for city in cities) + self.eot_token
        )

    def _stream(self, detector, chunks):
        normal_text = ""
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            all_calls.extend(result.calls)
        return normal_text, all_calls

    def _assert_calls(self, all_calls, *cities):
        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), len(cities))
        for call, city in zip(collected, cities):
            self.assertEqual(call["name"], "get_weather")
            self.assertEqual(json.loads(call["parameters"]), {"city": city})

    def test_prose_and_wrapped_opener_in_one_delta(self):
        """Prose + full wrapped tool call arriving in a single delta."""
        text = PREAMBLE + self._wrapped_call("Beijing")
        normal_text, calls = self._stream(self.detector_class(), [text])
        self.assertEqual(normal_text, PREAMBLE)
        self._assert_calls(calls, "Beijing")

    def test_prose_and_bare_invoke_in_one_delta(self):
        """Prose + bare ``<｜DSML｜invoke`` (no wrapper token) in one delta."""
        text = PREAMBLE + _invoke("Beijing")
        normal_text, calls = self._stream(self.detector_class(), [text])
        self.assertEqual(normal_text, PREAMBLE)
        self._assert_calls(calls, "Beijing")

    def test_prose_emitted_once_at_every_split_point(self):
        """Two-chunk streams split at every boundary: prose exactly once."""
        text = PREAMBLE + self._wrapped_call("Beijing")
        for i in range(1, len(text)):
            with self.subTest(split=i):
                chunks = [text[:i], text[i:]]
                normal_text, calls = self._stream(self.detector_class(), chunks)
                self.assertEqual(normal_text, PREAMBLE)
                self._assert_calls(calls, "Beijing")

    def test_bare_invoke_prose_at_every_split_point(self):
        """Same boundary sweep for the bare-invoke opener form."""
        text = PREAMBLE + _invoke("Beijing")
        for i in range(1, len(text)):
            with self.subTest(split=i):
                chunks = [text[:i], text[i:]]
                normal_text, calls = self._stream(self.detector_class(), chunks)
                self.assertEqual(normal_text, PREAMBLE)
                self._assert_calls(calls, "Beijing")

    def test_fixed_size_chunkings(self):
        """Small fixed-size chunks split the opener across >2 deltas."""
        text = PREAMBLE + self._wrapped_call("Beijing")
        for size in (1, 3, 7):
            with self.subTest(chunk_size=size):
                chunks = [text[i : i + size] for i in range(0, len(text), size)]
                normal_text, calls = self._stream(self.detector_class(), chunks)
                self.assertEqual(normal_text, PREAMBLE)
                self._assert_calls(calls, "Beijing")

    def test_partial_tag_prefix_held_back(self):
        """A delta ending in a partial tag prefix emits the preceding prose
        immediately and holds back only the prefix."""
        full_call = self._wrapped_call("Beijing")
        for prefix in ("<", "<｜"):
            with self.subTest(prefix=prefix):
                detector = self.detector_class()
                first = detector.parse_streaming_increment(
                    PREAMBLE + prefix, self.tools
                )
                self.assertEqual(first.normal_text, PREAMBLE)
                second = detector.parse_streaming_increment(
                    full_call[len(prefix) :], self.tools
                )
                normal_text = first.normal_text + second.normal_text
                self.assertEqual(normal_text, PREAMBLE)
                self._assert_calls(first.calls + second.calls, "Beijing")

    def test_literal_angle_bracket_without_tool_call(self):
        """A literal ``<`` in plain prose is never swallowed."""
        chunks = ["The check 3 <", " 5 held, done."]
        normal_text, calls = self._stream(self.detector_class(), chunks)
        self.assertEqual(normal_text, "The check 3 < 5 held, done.")
        self.assertEqual(calls, [])

    def test_literal_angle_bracket_before_tool_call(self):
        """A literal ``<`` inside the prose of a tool-call delta stays in the
        prose; the split happens at the DSML marker, not the first ``<``."""
        prose = "Since 3 < 5 holds, calling the tool. "
        text = prose + self._wrapped_call("Beijing")
        normal_text, calls = self._stream(self.detector_class(), [text])
        self.assertEqual(normal_text, prose)
        self._assert_calls(calls, "Beijing")

    def test_multiple_invokes_after_prose(self):
        """Prose + two invokes in one wrapped section: prose exactly once,
        both calls parsed, regardless of chunking."""
        text = PREAMBLE + self._wrapped_call("Beijing", "Tokyo")
        chunkings = [
            [text],
            [text[i : i + 8] for i in range(0, len(text), 8)],
        ]
        for chunks in chunkings:
            with self.subTest(num_chunks=len(chunks)):
                normal_text, calls = self._stream(self.detector_class(), chunks)
                self.assertEqual(normal_text, PREAMBLE)
                self._assert_calls(calls, "Beijing", "Tokyo")

    def test_streaming_matches_detect_and_parse(self):
        """Invariant: total streamed normal_text equals the non-streaming
        ``detect_and_parse`` normal_text for the same input, at every split."""
        text = PREAMBLE + self._wrapped_call("Beijing")
        reference = self.detector_class().detect_and_parse(text, self.tools)
        splits = [[text]] + [[text[:i], text[i:]] for i in range(1, len(text))]
        for chunks in splits:
            with self.subTest(split=len(chunks[0]) if len(chunks) > 1 else "one-shot"):
                normal_text, calls = self._stream(self.detector_class(), chunks)
                self.assertEqual(normal_text, reference.normal_text)
                collected = _collect_streamed_tool_calls(calls)
                self.assertEqual(
                    [c["name"] for c in collected],
                    [c.name for c in reference.calls],
                )
                self.assertEqual(
                    [json.loads(c["parameters"]) for c in collected],
                    [json.loads(c.parameters) for c in reference.calls],
                )


class TestDeepSeekV4StreamingPreamble(_StreamingPreambleTestsMixin, CustomTestCase):
    detector_class = DeepSeekV4Detector


class TestDeepSeekV32StreamingPreamble(_StreamingPreambleTestsMixin, CustomTestCase):
    detector_class = DeepSeekV32Detector


if __name__ == "__main__":
    unittest.main()
