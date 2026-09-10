"""Streaming unit tests for MistralDetector — no server, no model loading.

Asserts that replaying the same text through `parse_streaming_increment` in
arbitrary chunk sizes reproduces the `detect_and_parse` result, in particular
for JSON arrays holding more than one call.
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestMistralDetectorStreaming(CustomTestCase):
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
                            "days": {"type": "integer"},
                        },
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
                        "properties": {"q": {"type": "string"}},
                    },
                ),
            ),
        ]

    # ---------------- helpers ----------------

    def _stream(self, chunks):
        detector = MistralDetector()
        seq, normal, cur = [], "", None

        def absorb(result):
            nonlocal normal, cur
            normal += result.normal_text or ""
            for call in result.calls:
                if call.name:
                    cur = [call.name, ""]
                    seq.append(cur)
                if call.parameters:
                    if cur is None:
                        cur = ["", ""]
                        seq.append(cur)
                    cur[1] += call.parameters

        for chunk in chunks:
            absorb(detector.parse_streaming_increment(chunk, self.tools))
        # The base detector advances one call per invocation and real serving
        # keeps calling it per token, so drain until idle at end of stream.
        for _ in range(200):
            result = detector.parse_streaming_increment("", self.tools)
            if not result.calls and not result.normal_text:
                break
            absorb(result)

        return [tuple(x) for x in seq], normal

    def _chunkings(self, text):
        third = len(text) // 3
        return {
            "char": list(text),
            "whole": [text],
            "halves": [text[: len(text) // 2], text[len(text) // 2 :]],
            "thirds": [text[:third], text[third : 2 * third], text[2 * third :]],
        }

    def assert_stream_matches_final(self, text):
        reference = MistralDetector().detect_and_parse(text, self.tools)
        want = [(c.name, c.parameters) for c in reference.calls]
        want_normal = (reference.normal_text or "").strip()

        for label, chunks in self._chunkings(text).items():
            got, got_normal = self._stream(chunks)

            self.assertEqual(
                len(got), len(want), f"[{label}] call count differs for {text!r}"
            )
            for (g_name, g_args), (w_name, w_args) in zip(got, want):
                self.assertEqual(g_name, w_name, f"[{label}] name differs")
                self.assertEqual(
                    json.loads(g_args) if g_args else None,
                    json.loads(w_args) if w_args else None,
                    f"[{label}] arguments differ for {text!r}",
                )

            self.assertNotIn(
                "[TOOL_CALLS", got_normal, f"[{label}] marker leaked into content"
            )
            self.assertEqual(
                got_normal.strip(), want_normal, f"[{label}] normal text differs"
            )

    # ---------------- tests ----------------

    def test_single_call_no_arguments(self):
        self.assert_stream_matches_final(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}]'
        )

    def test_single_call_with_arguments(self):
        self.assert_stream_matches_final(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
        )

    def test_two_calls_empty_arguments(self):
        # Regression: the second call was emitted as normal text, because after
        # the first call the remaining buffer no longer carries the marker.
        self.assert_stream_matches_final(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}, '
            '{"name": "get_weather", "arguments": {}}]'
        )

    def test_two_calls_with_arguments(self):
        self.assert_stream_matches_final(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}, '
            '{"name": "search", "arguments": {"q": "food"}}]'
        )

    def test_three_calls(self):
        self.assert_stream_matches_final(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}, '
            '{"name": "search", "arguments": {"q": "a"}}, '
            '{"name": "get_weather", "arguments": {"days": 3}}]'
        )

    def test_empty_arguments_followed_by_populated(self):
        self.assert_stream_matches_final(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}, '
            '{"name": "search", "arguments": {"q": "x"}}]'
        )

    def test_integer_argument_keeps_type(self):
        self.assert_stream_matches_final(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"days": 7}}]'
        )

    def test_unicode_argument(self):
        self.assert_stream_matches_final(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "北京"}}]'
        )

    def test_normal_text_before_tool_call(self):
        self.assert_stream_matches_final(
            'Sure. [TOOL_CALLS] [{"name": "get_weather", "arguments": {}}]'
        )

    def test_plain_text_without_tool_call(self):
        self.assert_stream_matches_final("The weather is nice today.")


if __name__ == "__main__":
    import unittest

    unittest.main()
