"""Unit tests for MinimaxM2Detector — no server, no model loading.

Focused on streaming/non-streaming equivalence: feeding the same text through
`parse_streaming_increment` in arbitrary chunk sizes must yield the same calls
as `detect_and_parse`, and must never leak the XML markers into the content.
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

TOOL_CALL_START = "<minimax:tool_call>"
TOOL_CALL_END = "</minimax:tool_call>"


class TestMinimaxM2DetectorStreaming(CustomTestCase):
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
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="noop",
                    description="Takes no arguments",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

    # ---------------- helpers ----------------

    def _stream(self, text, chunks):
        detector = MinimaxM2Detector()
        acc, order, normal = {}, [], ""
        for chunk in list(chunks) + ["", ""]:  # trailing flushes, as serving does
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text or ""
            for call in result.calls:
                if call.tool_index not in acc:
                    acc[call.tool_index] = ["", ""]
                    order.append(call.tool_index)
                if call.name:
                    acc[call.tool_index][0] = call.name
                if call.parameters:
                    acc[call.tool_index][1] += call.parameters
        return [(acc[i][0], acc[i][1]) for i in order], normal

    def _chunkings(self, text):
        """Char-by-char (real detokenized streaming), whole, and split in half."""
        return {
            "char": list(text),
            "whole": [text],
            "halves": [text[: len(text) // 2], text[len(text) // 2 :]],
        }

    def assert_stream_matches_final(self, text):
        reference = MinimaxM2Detector().detect_and_parse(text, self.tools)
        want = [(c.name, c.parameters) for c in reference.calls]
        want_normal = (reference.normal_text or "").strip()

        for label, chunks in self._chunkings(text).items():
            got, got_normal = self._stream(text, chunks)

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

            # Markers must never reach user-visible content.
            for marker in (TOOL_CALL_START, TOOL_CALL_END, "<invoke", "<parameter"):
                self.assertNotIn(
                    marker, got_normal, f"[{label}] {marker!r} leaked into content"
                )
            self.assertEqual(
                got_normal.strip(),
                want_normal,
                f"[{label}] normal text differs for {text!r}",
            )

    # ---------------- tests ----------------

    def test_call_with_no_arguments(self):
        # Regression: streamed args stayed empty instead of "{}", and the marker
        # was emitted one character at a time as normal text.
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather"></invoke>{TOOL_CALL_END}'
        )

    def test_call_with_no_arguments_empty_schema(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="noop"></invoke>{TOOL_CALL_END}'
        )

    def test_single_string_argument(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather">'
            f'<parameter name="city">Beijing</parameter></invoke>{TOOL_CALL_END}'
        )

    def test_multiple_arguments(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather">'
            f'<parameter name="city">Beijing</parameter>'
            f'<parameter name="days">3</parameter></invoke>{TOOL_CALL_END}'
        )

    def test_integer_argument_keeps_type(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather">'
            f'<parameter name="days">7</parameter></invoke>{TOOL_CALL_END}'
        )

    def test_unicode_argument(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather">'
            f'<parameter name="city">北京</parameter></invoke>{TOOL_CALL_END}'
        )

    def test_two_calls_in_one_block(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather"></invoke>'
            f'<invoke name="noop"></invoke>{TOOL_CALL_END}'
        )

    def test_normal_text_before_tool_call(self):
        self.assert_stream_matches_final(
            f'Let me check. {TOOL_CALL_START}<invoke name="get_weather"></invoke>'
            f"{TOOL_CALL_END}"
        )

    def test_normal_text_after_tool_call(self):
        # Regression: text following the end token stayed buffered forever
        # because the end token itself was never consumed.
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather"></invoke>{TOOL_CALL_END}'
            " Done."
        )

    def test_plain_text_without_tool_call(self):
        self.assert_stream_matches_final("The weather is nice today.")

    def test_unknown_tool_is_dropped_not_leaked(self):
        # Regression: the invoke markup was emitted as content instead of being
        # dropped the way detect_and_parse drops it.
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="not_registered"></invoke>{TOOL_CALL_END}'
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
