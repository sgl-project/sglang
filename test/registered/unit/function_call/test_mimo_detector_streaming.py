"""Streaming regression tests for srt/function_call/mimo_detector.py (#33186).

No server, no model. Guards the equivalence contract between one-shot
detect_and_parse and parse_streaming_increment that sibling detectors
(Qwen25, GLM-4.7) already honor.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="search",
            description="Search",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        ),
    ),
]

CALL_BLOCK = (
    "<tool_call>\n<function=search>\n"
    "<parameter=query>ls</parameter>\n</function>\n</tool_call>"
)


def _stream(detector, text, chunk_size):
    normal = ""
    calls = []
    for i in range(0, len(text), chunk_size):
        r = detector.parse_streaming_increment(text[i : i + chunk_size], TOOLS)
        normal += r.normal_text
        calls.extend(r.calls)
    return normal, calls


class TestMiMoStreamingRegression(CustomTestCase):
    def test_text_after_last_tool_call_is_flushed(self):
        """Bug regression (#33186): after the first parsed call, increments
        with no further bot token were buffered forever, so anything the
        model said after its last tool call was silently dropped from the
        stream. Trailing text must still reach normal_text."""
        detector = MiMoDetector()
        r1 = detector.parse_streaming_increment(CALL_BLOCK, TOOLS)
        self.assertEqual([c.name for c in r1.calls], ["search"])
        r2 = detector.parse_streaming_increment("Done, files are listed.", TOOLS)
        self.assertEqual(r2.normal_text, "Done, files are listed.")

    def test_split_bot_token_across_increments_still_parses(self):
        """Bug regression (#33186): a bot token split across increments was
        flushed to normal_text as raw markup and the tool call was lost.
        Only a potential marker suffix may be held back."""
        detector = MiMoDetector()
        r1 = detector.parse_streaming_increment("I will run it.\n<tool", TOOLS)
        self.assertEqual(r1.normal_text, "I will run it.\n")
        r2 = detector.parse_streaming_increment(CALL_BLOCK[len("<tool") :], TOOLS)
        self.assertEqual([c.name for c in r2.calls], ["search"])
        self.assertNotIn("<tool", r1.normal_text + r2.normal_text)

    def test_char_level_streaming_matches_one_shot(self):
        """Derived property: streamed output (concatenated normal_text plus
        accumulated calls) must equal the one-shot parse of the same text,
        regardless of chunk boundaries — the guarantee Qwen25/GLM-4.7
        streaming already provides for the same wire format."""
        text = "Let me check.\n" + CALL_BLOCK
        one_shot = MiMoDetector().detect_and_parse(text, TOOLS)

        streamed_normal, streamed_calls = _stream(MiMoDetector(), text, 1)
        self.assertEqual(streamed_normal, one_shot.normal_text)
        self.assertEqual(
            [c.name for c in streamed_calls if c.name],
            [c.name for c in one_shot.calls],
        )
        self.assertEqual(
            "".join(c.parameters for c in streamed_calls if c.parameters),
            "".join(c.parameters for c in one_shot.calls),
        )

    def test_text_between_two_calls_is_preserved(self):
        """Completeness: interleaved normal text between two calls must not
        be dropped or reordered once both calls have streamed."""
        text = CALL_BLOCK + "\nand also\n" + CALL_BLOCK
        streamed_normal, streamed_calls = _stream(MiMoDetector(), text, 7)
        self.assertEqual(len([c for c in streamed_calls if c.name]), 2)
        self.assertIn("and also", streamed_normal)


if __name__ == "__main__":
    unittest.main()
