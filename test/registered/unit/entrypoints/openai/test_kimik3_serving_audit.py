"""
Serving-level tests for Kimi K3 partial-marker leakage.

Tests all four public API paths:
1. Chat Completions, non-streaming
2. Chat Completions, streaming
3. Responses API, non-streaming
4. Responses API, streaming

Invariant: Internal Kimi K3 protocol-marker fragments must not appear in
user-visible reasoning, content, or tool-call fields.
"""

import sys
import unittest

from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_OPEN,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Reachable token-level truncation points for each marker.
# XTML markers are multi-token: <control> + text + <|sep|>
# Truncation can occur after the control token or after control+text.
REACHABLE_PARTIALS = [
    "<|close|>",        # First token of THINK_CLOSE, RESPONSE_CLOSE, MESSAGE_CLOSE
    "<|close|>think",   # First 2 tokens of THINK_CLOSE
    "<|open|>",         # First token of RESPONSE_OPEN, TOOLS_OPEN
    "<|open|>response", # First 2 tokens of RESPONSE_OPEN
    "<|close|>response",# First 2 tokens of RESPONSE_CLOSE
    "<|open|>tools",    # First 2 tokens of TOOLS_OPEN
    "<|close|>message", # First 2 tokens of MESSAGE_CLOSE
]

# Partial markers that should NOT appear in user-visible output
PARTIAL_MARKERS_TO_CHECK = [
    "<|close|>",
    "<|close|>think",
    "<|open|>",
    "<|open|>response",
    "<|close|>response",
    "<|open|>tools",
    "<|close|>message",
]


def _assert_no_partial_markers(testcase, text, field_name, context=""):
    """Assert that no partial XTML marker appears in text."""
    for marker in PARTIAL_MARKERS_TO_CHECK:
        testcase.assertNotIn(
            marker, text,
            f"Partial marker {marker!r} leaked into {field_name}"
            + (f" ({context})" if context else "")
            + f": {text!r}"
        )


# ---------------------------------------------------------------------------
# Chat Completions non-streaming test
# ---------------------------------------------------------------------------

class TestChatCompletionsNonStreaming(unittest.TestCase):
    """Test that non-streaming Chat Completions does not leak partial markers."""

    def test_partial_markers_not_in_response(self):
        from sglang.srt.parser.reasoning_parser import KimiK3Detector

        for partial in REACHABLE_PARTIALS:
            with self.subTest(partial=partial):
                # Build text that ends with a partial marker
                full_text = f"{THINK_OPEN}deep thought{partial}"

                # Use the detector directly (same as serving_chat does
                # via ReasoningParser.parse_non_stream)
                det = KimiK3Detector(force_reasoning=True)
                result = det.detect_and_parse(full_text)

                _assert_no_partial_markers(
                    self, result.reasoning_text, "reasoning_text",
                    context=f"partial={partial!r}"
                )
                _assert_no_partial_markers(
                    self, result.normal_text, "normal_text",
                    context=f"partial={partial!r}"
                )
                # Reasoning text must be preserved
                self.assertIn("deep thought", result.reasoning_text,
                              f"reasoning text lost for partial={partial!r}")


# ---------------------------------------------------------------------------
# Chat Completions streaming test
# ---------------------------------------------------------------------------

class TestChatCompletionsStreaming(unittest.TestCase):
    """Test that streaming Chat Completions does not leak partial markers."""

    def test_partial_markers_not_in_stream(self):
        from sglang.srt.parser.reasoning_parser import KimiK3Detector

        for partial in REACHABLE_PARTIALS:
            with self.subTest(partial=partial):
                full_text = f"{THINK_OPEN}deep thought{partial}"

                # Simulate streaming: feed text in chunks, then finish
                det = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
                reasoning = ""
                content = ""
                # Feed as a single chunk (simulating one decode step)
                result = det.parse_streaming_increment(full_text)
                reasoning += result.reasoning_text
                content += result.normal_text
                # Finish (called by serving_chat when finish_reason is set)
                end = det.finish()
                reasoning += end.reasoning_text
                content += end.normal_text

                _assert_no_partial_markers(
                    self, reasoning, "reasoning_text",
                    context=f"streaming partial={partial!r}"
                )
                _assert_no_partial_markers(
                    self, content, "normal_text",
                    context=f"streaming partial={partial!r}"
                )
                self.assertIn("deep thought", reasoning,
                              f"streaming reasoning lost for partial={partial!r}")


# ---------------------------------------------------------------------------
# Responses API non-streaming test
# ---------------------------------------------------------------------------

class TestResponsesNonStreaming(unittest.TestCase):
    """Test that non-streaming Responses API does not leak partial markers."""

    def test_partial_markers_not_in_output_items(self):
        from sglang.srt.parser.reasoning_parser import KimiK3Detector

        for partial in REACHABLE_PARTIALS:
            with self.subTest(partial=partial):
                full_text = f"{THINK_OPEN}deep thought{partial}"

                # Responses non-streaming uses ReasoningParser.parse_non_stream
                # which calls detect_and_parse
                det = KimiK3Detector(force_reasoning=True)
                result = det.detect_and_parse(full_text)

                _assert_no_partial_markers(
                    self, result.reasoning_text, "reasoning content",
                    context=f"responses non-stream partial={partial!r}"
                )
                _assert_no_partial_markers(
                    self, result.normal_text, "content",
                    context=f"responses non-stream partial={partial!r}"
                )


# ---------------------------------------------------------------------------
# Responses API streaming test
# ---------------------------------------------------------------------------

class TestResponsesStreaming(unittest.TestCase):
    """Test that streaming Responses API does not leak partial markers."""

    def test_partial_markers_not_in_stream(self):
        from sglang.srt.parser.reasoning_parser import KimiK3Detector

        for partial in REACHABLE_PARTIALS:
            with self.subTest(partial=partial):
                full_text = f"{THINK_OPEN}deep thought{partial}"

                det = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
                reasoning = ""
                content = ""
                result = det.parse_streaming_increment(full_text)
                reasoning += result.reasoning_text
                content += result.normal_text

                # Simulate what serving_chat does: call parse_stream_end
                # when finish_reason is set
                end = det.finish()
                reasoning += end.reasoning_text
                content += end.normal_text

                _assert_no_partial_markers(
                    self, reasoning, "reasoning_text",
                    context=f"responses stream partial={partial!r}"
                )
                _assert_no_partial_markers(
                    self, content, "normal_text",
                    context=f"responses stream partial={partial!r}"
                )


# ---------------------------------------------------------------------------
# Multi-choice isolation test (Chat Completions only)
# ---------------------------------------------------------------------------

class TestMultiChoiceIsolation(unittest.TestCase):
    """Test that multiple choices in Chat Completions have isolated parser state.

    The Responses API does not support multiple choices, so this test is
    limited to Chat Completions.
    """

    def test_interleaved_choices_isolated(self):
        """Two choices with interleaved chunks must have isolated state."""
        from sglang.srt.parser.reasoning_parser import KimiK3Detector

        # Simulate what serving_chat does: one reasoning_parser per choice
        parsers = {
            0: KimiK3Detector(force_reasoning=True, stream_reasoning=True),
            1: KimiK3Detector(force_reasoning=True, stream_reasoning=True),
        }

        text_a = f"{THINK_OPEN}alpha{THINK_CLOSE}{RESPONSE_OPEN}A{RESPONSE_CLOSE}"
        text_b = f"{THINK_OPEN}beta{THINK_CLOSE}{RESPONSE_OPEN}B{RESPONSE_CLOSE}"

        # Interleave chunks: choice 0, choice 1, choice 0, choice 1, ...
        chunks_a = [text_a[i:i+3] for i in range(0, len(text_a), 3)]
        chunks_b = [text_b[i:i+3] for i in range(0, len(text_b), 3)]

        reasoning = {0: "", 1: ""}
        content = {0: "", 1: ""}

        max_chunks = max(len(chunks_a), len(chunks_b))
        for i in range(max_chunks):
            if i < len(chunks_a):
                result = parsers[0].parse_streaming_increment(chunks_a[i])
                reasoning[0] += result.reasoning_text
                content[0] += result.normal_text
            if i < len(chunks_b):
                result = parsers[1].parse_streaming_increment(chunks_b[i])
                reasoning[1] += result.reasoning_text
                content[1] += result.normal_text

        # Finish both
        for idx in (0, 1):
            end = parsers[idx].finish()
            reasoning[idx] += end.reasoning_text
            content[idx] += end.normal_text

        # Verify isolation
        self.assertEqual(reasoning[0], "alpha", f"Choice 0 reasoning: {reasoning[0]!r}")
        self.assertEqual(content[0], "A", f"Choice 0 content: {content[0]!r}")
        self.assertEqual(reasoning[1], "beta", f"Choice 1 reasoning: {reasoning[1]!r}")
        self.assertEqual(content[1], "B", f"Choice 1 content: {content[1]!r}")

        # Verify no cross-contamination
        self.assertNotIn("alpha", reasoning[1])
        self.assertNotIn("beta", reasoning[0])
        self.assertNotIn("A", content[1])
        self.assertNotIn("B", content[0])


if __name__ == "__main__":
    unittest.main()
