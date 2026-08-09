"""DeepSeek-V4 reasoning parser integration tests.

Exposes BUG #1 (missing tool_start_token), BUG #2 (chunk boundary destroys
partial think_end_token with stream_reasoning=True), and BUG #4 (multi-think-block
cycling broken — stripped_think_start never reset).

These tests use unittest.TestCase (not CustomTestCase) to avoid the heavy
test_utils import chain that segfaults on macOS.  All imports are lightweight
parser-only paths.

Related OpenSpec change: dsv4-reasoning-tool-parser-joint-test
"""

import unittest

from sglang.srt.entrypoints.openai.encoding_dsv4 import dsml_token as DSML_TOKEN
from sglang.srt.entrypoints.openai.encoding_dsv4 import thinking_end_token as THINK_END
from sglang.srt.entrypoints.openai.encoding_dsv4 import (
    thinking_start_token as THINK_START,
)
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import (
    DeepSeekV4Detector as V4ToolDetector,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# ── Constants ───────────────────────────────────────────────────────────────

TOOL_START = f"<{DSML_TOKEN}"  # "<｜DSML｜"
DSML_OPEN = f"<{DSML_TOKEN}tool_calls>"
DSML_CLOSE = f"</{DSML_TOKEN}tool_calls>"
INVOKE_OPEN = f'<{DSML_TOKEN}invoke name="ls">'
INVOKE_CLOSE = f"</{DSML_TOKEN}invoke>"
PARAM_FMT = (
    f'<{DSML_TOKEN}parameter name="command" string="true">pwd</{DSML_TOKEN}parameter>'
)

SAMPLE_TOOL_CALL = (
    f"{DSML_OPEN}\n{INVOKE_OPEN}\n{PARAM_FMT}\n{INVOKE_CLOSE}\n{DSML_CLOSE}"
)
SAMPLE_TOOLS = [
    Tool(function=Function(name="ls", parameters={"type": "object", "properties": {}}))
]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_detector(**kwargs):
    """Create a fresh DeepSeekV4 reasoning detector."""
    return DeepSeekV4Detector(**kwargs)


def _feed_streaming(detector, text, chunk_size=1):
    """Feed text char-by-char (or in fixed-size chunks) to streaming parser.
    Returns (reasoning, normal) accumulated text.
    """
    reasoning = ""
    normal = ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        r = detector.parse_streaming_increment(chunk)
        reasoning += r.reasoning_text
        normal += r.normal_text
    r = detector.finish()
    reasoning += r.reasoning_text
    normal += r.normal_text
    return reasoning, normal


def _feed_non_streaming(detector, text):
    """Feed text as a single block to non-streaming parser."""
    r = detector.detect_and_parse(text)
    return r.reasoning_text, r.normal_text


# ── Tests ───────────────────────────────────────────────────────────────────


class TestDSV4ToolStartToken(unittest.TestCase):
    """BUG #1: DeepSeekV4Detector does not pass tool_start_token to super().__init__().

    Without tool_start_token, the DSML tool-call block stays in reasoning_content
    and the tool call parser never sees it.  These tests expose the missing
    routing.
    """

    def test_tool_start_token_is_set(self):
        """tool_start_token SHALL be set to '<｜DSML｜' so DSML blocks route to
        the tool call parser instead of staying in reasoning_content."""
        detector = _make_detector()
        self.assertIsNotNone(detector.tool_start_token)
        self.assertEqual(detector.tool_start_token, TOOL_START)

    def test_dsml_block_routes_to_normal_not_reasoning(self):
        """When the model emits reasoning then a DSML tool call (without an
        explicit think_end_token), the DSML block SHALL appear in normal_text
        so the tool call parser can detect it."""
        detector = _make_detector(stream_reasoning=True)
        text = f"{THINK_START}reasoning{SAMPLE_TOOL_CALL}"
        reasoning, normal = _feed_streaming(detector, text)
        self.assertIn(TOOL_START, normal)
        self.assertNotIn(DSML_OPEN, reasoning)

    def test_joint_reasoning_then_tool_call_streaming(self):
        """End-to-end: reasoning parser routes DSML to tool parser, tool parser
        extracts the call."""
        detector = _make_detector(stream_reasoning=True)
        text = f"{THINK_START}I should list files.{SAMPLE_TOOL_CALL}"
        reasoning, normal = _feed_streaming(detector, text)
        # Feed normal_text to tool call parser
        tool_parser = FunctionCallParser(
            tools=SAMPLE_TOOLS, tool_call_parser="deepseekv4"
        )
        _, calls = tool_parser.parse_non_stream(normal)
        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[0].name, "ls")


class TestDSV4ChunkBoundaryBUG2(unittest.TestCase):
    """BUG #2: stream_reasoning=True clears _buffer after every reasoning
    emission, destroying partial think_end_token fragments split across chunk
    boundaries.

    Sibling detectors Apertus2509Detector and CohereCommand4Detector implement
    _ends_with_partial_token and are immune.  DeepSeekV4Detector inherits the
    vulnerable base implementation.
    """

    def test_think_end_split_across_chunks_stream_reasoning(self):
        """With stream_reasoning=True and chunk_size > 1, a think_end_token
        split across chunk boundaries SHALL still be detected."""
        detector = _make_detector(stream_reasoning=True)
        # Feed reasoning first
        detector.parse_streaming_increment(THINK_START)
        detector.parse_streaming_increment("reasoning text")
        # Now feed a chunk that ends with partial THINK_END
        half = len(THINK_END) // 2
        r1 = detector.parse_streaming_increment(THINK_END[:half])
        # Partial token SHALL NOT leak into reasoning_text
        self.assertNotIn(THINK_END[:half], r1.reasoning_text)
        # Complete the token
        r2 = detector.parse_streaming_increment(THINK_END[half:] + "normal text")
        # After think_end, _in_reasoning SHALL be False
        self.assertFalse(detector._in_reasoning)
        # Normal text SHALL appear
        self.assertIn("normal text", r2.normal_text)

    def test_chunk_size_invariance_stream_reasoning(self):
        """Same input at different chunk sizes SHALL produce identical output."""
        source = f"{THINK_START}reasoning{THINK_END}normal answer"
        # char-by-char
        d1 = _make_detector(stream_reasoning=True)
        r1, n1 = _feed_streaming(d1, source, chunk_size=1)
        # multi-char chunks
        d2 = _make_detector(stream_reasoning=True)
        r2, n2 = _feed_streaming(d2, source, chunk_size=5)
        self.assertEqual(r1, r2)
        self.assertEqual(n1, n2)

    def test_chunk_size_invariance_stream_reasoning_false(self):
        """With stream_reasoning=False, chunk size SHALL NOT affect output.
        (Buffer is never cleared, so partial tokens are preserved.)"""
        source = f"{THINK_START}reasoning{THINK_END}normal answer"
        d1 = _make_detector(stream_reasoning=False)
        r1, n1 = _feed_streaming(d1, source, chunk_size=1)
        d2 = _make_detector(stream_reasoning=False)
        r2, n2 = _feed_streaming(d2, source, chunk_size=5)
        self.assertEqual(r1, r2)
        self.assertEqual(n1, n2)


class TestDSV4MultiThinkBlockBUG4(unittest.TestCase):
    """BUG #4: stripped_think_start is never reset after the first reasoning
    block closes, so a second think_start token leaks into normal_text instead
    of starting a new reasoning block.
    """

    def test_multi_think_block_cycling(self):
        """reasoning → think_end → content → think_start → reasoning → think_end → content
        SHALL produce two reasoning blocks and two normal blocks."""
        detector = _make_detector(stream_reasoning=True)
        source = (
            f"{THINK_START}first reasoning{THINK_END}"
            f"first answer"
            f"{THINK_START}second reasoning{THINK_END}"
            f"second answer"
        )
        reasoning, normal = _feed_streaming(detector, source, chunk_size=1)
        self.assertIn("first reasoning", reasoning)
        self.assertIn("second reasoning", reasoning)
        self.assertIn("first answer", normal)
        self.assertIn("second answer", normal)

    def test_single_think_block_works(self):
        """Sanity: a single reasoning block SHALL work correctly."""
        detector = _make_detector(stream_reasoning=True)
        source = f"{THINK_START}only reasoning{THINK_END}only answer"
        reasoning, normal = _feed_streaming(detector, source, chunk_size=1)
        self.assertIn("only reasoning", reasoning)
        self.assertIn("only answer", normal)


class TestDSV4NonStreamingToolRouting(unittest.TestCase):
    """Non-streaming path: verify DSML block routing with and without
    tool_start_token fix."""

    def test_non_streaming_dsml_routes_to_normal(self):
        """In non-streaming mode, DSML tool call SHALL appear in normal_text
        (not reasoning_content) so the tool call parser can detect it."""
        detector = _make_detector()
        text = f"{THINK_START}I should run ls{THINK_END}{SAMPLE_TOOL_CALL}"
        reasoning, normal = _feed_non_streaming(detector, text)
        self.assertIn(TOOL_START, normal)
        self.assertNotIn(DSML_OPEN, reasoning)

    def test_non_streaming_pure_reasoning(self):
        """Pure reasoning without tool calls SHALL work correctly."""
        detector = _make_detector()
        text = f"{THINK_START}just thinking{THINK_END}just answer"
        reasoning, normal = _feed_non_streaming(detector, text)
        self.assertIn("just thinking", reasoning)
        self.assertIn("just answer", normal)


class TestDSV4ArgumentAssertions(unittest.TestCase):
    """Verify tool call arguments are correctly parsed when the fix is applied."""

    def test_tool_call_arguments_parsed(self):
        """When DSML routes to tool parser, arguments SHALL be correctly
        extracted as JSON."""
        detector = _make_detector(stream_reasoning=True)
        text = f"{THINK_START}reasoning{SAMPLE_TOOL_CALL}"
        _, normal = _feed_streaming(detector, text)
        tool_parser = FunctionCallParser(
            tools=SAMPLE_TOOLS, tool_call_parser="deepseekv4"
        )
        _, calls = tool_parser.parse_non_stream(normal)
        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[0].name, "ls")
        # Arguments SHALL be valid JSON with expected fields
        import json

        args = json.loads(calls[0].parameters)
        self.assertEqual(args["command"], "pwd")


class TestDSV4FinishLeakage(unittest.TestCase):
    """Bug 4 (think_start buffer writeback): After the fix, finish() SHALL NOT
    emit think_start_token in reasoning_text, even when the stream is cut off
    mid-reasoning (max_tokens, client disconnect)."""

    def test_finish_no_think_start_in_reasoning_streaming(self):
        """stream_reasoning=True: finish() after mid-reasoning cutoff SHALL NOT
        contain think_start_token in reasoning_text."""
        detector = _make_detector(stream_reasoning=True)
        detector.parse_streaming_increment(THINK_START)
        detector.parse_streaming_increment("partial reasoning")
        r = detector.finish()
        self.assertNotIn(THINK_START, r.reasoning_text)

    def test_finish_no_think_start_in_reasoning_buffered(self):
        """stream_reasoning=False: finish() after mid-reasoning cutoff SHALL NOT
        contain think_start_token in reasoning_text (Bug 4 fix: buffer writeback)."""
        detector = _make_detector(stream_reasoning=False)
        detector.parse_streaming_increment(THINK_START)
        detector.parse_streaming_increment("partial reasoning")
        r = detector.finish()
        self.assertNotIn(THINK_START, r.reasoning_text)


class TestDSV4TerminalLeakage(unittest.TestCase):
    """After parsing, neither think_start_token nor think_end_token SHALL appear
    in reasoning_text or normal_text — they are structural markers, not content."""

    def test_no_think_tokens_leak_streaming(self):
        """stream_reasoning=True: after a full reasoning→normal cycle, neither
        THINK_START nor THINK_END shall appear in either output."""
        detector = _make_detector(stream_reasoning=True)
        source = f"{THINK_START}reasoning here{THINK_END}normal answer"
        reasoning, normal = _feed_streaming(detector, source, chunk_size=1)
        self.assertNotIn(THINK_START, reasoning)
        self.assertNotIn(THINK_END, reasoning)
        self.assertNotIn(THINK_START, normal)
        self.assertNotIn(THINK_END, normal)

    def test_no_think_tokens_leak_buffered(self):
        """stream_reasoning=False: same invariant — no think tokens in output."""
        detector = _make_detector(stream_reasoning=False)
        source = f"{THINK_START}reasoning here{THINK_END}normal answer"
        reasoning, normal = _feed_streaming(detector, source, chunk_size=5)
        self.assertNotIn(THINK_START, reasoning)
        self.assertNotIn(THINK_END, reasoning)
        self.assertNotIn(THINK_START, normal)
        self.assertNotIn(THINK_END, normal)

    def test_no_think_tokens_leak_multi_block(self):
        """Multi-block cycling: no think tokens leak across two reasoning blocks."""
        detector = _make_detector(stream_reasoning=True)
        source = (
            f"{THINK_START}first{THINK_END}answer1"
            f"{THINK_START}second{THINK_END}answer2"
        )
        reasoning, normal = _feed_streaming(detector, source, chunk_size=1)
        self.assertNotIn(THINK_START, reasoning)
        self.assertNotIn(THINK_END, reasoning)
        self.assertNotIn(THINK_START, normal)
        self.assertNotIn(THINK_END, normal)


class TestDSV4MalformedJSONStreaming(unittest.TestCase):
    """Bug 6: MalformedJSON not caught in parse_streaming_increment.

    The tool call parser's _parse_parameters_from_xml catches MalformedJSON
    internally (line 183).  The outer except-Exception handler (line 375)
    catches any other exception and clears the buffer to prevent cascading
    parse failures (Bug 6 fix).  These tests verify both layers.
    """

    _TOOLS = [
        Tool(
            function=Function(
                name="ls", parameters={"type": "object", "properties": {}}
            )
        )
    ]

    def test_malformed_json_params_no_crash(self):
        """Feeding a DSML invoke block with malformed JSON parameters SHALL NOT
        crash — MalformedJSON is caught inside _parse_parameters_from_xml."""
        det = V4ToolDetector()
        malformed = (
            f"{DSML_OPEN}\n"
            f'<{DSML_TOKEN}invoke name="ls">\n'
            f'<{DSML_TOKEN}parameter name="cmd" string="true">'
            f'{{{{"broken json'  # malformed JSON inside XML parameter
            f"</{DSML_TOKEN}parameter>\n"
            f"</{DSML_TOKEN}invoke>\n"
            f"</{DSML_TOKEN}tool_calls>"
        )
        # This should not raise
        result = det.parse_streaming_increment(malformed, self._TOOLS)
        # The tool call SHALL be extracted (with garbled parameters)
        self.assertTrue(
            len(result.calls) > 0 or result.normal_text != "",
            "Expected either tool calls or normal_text from malformed input",
        )

    def test_outer_exception_clears_buffer(self):
        """When an unexpected exception escapes _parse_parameters_from_xml,
        the outer except handler SHALL clear the buffer (Bug 6 fix)."""
        det = V4ToolDetector()
        # Monkey-patch _parse_parameters_from_xml to raise a non-JSON exception
        original = det._parse_parameters_from_xml

        def raise_runtime(*a, **kw):
            raise RuntimeError("simulated unexpected error")

        setattr(det, "_parse_parameters_from_xml", raise_runtime)
        try:
            dsml_input = (
                f"{DSML_OPEN}\n"
                f'<{DSML_TOKEN}invoke name="ls">\n'
                f"</{DSML_TOKEN}invoke>\n"
                f"</{DSML_TOKEN}tool_calls>"
            )
            result = det.parse_streaming_increment(dsml_input, self._TOOLS)
            # Buffer SHALL be cleared by outer exception handler
            self.assertEqual(det._buffer, "")
        finally:
            setattr(det, "_parse_parameters_from_xml", original)


if __name__ == "__main__":
    unittest.main()
