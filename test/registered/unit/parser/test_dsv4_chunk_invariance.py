"""DeepSeek-V4 reasoning+tool parser chunk-size invariance tests.

Parametric coverage that feeds the SAME input to the streaming parser at
multiple chunk sizes and asserts the accumulated (reasoning, normal) output is
identical.  This is the invariant that production piped streaming guarantees
must hold: decode steps deliver tokens in arbitrarily-sized batches, and the
parser must not let the batch boundaries leak into its output.

These tests expose two distinct ways the current DeepSeekV4Detector breaks
that invariant:

BUG #1 -- tool_start_token is never set (not passed to the base class), so the
          DSML tool-start token "<{DSML_TOKEN}" is not routed out of reasoning
          content.  A tool block that directly follows reasoning (no
          think_end_token) stays in reasoning_text no matter how it is chunked
          or where its leading token is split.
BUG #2 -- BaseReasoningFormatDetector._parse_streaming_increment_impl clears
          self._buffer after every reasoning emission when stream_reasoning=True,
          destroying partial think_end_token / tool_start_token fragments that
          straddle a chunk boundary.

Sibling detectors (Apertus2509Detector, KimiK3Detector) implement
_ends_with_partial_token holdback and are immune; DeepSeekV4Detector inherits
the vulnerable base implementation.

stream_reasoning=False accumulates the whole stream in _buffer until the end
token is seen, so the buffer is never cleared mid-reasoning and the output is
truly chunk-size invariant (verified by the passing tests below).

These tests use unittest.TestCase (NOT CustomTestCase) to avoid the heavy
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
from sglang.srt.parser.reasoning_parser import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# ── Constants ───────────────────────────────────────────────────────────────

# Chunk sizes swept by every invariance test (vLLM uses [1,2,3,5,11,23,None]).
# None = feed entire text in one chunk (gold-standard reference).
CHUNK_SIZES = [1, 2, 3, 5, 7, 11, 23, None]

TOOL_START = f"<{DSML_TOKEN}"  # leading token of a DSML tool block
DSML_OPEN = f"<{DSML_TOKEN}tool_calls>"
TOOL_CALL = (
    f"<{DSML_TOKEN}tool_calls>"
    f'<{DSML_TOKEN}invoke name="search">'
    f'<{DSML_TOKEN}parameter name="query" string="true">hello</{DSML_TOKEN}parameter>'
    f"</{DSML_TOKEN}invoke>"
    f"</{DSML_TOKEN}tool_calls>"
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_detector(**kwargs):
    """Create a fresh DeepSeekV4 reasoning detector."""
    return DeepSeekV4Detector(**kwargs)


def _feed_streaming(detector, text, chunk_size=1):
    """Feed text char-by-char (or in fixed-size chunks) to the streaming parser.
    Returns (reasoning, normal) accumulated text.
    chunk_size=None feeds the entire text as a single chunk (gold-standard)."""
    reasoning = ""
    normal = ""
    if chunk_size is None:
        chunks = [text]
    else:
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        r = detector.parse_streaming_increment(chunk)
        reasoning += r.reasoning_text
        normal += r.normal_text
    r = detector.finish()
    reasoning += r.reasoning_text
    normal += r.normal_text
    return reasoning, normal


def _feed_sequence(detector, chunks):
    """Feed an explicit list of chunks (non-uniform boundaries), plus finish()."""
    reasoning = ""
    normal = ""
    for chunk in chunks:
        r = detector.parse_streaming_increment(chunk)
        reasoning += r.reasoning_text
        normal += r.normal_text
    r = detector.finish()
    reasoning += r.reasoning_text
    normal += r.normal_text
    return reasoning, normal


def _think_end_was_detected(reasoning, normal):
    """True iff the think_end_token did NOT leak into reasoning and the trailing
    ' normal' text was emitted as normal content (i.e. the token was recognized)."""
    return THINK_END not in reasoning and " normal" in normal


# ── Tests ───────────────────────────────────────────────────────────────────


class TestChunkSizeInvarianceReasoning(unittest.TestCase):
    """Feeding the SAME input at different chunk sizes SHALL yield identical
    (reasoning_text, normal_text) output.

    stream_reasoning=True violates this because of BUG #2; stream_reasoning=False
    preserves the buffer and is truly invariant.
    """

    def test_pure_reasoning_stdstream_chunk_invariant(self):
        """stream_reasoning=True (BUG #2): a think_end_token split across a
        multi-char chunk is destroyed when _buffer is cleared after a reasoning
        emission, so the end token leaks into reasoning_text and the normal text
        is never emitted.  Different chunk sizes therefore diverge."""
        source = f"{THINK_START}reasoning content here{THINK_END}normal content"
        outputs = [
            _feed_streaming(_make_detector(stream_reasoning=True), source, cs)
            for cs in CHUNK_SIZES
        ]
        for r, n in outputs[1:]:
            self.assertEqual(r, outputs[0][0])
            self.assertEqual(n, outputs[0][1])

    def test_pure_reasoning_buffered_chunk_invariant(self):
        """stream_reasoning=False (no buffer clearing) SHALL be chunk-size
        invariant for pure reasoning content."""
        source = f"{THINK_START}reasoning content here{THINK_END}normal content"
        outputs = [
            _feed_streaming(_make_detector(stream_reasoning=False), source, cs)
            for cs in CHUNK_SIZES
        ]
        for r, n in outputs[1:]:
            self.assertEqual(r, outputs[0][0])
            self.assertEqual(n, outputs[0][1])

    def test_reasoning_tool_streams_chunk_invariant(self):
        """stream_reasoning=True (BUG #2): a DSML tool call following a
        reasoning block also gets its think_end_token destroyed for multi-char
        chunks, so the tool block leaks into reasoning_content and the outputs
        across chunk sizes diverge."""
        source = f"{THINK_START}my reasoning{THINK_END}{TOOL_CALL}"
        outputs = [
            _feed_streaming(_make_detector(stream_reasoning=True), source, cs)
            for cs in CHUNK_SIZES
        ]
        for r, n in outputs[1:]:
            self.assertEqual(r, outputs[0][0])
            self.assertEqual(n, outputs[0][1])

    def test_reasoning_tool_buffered_chunk_invariant(self):
        """stream_reasoning=False SHALL be chunk-size invariant when a DSML tool
        block follows a reasoning block."""
        source = f"{THINK_START}my reasoning{THINK_END}{TOOL_CALL}"
        outputs = [
            _feed_streaming(_make_detector(stream_reasoning=False), source, cs)
            for cs in CHUNK_SIZES
        ]
        for r, n in outputs[1:]:
            self.assertEqual(r, outputs[0][0])
            self.assertEqual(n, outputs[0][1])


class TestThinkEndBoundarySplit(unittest.TestCase):
    """The think_end_token may straddle two chunks; the parser SHALL still
    recognize it.  We split think_end_token at every internal position and assert
    it is detected (i.e. the trailing ' normal' lands in normal_text and the token
    does NOT leak into reasoning_text)."""

    @staticmethod
    def _split_positions():
        # Split ONLY inside the token (positions 1..len-1); boundaries at either
        # edge leave the token contiguous.
        return list(range(1, len(THINK_END)))

    def test_think_end_split_across_chunks_streaming(self):
        """stream_reasoning=True (BUG #2): when think_end_token is split across
        a chunk boundary, the buffer is cleared after the previous reasoning
        emission and the first fragment is lost, so the token is never re-seen
        and the trailing normal text leaks into reasoning_content."""
        for i in self._split_positions():
            chunks = [
                f"{THINK_START}reasoning{THINK_END[:i]}",
                f"{THINK_END[i:]} normal",
            ]
            reasoning, normal = _feed_sequence(
                _make_detector(stream_reasoning=True), chunks
            )
            self.assertTrue(
                _think_end_was_detected(reasoning, normal),
                msg=f"split at {i}: reasoning={reasoning!r} normal={normal!r}",
            )

    def test_think_end_split_across_chunks_buffered(self):
        """stream_reasoning=False SHALL detect a think_end_token split at any
        internal position."""
        for i in self._split_positions():
            chunks = [
                f"{THINK_START}reasoning{THINK_END[:i]}",
                f"{THINK_END[i:]} normal",
            ]
            reasoning, normal = _feed_sequence(
                _make_detector(stream_reasoning=False), chunks
            )
            self.assertTrue(
                _think_end_was_detected(reasoning, normal),
                msg=f"split at {i}: reasoning={reasoning!r}, normal={normal!r}",
            )


class TestToolStartBoundarySplit(unittest.TestCase):
    """The tool_start_token ("<{DSML_TOKEN}") DIRECTLY following reasoning (no
    think_end) SHALL route the DSML block to normal_text so the tool call parser
    can detect it.  This must hold regardless of where the token is split across
    chunks.  BUG #1 makes every split position fail.
    """

    @staticmethod
    def _tool_after_reasoning(i):
        # Split TOOL_START at internal position i: chunk1 ends in a partial
        # tool_start_token, chunk2 completes it and carries the whole body.
        return [
            f"{THINK_START}reasoning{TOOL_START[:i]}",
            f"{TOOL_START[i:]}{TOOL_CALL[len(TOOL_START):]}",
        ]

    def _assert_routed_to_normal(self, i, reasoning, normal):
        self.assertIn(TOOL_START, normal, f"split at {i}: tool not in normal_text")
        self.assertNotIn(
            DSML_OPEN, reasoning, f"split at {i}: DSML leaked into reasoning"
        )

    def test_direct_tool_split_streaming_routes_to_normal(self):
        """BUG #1: tool_start_token is not set, so a DSML tool call that
        directly follows reasoning (without think_end) stays in reasoning_text
        for every split position of the tool_start_token header."""
        for i in range(1, len(TOOL_START)):
            reasoning, normal = _feed_sequence(
                _make_detector(stream_reasoning=True), self._tool_after_reasoning(i)
            )
            self._assert_routed_to_normal(i, reasoning, normal)

    def test_direct_tool_split_buffered_routes_to_normal(self):
        """BUG #1 is independent of stream_reasoning: even with the buffer never
        cleared, the missing tool_start_token keeps the direct tool block inside
        reasoning_content at every split position."""
        for i in range(1, len(TOOL_START)):
            reasoning, normal = _feed_sequence(
                _make_detector(stream_reasoning=False), self._tool_after_reasoning(i)
            )
            self._assert_routed_to_normal(i, reasoning, normal)


class TestMTPScenario(unittest.TestCase):
    """MTP / multi-token batch: stream_interval > 1 delivers 2-3 tokens per
    decode step.  The final accumulated output SHALL match a one-shot feed of
    the same stream.  With stream_reasoning=True, BUG #2 breaks this."""

    _MTP = (
        f"{THINK_START}Let me reason about it." f"{THINK_END}The command is{TOOL_CALL}"
    )

    def _baseline_reference(self):
        # The intended stream output: char-by-char feeding of the same stream
        # (stream_reasoning=False is chunk-invariant, so larger batches SHALL
        r, n = _feed_streaming(
            _make_detector(stream_reasoning=False), self._MTP, chunk_size=1
        )
        return r, n

    def test_mtp_batch_streams_matches_non_streaming(self):
        """stream_reasoning=True (BUG #2): 2-3-token batches cross the
        think_end/tool boundaries and the chunk-wise buffer-clearing corrupts the
        accumulated output vs an equivalent one-shot parse."""
        for cs in (2, 3, 4):
            reasoning, normal = _feed_streaming(
                _make_detector(stream_reasoning=True), self._MTP, cs
            )
            ref_r, ref_n = self._baseline_reference()
            self.assertEqual(reasoning, ref_r)
            self.assertEqual(normal, ref_n)

    def test_mtp_batch_buffered(self):
        """stream_reasoning=False: multi-token batches SHALL match a one-shot
        parse of the same stream."""
        for cs in (2, 3, 4):
            reasoning, normal = _feed_streaming(
                _make_detector(stream_reasoning=False), self._MTP, cs
            )
            ref_r, ref_n = self._baseline_reference()
            self.assertEqual(reasoning, ref_r)
            self.assertEqual(normal, ref_n)


class TestDetokenizerHoldback(unittest.TestCase):
    """A detokenizer may deliver a chunk ending mid-way through a special token,
    with the completion arriving on the next chunk.  The parser SHALL hold (and
    later re-assemble) the partial token instead of emitting a torn fragment.
    """

    _PARTIAL_THINK_END = THINK_END[:3]
    _PARTIAL_TOOL_START = TOOL_START[:3]

    def test_partial_think_end_holdback_streams(self):
        """stream_reasoning=True (BUG #2): a think_end_token cut mid-way at the
        end of a chunk is cleared together with the reasoning buffer, so the
        completion on the next chunk starts a fresh buffer and the trailing
        normal text leaks into reasoning_content."""
        chunks = [
            f"{THINK_START}reasoning{self._PARTIAL_THINK_END}",
            f"{THINK_END[3:]} normal",
        ]
        reasoning, normal = _feed_sequence(
            _make_detector(stream_reasoning=True), chunks
        )
        self.assertTrue(_think_end_was_detected(reasoning, normal))

    def test_partial_think_end_holdback_buffered(self):
        """stream_reasoning=False: a think_end_token written across a chunk
        boundary is held in _buffer and re-assembled on the next chunk."""
        chunks = [
            f"{THINK_START}reasoning{self._PARTIAL_THINK_END}",
            f"{THINK_END[3:]} normal",
        ]
        reasoning, normal = _feed_sequence(
            _make_detector(stream_reasoning=False), chunks
        )
        self.assertTrue(_think_end_was_detected(reasoning, normal))

    def test_partial_tool_start_holdback_streams(self):
        """stream_reasoning=True (BUG #2): a tool_start_token cut mid-way at the
        end of a chunk is dropped by the buffer-clear, so the DSML block is never
        routed to normal_text."""
        chunks = [
            f"{THINK_START}reasoning{self._PARTIAL_TOOL_START}",
            f"{TOOL_START[3:]}{TOOL_CALL[len(TOOL_START):]}",
        ]
        reasoning, normal = _feed_sequence(
            _make_detector(stream_reasoning=True), chunks
        )
        self.assertIn(TOOL_START, normal)
        self.assertNotIn(DSML_OPEN, reasoning)

    def test_partial_tool_holdback_buffered(self):
        """stream_reasoning=False: the tool_start_token is re-assembled across a
        boundary without being torn and routed to normal_text (BUG #1 fix)."""
        chunks = [
            f"{THINK_START}reasoning{self._PARTIAL_TOOL_START}",
            f"{TOOL_START[3:]}{TOOL_CALL[len(TOOL_START):]}",
        ]
        _, normal = _feed_sequence(_make_detector(stream_reasoning=False), chunks)
        self.assertIn(
            TOOL_START,
            normal,
            "partial tool_start SHALL be re-assembled and routed to normal",
        )


if __name__ == "__main__":
    unittest.main()
