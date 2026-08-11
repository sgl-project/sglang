"""Unit tests for srt/function_call/inkling_detector.py — no server, no model loading.

Complements TestInklingDetector in test_function_call_parser.py. The cases here
pin branches that file does not cover: the has_tool_call marker contract, marker
ordering by position, empty/misshapen/unterminated tool-call frames, the
recovery paths (through end-of-response, control-token cleaning, missing
fields, discard-on-success), the streaming raw-text frame, chunk-boundary
atomicity of control tokens, the incomplete-frame prefix flush, and the
index continuity of a recovered streaming call.
"""

import json
import math
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.inkling_detector import InklingDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Inkling framing tokens in the exact wire form the model emits.
MESSAGE_MODEL = "<|message_model|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
END_MESSAGE = "<|end_message|>"


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="weather",
                description="Lookup weather",
                parameters={"type": "object"},
            ),
        )
    ]


def _canonical_call(args=None):
    """A complete canonical Inkling JSON tool-call block for the weather tool."""
    payload = json.dumps(
        {"name": "weather", "args": args if args is not None else {}},
        separators=(",", ":"),
    )
    return f"{MESSAGE_MODEL}weather{INVOKE_TOOL_JSON}{payload}{END_MESSAGE}"


class TestInklingDetectorMarkerDetection(CustomTestCase):
    def test_has_tool_call_fires_only_on_invoke_markers(self):
        """Marker contract: only the two invoke tokens count as tool-call
        starts. Other framing tokens (message boundaries, content kinds) must
        not route plain responses into the tool-parse path."""
        detector = InklingDetector()
        self.assertTrue(detector.has_tool_call(f"before {INVOKE_TOOL_JSON} after"))
        self.assertTrue(detector.has_tool_call(f"before {INVOKE_TOOL_TEXT} after"))
        self.assertFalse(detector.has_tool_call("plain assistant text"))
        self.assertFalse(
            detector.has_tool_call(MESSAGE_MODEL + CONTENT_TEXT + END_MESSAGE)
        )


class TestInklingDetectorOneShotParse(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = InklingDetector()

    # ==================== detect_and_parse: text around calls ====================

    def test_visible_prefix_survives_header_stripping(self):
        """Visible text before a call is content; the trailing
        <|message_model|>name header is author metadata and must be stripped,
        leaving only the prefix as normal text."""
        source = "Sure, let me check." + _canonical_call(args={"city": "SF"})
        result = self.detector.detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, "Sure, let me check.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")

    # ==================== detect_and_parse: indices and marker order ====================

    def test_batch_calls_get_sequential_indices(self):
        """Two valid calls in one response are indexed 0 then 1 by position,
        each keeping its own args."""
        source = _canonical_call(args={"city": "SF"}) + _canonical_call(
            args={"city": "NY"}
        )
        result = self.detector.detect_and_parse(source, self.tools)
        self.assertEqual([c.tool_index for c in result.calls], [0, 1])
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "NY"})

    def test_markers_order_by_position_not_type(self):
        """The next call is whichever marker comes first, regardless of kind:
        a text marker before a json marker must yield the raw-text call first,
        and vice versa."""
        json_first = (
            _canonical_call(args={"city": "SF"})
            + f"{INVOKE_TOOL_TEXT}raw body{END_MESSAGE}"
        )
        result = self.detector.detect_and_parse(json_first, self.tools)
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in result.calls],
            [("weather", {"city": "SF"}), ("", {"text": "raw body"})],
        )

        text_first = f"{INVOKE_TOOL_TEXT}raw body{END_MESSAGE}" + _canonical_call(
            args={"city": "SF"}
        )
        result = self.detector.detect_and_parse(text_first, self.tools)
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in result.calls],
            [("", {"text": "raw body"}), ("weather", {"city": "SF"})],
        )

    # ==================== detect_and_parse: empty / misshapen frames ====================

    def test_degenerate_payloads_yield_no_call(self):
        """A payload that is not a name/args object never becomes a call —
        neither an empty frame nor valid JSON of the wrong shape (e.g. an
        array): both fail the strict parse and the recovery shape check, so
        the body is surfaced as content instead."""
        empty = self.detector.detect_and_parse(
            INVOKE_TOOL_JSON + END_MESSAGE, self.tools
        )
        self.assertEqual(empty.calls, [])
        self.assertEqual(empty.normal_text, "")

        array = self.detector.detect_and_parse(
            INVOKE_TOOL_JSON + "[1, 2, 3]" + END_MESSAGE, self.tools
        )
        self.assertEqual(array.calls, [])
        self.assertIn("[1, 2, 3]", array.normal_text)
        self.assertNotIn("<|", array.normal_text)

    def test_payload_missing_name_yields_no_call(self):
        """A payload without a name fails the strict path (name must be a
        string) and the recovery path (recovery requires a nonempty name), so
        nothing is emitted as a call."""
        result = self.detector.detect_and_parse(
            INVOKE_TOOL_JSON + '{"args":{}}' + END_MESSAGE, self.tools
        )
        self.assertEqual(result.calls, [])
        self.assertIn('{"args":{}}', result.normal_text)

    def test_payload_missing_args_yields_no_call(self):
        """A payload with a name but no args object fails even recovery: the
        name gate passes, yet the shape check still rejects it, so no call is
        surfaced."""
        result = self.detector.detect_and_parse(
            INVOKE_TOOL_JSON + '{"name":"weather"}' + END_MESSAGE, self.tools
        )
        self.assertEqual(result.calls, [])
        self.assertIn('{"name":"weather"}', result.normal_text)

    # ==================== detect_and_parse: unterminated frames ====================

    def test_unterminated_json_frame_is_recovered_through_end_of_response(self):
        """A json frame whose <|end_message|> never arrives (truncated output)
        is recovered by reading the payload through the end of the response,
        so the call is not lost. Recovery succeeding also means everything
        else in the response is discarded: visible text before the marker
        must not resurface alongside the recovered call."""
        source = (
            "Sure, let me check."
            + INVOKE_TOOL_JSON
            + '{"name":"weather","args":{"city":"SF"}}'
        )
        result = self.detector.detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})

    def test_unterminated_text_frame_falls_back_to_text(self):
        """A raw-text frame without an end marker cannot be recovered
        (recovery only scans json markers); its body is surfaced as content
        with the framing token stripped."""
        result = self.detector.detect_and_parse(
            INVOKE_TOOL_TEXT + "search the web", self.tools
        )
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "search the web")

    # ==================== detect_and_parse: control-token hygiene ====================

    def test_text_call_body_strips_control_tokens(self):
        """Control tokens inside a raw-text invocation body never leak into
        the surfaced tool arguments."""
        source = INVOKE_TOOL_TEXT + "hello" + CONTENT_THINKING + "world" + END_MESSAGE
        result = self.detector.detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"text": "helloworld"})

    def test_recovery_cleans_control_tokens_before_parsing(self):
        """A control token trailing the JSON defeats the strict parse, but
        recovery strips control tokens from the payload before parsing, so the
        call is still surfaced."""
        source = (
            INVOKE_TOOL_JSON
            + '{"name":"weather","args":{}}'
            + CONTENT_THINKING
            + END_MESSAGE
        )
        result = self.detector.detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {})


class TestInklingDetectorStreamingFrames(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_control_token_split_across_chunks_never_leaks(self):
        """Chunk-boundary atomicity: no matter where the stream is split —
        including inside a framing token or inside a forming message header —
        control tokens are held back until they resolve and never leak into
        the visible text."""
        source = "Hi" + MESSAGE_MODEL + CONTENT_TEXT + "there" + END_MESSAGE
        for split in range(1, len(source)):
            with self.subTest(split=split):
                detector = InklingDetector()
                normal = ""
                calls = []
                for chunk in (source[:split], source[split:]):
                    result = detector.parse_streaming_increment(chunk, self.tools)
                    self.assertNotIn("<|", result.normal_text)
                    normal += result.normal_text
                    calls.extend(result.calls)
                self.assertEqual(normal, "Hithere")
                self.assertEqual(calls, [])

    def test_text_marker_frame_streams_as_a_single_raw_call(self):
        """A raw-text invocation streamed across chunk boundaries (header and
        body split mid-token) emits exactly one call carrying the reassembled
        body, and the forming header text never leaks into the visible output."""
        detector = InklingDetector()
        chunks = [
            MESSAGE_MODEL,
            "weat",
            "her",
            INVOKE_TOOL_TEXT,
            "search the w",
            "eb" + END_MESSAGE,
        ]
        normal = ""
        calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            self.assertNotIn("<|", result.normal_text)
            normal += result.normal_text
            calls.extend(result.calls)
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "")
        self.assertEqual(calls[0].tool_index, 0)
        self.assertEqual(json.loads(calls[0].parameters), {"text": "search the web"})

    def test_incomplete_frame_flushes_visible_prefix(self):
        """A coalesced delta carrying visible text plus a tool frame cut
        mid-args must flush the text immediately: the incomplete-frame hold
        rebinds the buffer to start at the marker, so any visible prefix not
        emitted in that same return is discarded forever."""
        detector = InklingDetector()
        pre = detector.parse_streaming_increment(
            "Sure, let me check."
            + MESSAGE_MODEL
            + "weather"
            + INVOKE_TOOL_JSON
            + '{"name":"wea',
            self.tools,
        )
        self.assertEqual(pre.normal_text, "Sure, let me check.")
        self.assertEqual(pre.calls, [])

        post = detector.parse_streaming_increment(
            'ther","args":{"city":"SF"}}' + END_MESSAGE, self.tools
        )
        self.assertEqual(len(post.calls), 1)
        self.assertEqual(post.calls[0].name, "weather")
        self.assertEqual(json.loads(post.calls[0].parameters), {"city": "SF"})

    def test_recovered_streaming_call_keeps_the_index_sequence(self):
        """A frame the strict pass rejects (NaN is not valid canonical JSON)
        but recovery accepts still streams as a call — and it must carry the
        next free tool_index, not restart at 0: the serving layer keys
        tool_call state by index, so a duplicate index would merge two
        distinct calls' arguments. Once the frame failed to parse strictly,
        everything afterwards is passed through as text — even a later
        well-formed call."""
        detector = InklingDetector()
        first = detector.parse_streaming_increment(
            _canonical_call(args={"city": "SF"}), self.tools
        )
        self.assertEqual([c.tool_index for c in first.calls], [0])

        second = detector.parse_streaming_increment(
            INVOKE_TOOL_JSON + '{"name":"weather","args":{"v":NaN}}' + END_MESSAGE,
            self.tools,
        )
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(second.calls[0].name, "weather")
        self.assertEqual(second.calls[0].tool_index, 1)
        self.assertTrue(math.isnan(json.loads(second.calls[0].parameters)["v"]))

        third = detector.parse_streaming_increment(
            _canonical_call(args={"city": "LA"}), self.tools
        )
        self.assertEqual(third.calls, [])
        self.assertEqual(
            third.normal_text, 'weather{"name":"weather","args":{"city":"LA"}}'
        )

    def test_same_increment_good_then_recoverable_frame_emits_only_recovered(self):
        """All-or-nothing within one delta, mirroring one-shot parsing: when a
        good frame and a strict-failing-but-recoverable frame arrive in the
        same increment, the good frame's call is dropped and only the
        recovered call is emitted, carrying the next index."""
        detector = InklingDetector()
        source = (
            _canonical_call(args={"city": "SF"})
            + INVOKE_TOOL_JSON
            + '{"name":"weather","args":{"v":NaN}}'
            + END_MESSAGE
        )
        result = detector.parse_streaming_increment(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")
        self.assertEqual(result.calls[0].tool_index, 1)
        self.assertTrue(math.isnan(json.loads(result.calls[0].parameters)["v"]))


if __name__ == "__main__":
    unittest.main()
