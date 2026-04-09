"""Unit tests for DeepSeekV31Detector — no server, no model loading.

Covers the DeepSeek V3.1 function-call format:

    <｜tool▁calls▁begin｜>
      <｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json_args}<｜tool▁call▁end｜>
      ...
    <｜tool▁calls▁end｜>

which differs from V3 (no ``function`` literal, no ```json code fence).
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(5, "stage-a-test-cpu")


BOT = "<｜tool▁calls▁begin｜>"
EOT = "<｜tool▁calls▁end｜>"
CALL_BEGIN = "<｜tool▁call▁begin｜>"
CALL_END = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


def _wrap_single(name: str, args_json: str) -> str:
    return f"{BOT}{CALL_BEGIN}{name}{SEP}{args_json}{CALL_END}{EOT}"


def _make_tools() -> list:
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information for a city.",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="search_web",
                description="Search the web.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]


class TestDeepSeekV31DetectorHasToolCall(CustomTestCase):
    def setUp(self):
        self.detector = DeepSeekV31Detector()

    def test_positive(self):
        text = f"some prefix {BOT}{CALL_BEGIN}get_weather{SEP}{{}}{CALL_END}{EOT}"
        self.assertTrue(self.detector.has_tool_call(text))

    def test_negative_plain_text(self):
        self.assertFalse(self.detector.has_tool_call("just a normal response"))

    def test_negative_inner_begin_token_only(self):
        # `has_tool_call` looks for the OUTER begin token (`<｜tool▁calls▁begin｜>`),
        # not the inner per-call token. A raw inner token alone must not trigger.
        text = f"{CALL_BEGIN}get_weather{SEP}{{}}{CALL_END}"
        self.assertFalse(self.detector.has_tool_call(text))


class TestDeepSeekV31DetectorDetectAndParse(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = DeepSeekV31Detector()

    def test_no_tool_call_returns_text_as_normal(self):
        text = "Hello, this is a plain answer."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertIsInstance(result, StreamingParseResult)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_single_tool_call(self):
        text = _wrap_single("get_weather", '{"city": "Tokyo"}')
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        call = result.calls[0]
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.tool_index, 0)
        self.assertEqual(json.loads(call.parameters), {"city": "Tokyo"})

    def test_text_prefix_is_stripped_and_preserved(self):
        prefix = "Sure, I'll check.   "
        text = prefix + _wrap_single("get_weather", '{"city": "Paris"}')
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, prefix.strip())
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_multiple_tool_calls(self):
        # Call tools in REVERSE order of the tools list. This is critical:
        # if tool_index were (buggily) assigned from call order instead of the
        # tools-list position, indices would come out as [0, 1] and a same-order
        # test would pass despite the bug. Reversing forces [1, 0].
        body = (
            f"{CALL_BEGIN}search_web{SEP}"
            f'{{"query": "hotels"}}{CALL_END}'
            f"{CALL_BEGIN}get_weather{SEP}"
            f'{{"city": "Tokyo"}}{CALL_END}'
        )
        text = f"{BOT}{body}{EOT}"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls[0].name, "search_web")
        self.assertEqual(result.calls[0].tool_index, 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"query": "hotels"})
        self.assertEqual(result.calls[1].name, "get_weather")
        self.assertEqual(result.calls[1].tool_index, 0)
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "Tokyo"})

    def test_invalid_json_falls_back_to_raw_text(self):
        # Malformed JSON inside the args → exception is caught and the whole
        # text is returned as normal_text with no calls.
        text = _wrap_single("get_weather", '{"city": "Tokyo"')  # missing brace
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)

    def test_unknown_tool_is_skipped(self):
        # parse_base_json skips calls whose `name` is not in the tools list
        # (default env: SGLANG_FORWARD_UNKNOWN_TOOLS is off). The raw text of
        # the unknown call must also be swallowed — not leaked into normal_text.
        text = _wrap_single("nonexistent_tool", '{"x": 1}')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "")

    def test_unicode_arguments_preserved(self):
        # JSON preserves non-ASCII content through the parse round-trip.
        text = _wrap_single("search_web", '{"query": "东京 天气"}')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"query": "东京 天气"})


class TestDeepSeekV31DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = DeepSeekV31Detector()

    def _feed(self, chunks):
        """Feed chunks to the streaming parser and flatten emitted calls."""
        all_calls = []
        normal_texts = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                all_calls.extend(result.calls)
            if result.normal_text:
                normal_texts.append(result.normal_text)
        return all_calls, normal_texts

    def test_plain_text_passthrough(self):
        result = self.detector.parse_streaming_increment(
            "just text, no tool call", self.tools
        )
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "just text, no tool call")

    def test_plain_text_strips_stray_end_tokens(self):
        # When there's no tool call context, stray end tokens in the chunk
        # must be scrubbed from the emitted normal_text (this is the detector's
        # cleanup path for trailing `<｜tool▁calls▁end｜>` after content).
        result = self.detector.parse_streaming_increment(
            f"all done{EOT}{CALL_END}", self.tools
        )
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "all done")

    def test_single_tool_call_name_precedes_arguments(self):
        # After the first chunk delivering the begin-sep sequence, the detector
        # must emit a name-only event (parameters="") before any argument bytes.
        chunks = [
            f"{BOT}{CALL_BEGIN}get_weather{SEP}",
            '{"city":',
            ' "Tokyo"}',
            CALL_END + EOT,
        ]
        calls, _ = self._feed(chunks)

        # First emitted call must carry the name with empty parameters.
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].parameters, "")
        self.assertEqual(calls[0].tool_index, 0)

        # Subsequent emissions must be anonymous argument diffs.
        arg_calls = [c for c in calls[1:] if c.parameters]
        self.assertTrue(all(c.name is None for c in arg_calls))
        # And their concatenation must reconstruct the full JSON string.
        reconstructed = "".join(c.parameters for c in arg_calls)
        self.assertEqual(json.loads(reconstructed), {"city": "Tokyo"})

    def test_streaming_state_advances_after_completion(self):
        # After a tool call completes, internal state must reset so the next
        # call starts fresh: tool id advances, _last_arguments clears,
        # current_tool_name_sent goes back to False.
        chunks = [
            f"{BOT}{CALL_BEGIN}get_weather{SEP}",
            '{"city": "Tokyo"}',
            CALL_END,
        ]
        self._feed(chunks)

        self.assertEqual(self.detector.current_tool_id, 1)
        self.assertEqual(self.detector._last_arguments, "")
        self.assertFalse(self.detector.current_tool_name_sent)

    def test_streaming_two_sequential_tool_calls(self):
        chunks = [
            BOT,
            f"{CALL_BEGIN}get_weather{SEP}",
            '{"city": "Tokyo"}',
            CALL_END,
            f"{CALL_BEGIN}search_web{SEP}",
            '{"query": "sushi"}',
            CALL_END + EOT,
        ]
        calls, _ = self._feed(chunks)

        name_events = [c for c in calls if c.name]
        self.assertEqual([c.name for c in name_events], ["get_weather", "search_web"])
        # Both tools end up with distinct tool_index values.
        self.assertEqual(name_events[0].tool_index, 0)
        self.assertEqual(name_events[1].tool_index, 1)

        # Arg diffs for the first tool should reconstruct its JSON.
        first_arg_diffs = [
            c.parameters
            for c in calls
            if c.name is None and c.tool_index == 0 and c.parameters
        ]
        second_arg_diffs = [
            c.parameters
            for c in calls
            if c.name is None and c.tool_index == 1 and c.parameters
        ]
        self.assertEqual(json.loads("".join(first_arg_diffs)), {"city": "Tokyo"})
        self.assertEqual(json.loads("".join(second_arg_diffs)), {"query": "sushi"})

    def test_streaming_argument_diffs_are_incremental(self):
        # Diffs must be strict suffixes of what's been streamed so far, never
        # overlapping and never skipping bytes.
        chunks = [
            f"{BOT}{CALL_BEGIN}get_weather{SEP}",
            '{"ci',
            'ty": "Pa',
            'ris"}',
            CALL_END,
        ]
        calls, _ = self._feed(chunks)

        arg_diffs = [c.parameters for c in calls if c.name is None and c.parameters]
        # Exact diffs — one per mid-JSON chunk, with the final chunk triggering
        # completion. Verifying the exact list (not just reconstruction) catches
        # any off-by-one in the `startswith(_last_arguments)` prefix math.
        self.assertEqual(arg_diffs, ['{"ci', 'ty": "Pa', 'ris"}'])
        self.assertEqual(json.loads("".join(arg_diffs)), {"city": "Paris"})


class TestDeepSeekV31DetectorStructureInfo(CustomTestCase):
    def test_structure_info_shape(self):
        detector = DeepSeekV31Detector()
        info_fn = detector.structure_info()
        info = info_fn("get_weather")

        self.assertEqual(info.begin, f"{CALL_BEGIN}get_weather{SEP}")
        self.assertEqual(info.end, CALL_END)
        self.assertEqual(info.trigger, CALL_BEGIN)

    def test_supports_structural_tag(self):
        # Inherited from BaseFormatDetector; confirm V31 hasn't disabled it,
        # since the serving layer uses this to gate constrained generation.
        self.assertTrue(DeepSeekV31Detector().supports_structural_tag())


if __name__ == "__main__":
    unittest.main()
