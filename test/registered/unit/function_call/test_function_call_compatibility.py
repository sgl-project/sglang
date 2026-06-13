"""Unit tests for the function_call compatibility mode (see the ``compatibility`` package).

Covers the fail-open ladder at the FunctionCallParser boundary, the
synthesized JSON close for mid-stream calls, compatibility-event recording, strict
mode, and schema-driven value conversion fallbacks.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.compatibility import (
    CompatibilityEvent,
    CompatibilityMode,
    CompatibilityViolation,
    synthesize_json_close,
)
from sglang.srt.function_call.parsing import GeneratorParser, default_tool_call_output_key
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.minimax_m3_nom import M3TextParser, MinimaxM3NomDetector
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.srt.function_call.compatibility.param_types import FunctionCallParameterDataType
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.tag_format_detector import TagToolCallDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NS = "]<]minimax[>["

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "location": {"type": "object"},
                },
                "required": ["city"],
            },
        ),
    ),
]

M3_FUNCTIONS = {
    "get_weather": {"parameters": TOOLS[0].function.parameters},
}

PLAN_TRIP_TOOL = Tool(
    type="function",
    function=Function(
        name="plan_trip",
        parameters={
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "enum": ["Paris", "Tokyo", "Shanghai"],
                },
                "days": {"type": "integer"},
                "budget": {"type": "number"},
                "travelers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "preferences": {
                                "type": "object",
                                "properties": {
                                    "vegetarian": {"type": "boolean"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "required": ["name"],
                    },
                },
                "itinerary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "integer"},
                            "city": {"type": "string"},
                            "activities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "contact": {"type": ["string", "null"]},
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "transport": {
                    "oneOf": [
                        {"type": "string", "enum": ["train", "flight"]},
                        {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string"},
                                "priority": {"type": "integer"},
                            },
                        },
                    ]
                },
            },
            "required": ["destination", "travelers"],
        },
    ),
)

COMPLEX_TOOLS = TOOLS + [PLAN_TRIP_TOOL]


def _make_parser(
    detector=None, name="minimax-m3-nom", enable_compatibility_mode=True
):
    if detector is not None:
        return FunctionCallParser.with_detector(
            detector, TOOLS, enable_compatibility_mode=enable_compatibility_mode
        )
    parser = FunctionCallParser(
        TOOLS, name, enable_compatibility_mode=enable_compatibility_mode
    )
    return parser


def _stream(parser, text, chunk_size):
    """Feed text in fixed-size chunks; return (normal_text, calls)."""
    normal_text = ""
    calls = []
    for i in range(0, len(text), chunk_size):
        chunk_normal, chunk_calls = parser.parse_stream_chunk(
            text[i : i + chunk_size]
        )
        normal_text += chunk_normal
        calls.extend(chunk_calls)
    # Allow deferred emission, mirroring the serving loop's trailing ticks.
    for _ in range(4):
        chunk_normal, chunk_calls = parser.parse_stream_chunk("")
        if not chunk_normal and not chunk_calls:
            break
        normal_text += chunk_normal
        calls.extend(chunk_calls)
    return normal_text, calls


class TestSynthesizeJsonClose(CustomTestCase):
    def _check(self, partial, expected_value):
        closing = synthesize_json_close(partial)
        self.assertIsNotNone(closing, partial)
        self.assertEqual(json.loads(partial + closing), expected_value)

    def test_closes_partials(self):
        self._check("{", {})
        self._check('{"a": "x', {"a": "x"})
        self._check('{"a": ', {"a": None})
        self._check('{"a": 1, ', {"a": 1, "": None})
        self._check('{"a": [1, ', {"a": [1, None]})
        self._check('{"a": {"b": "c', {"a": {"b": "c"}})
        self._check('{"a": "x\\', {"a": "x\\"})
        self._check('{"a": "say \\"hi', {"a": 'say "hi'})

    def test_already_valid_needs_nothing(self):
        self.assertEqual(synthesize_json_close('{"a": 1}'), "")

    def test_unfixable_returns_none(self):
        # More closers than openers cannot be fixed by appending.
        self.assertIsNone(synthesize_json_close('{"a": 1}}'))


class _MidCallFailingGrammar(GeneratorParser):
    """Emits a tool name and partial arguments, then fails on the char 'X'."""

    def _process(self):
        self._append_delta(
            default_tool_call_output_key(
                0, {"name": "get_weather", "arguments": "{"}
            )
        )
        self._append_delta(default_tool_call_output_key(0, {"arguments": '"city": "Par'}))
        while True:
            ch = yield from self._peek(0)
            if ch == "X":
                raise self._error(expected="not-X", actual=ch, reason="forced failure")
            self._read_one()


class _MidCallFailingDetector(TagToolCallDetector):
    def _make_grammar(self, functions, compatibility):
        return _MidCallFailingGrammar(compatibility=compatibility)

    def has_tool_call(self, text: str) -> bool:
        return True


class TestFailOpenLadder(CustomTestCase):
    """Scope 4: the FunctionCallParser boundary never crashes a request."""

    def test_error_before_tool_call_flushes_raw_text(self):
        """Garbage inside the tool block: everything comes back as content."""
        text = f"Hello {NS}<tool_call>\nGARBAGE no invoke here"
        for chunk_size in (1, 7, len(text)):
            parser = _make_parser()
            normal_text, calls = _stream(parser, text, chunk_size)
            self.assertEqual(calls, [], chunk_size)
            self.assertEqual(normal_text, text, chunk_size)
            # Subsequent chunks keep passing through.
            chunk_normal, _ = parser.parse_stream_chunk(" more")
            self.assertEqual(chunk_normal, " more")
            # The failure is on the audit trail.
            events = [r.event for r in parser.detector.compatibility_records]
            self.assertIn(CompatibilityEvent.FAIL_OPEN, events)

    def test_trailing_text_after_block_is_not_swallowed(self):
        """A complete call followed by trailing text: call survives, text flows."""
        block = (
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>Paris{NS}</city>"
            f"{NS}</invoke>\n"
            f"{NS}</tool_call>"
        )
        text = f"Lead. {block}\nTrailing."
        for chunk_size in (1, 5, len(text)):
            parser = _make_parser()
            normal_text, calls = _stream(parser, text, chunk_size)
            self.assertEqual(normal_text, "Lead. \nTrailing.", chunk_size)
            names = [call.name for call in calls if call.name]
            self.assertEqual(names, ["get_weather"], chunk_size)
            streamed = "".join(call.parameters for call in calls if call.parameters)
            self.assertEqual(json.loads(streamed), {"city": "Paris"})

    def test_mid_call_failure_closes_streamed_arguments(self):
        parser = _make_parser(_MidCallFailingDetector())
        normal1, calls1 = parser.parse_stream_chunk("abc")
        names = [call.name for call in calls1 if call.name]
        self.assertEqual(names, ["get_weather"])

        normal2, calls2 = parser.parse_stream_chunk("X")
        streamed = "".join(
            call.parameters for call in calls1 + calls2 if call.parameters
        )
        # The synthesized close makes the streamed fragments valid JSON.
        self.assertEqual(json.loads(streamed), {"city": "Par"})
        # The failing text is flushed as content, later text passes through.
        self.assertEqual(normal2, "X")
        normal3, calls3 = parser.parse_stream_chunk("after")
        self.assertEqual(normal3, "after")
        self.assertEqual(calls3, [])

    def test_non_stream_error_falls_back_to_full_text(self):
        text = f"Hi {NS}<tool_call>\nnot a valid invoke"
        # Detectors are pure may-raise parsers; recovery is the boundary's.
        detector = MinimaxM3NomDetector()
        with self.assertRaises(Exception):
            detector.detect_and_parse(text, TOOLS)
        parser = FunctionCallParser(TOOLS, "minimax-m3-nom")
        normal_text, calls = parser.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)
        events = [r.event for r in parser.detector.compatibility_records]
        self.assertIn(CompatibilityEvent.FAIL_OPEN, events)

    def test_non_stream_error_salvages_complete_calls(self):
        """Complete calls parsed before the error survive; the tail flows."""
        text = (
            f"Lead. {NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>Paris{NS}</city>"
            f"{NS}</invoke>\n"
            f"{NS}</tool_call>TRAILING GARBAGE"
        )
        parser = FunctionCallParser(TOOLS, "minimax-m3-nom")
        normal_text, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris"})
        self.assertIn("TRAILING GARBAGE", normal_text)


class _RaisingDetector(BaseFormatDetector):
    has_streamed_partial_args = False

    def has_tool_call(self, text: str) -> bool:
        return True

    def detect_and_parse(self, text, tools) -> StreamingParseResult:
        raise RuntimeError("boom")

    def parse_streaming_increment(self, new_text, tools) -> StreamingParseResult:
        if self.has_streamed_partial_args and self.current_tool_id == -1:
            # Simulate a generic detector mid-call: name and partial JSON
            # arguments already on the wire, tracked in base bookkeeping.
            self.current_tool_id = 0
            self.prev_tool_call_arr.append({"name": "get_weather", "arguments": {}})
            self.streamed_args_for_tool.append('{"city": "Par')
            return StreamingParseResult(
                calls=[
                    ToolCallItem(tool_index=0, name="get_weather", parameters=""),
                    ToolCallItem(tool_index=0, parameters='{"city": "Par'),
                ]
            )
        raise RuntimeError("boom")

    def structure_info(self):
        raise NotImplementedError()


class TestGenericDetectorFailOpen(CustomTestCase):
    """The fail-open ladder applies to every detector, not only tag formats."""

    def test_stream_chunk_fails_open_and_latches(self):
        parser = _make_parser(_RaisingDetector())
        self.assertEqual(parser.parse_stream_chunk("chunk one"), ("chunk one", []))
        # Latched: the detector is not called again.
        self.assertEqual(parser.parse_stream_chunk("chunk two"), ("chunk two", []))
        events = [r.event for r in parser.detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.FAIL_OPEN])

    def test_non_stream_fails_open(self):
        parser = _make_parser(_RaisingDetector())
        self.assertEqual(parser.parse_non_stream("full text"), ("full text", []))
        events = [r.event for r in parser.detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.FAIL_OPEN])

    def test_mid_call_close_synthesis_from_base_bookkeeping(self):
        detector = _RaisingDetector()
        detector.has_streamed_partial_args = True
        parser = _make_parser(detector)
        _, calls1 = parser.parse_stream_chunk("first")
        normal2, calls2 = parser.parse_stream_chunk("second")
        streamed = "".join(
            call.parameters for call in calls1 + calls2 if call.parameters
        )
        self.assertEqual(json.loads(streamed), {"city": "Par"})
        self.assertEqual(detector.prev_tool_call_arr[0]["arguments"], {"city": "Par"})
        self.assertEqual(normal2, "second")


def _m3_parser(strict=False):
    return M3TextParser(
        functions=M3_FUNCTIONS,
        compatibility=CompatibilityMode(strict=strict),
    )


class TestCompatibilityEvents(CustomTestCase):
    """Tolerances are named, recorded, and (in strict mode) raise instead."""

    DUPLICATE_TAG_TEXT = (
        f"{NS}<tool_call>\n"
        f'{NS}<invoke name="get_weather">'
        f"{NS}<location>"
        f"{NS}<city>a{NS}</city>{NS}<city>b{NS}</city>"
        f"{NS}</location>"
        f"{NS}</invoke>\n"
        f"{NS}</tool_call>"
    )

    def test_clean_parse_records_nothing(self):
        grammar = _m3_parser()
        grammar.update(
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>Paris{NS}</city>{NS}<days>3{NS}</days>"
            f"{NS}</invoke>\n"
            f"{NS}</tool_call>"
        )
        self.assertEqual(grammar.compatibility.records, [])

    def test_duplicate_tag_collapses_to_list_and_records(self):
        grammar = _m3_parser()
        grammar.update(self.DUPLICATE_TAG_TEXT)
        args = grammar.get_final()["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(args), {"location": {"city": ["a", "b"]}})
        events = [r.event for r in grammar.compatibility.records]
        self.assertEqual(events, [CompatibilityEvent.DUPLICATE_TAG_AS_LIST])

    def test_strict_mode_raises_instead_of_recovering(self):
        grammar = _m3_parser(strict=True)
        with self.assertRaises(CompatibilityViolation) as ctx:
            grammar.update(self.DUPLICATE_TAG_TEXT)
        self.assertEqual(
            ctx.exception.record.event, CompatibilityEvent.DUPLICATE_TAG_AS_LIST
        )

    def test_unconvertible_value_kept_raw_and_recorded(self):
        detector = MinimaxM3NomDetector()
        result = detector.detect_and_parse(
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>Paris{NS}</city>{NS}<days>tomorrow{NS}</days>"
            f"{NS}</invoke>\n"
            f"{NS}</tool_call>",
            TOOLS,
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"city": "Paris", "days": "tomorrow"},
        )
        events = [r.event for r in detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW])

    def test_skipped_garbage_inside_m2_invoke_recovers_and_records(self):
        detector = MinimaxM2Detector()
        result = detector.detect_and_parse(
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            "JUNK that matches no tag\n"
            '<parameter name="city">Paris</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>",
            TOOLS,
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})
        events = [r.event for r in detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.SKIPPED_GARBAGE])

    def test_mixed_text_captured_under_text_key(self):
        grammar = _m3_parser()
        grammar.update(
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<location>"
            f"{NS}<city>a{NS}</city>stray"
            f"{NS}</location>"
            f"{NS}</invoke>\n"
            f"{NS}</tool_call>"
        )
        args = grammar.get_final()["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(
            json.loads(args), {"location": {"city": "a", "$text": "stray"}}
        )
        events = [r.event for r in grammar.compatibility.records]
        self.assertIn(CompatibilityEvent.MIXED_TEXT_CAPTURED, events)


class _BufferedRaisingDetector(BaseFormatDetector):
    """Buffers chunks (emitting nothing), then raises when it sees 'X'."""

    def has_tool_call(self, text: str) -> bool:
        return True

    def detect_and_parse(self, text, tools) -> StreamingParseResult:
        raise RuntimeError("boom")

    def parse_streaming_increment(self, new_text, tools) -> StreamingParseResult:
        self._buffer += new_text
        if "X" in self._buffer:
            raise RuntimeError("boom")
        return StreamingParseResult()

    def structure_info(self):
        raise NotImplementedError()


class TestJsonDetectorCompatibility(CustomTestCase):
    """Compatibility events and fail-open for JSON-wrapper (non-tag) detectors."""

    def test_malformed_json_block_dropped_and_recorded(self):
        detector = Qwen25Detector()
        text = (
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "</tool_call>\n"
            "<tool_call>\n{broken json\n</tool_call>"
        )
        result = detector.detect_and_parse(text, TOOLS)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Paris"}
        )
        events = [r.event for r in detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.MALFORMED_JSON_DROPPED])

    def test_unknown_tool_dropped_and_recorded(self):
        detector = Qwen25Detector()
        text = '<tool_call>\n{"name": "no_such_tool", "arguments": {}}\n</tool_call>'
        result = detector.detect_and_parse(text, TOOLS)
        self.assertEqual(result.calls, [])
        events = [r.event for r in detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.UNKNOWN_TOOL_DROPPED])

    def test_strict_mode_raises_on_unknown_tool(self):
        detector = Qwen25Detector()
        detector.compatibility = CompatibilityMode(strict=True)
        text = '<tool_call>\n{"name": "no_such_tool", "arguments": {}}\n</tool_call>'
        with self.assertRaises(CompatibilityViolation):
            detector.detect_and_parse(text, TOOLS)

    def test_deepseek_malformed_block_dropped_others_survive(self):
        detector = DeepSeekV3Detector()
        good = (
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Paris"}\n```<｜tool▁call▁end｜>'
        )
        bad = (
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n{broken\n```<｜tool▁call▁end｜>"
        )
        text = f"<｜tool▁calls▁begin｜>{good}{bad}<｜tool▁calls▁end｜>"
        result = detector.detect_and_parse(text, TOOLS)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})
        events = [r.event for r in detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.MALFORMED_JSON_DROPPED])

    def test_pythonic_non_literal_args_dropped_and_recorded(self):
        detector = PythonicDetector()
        result = detector.detect_and_parse("[get_weather(city=some_var)]", TOOLS)
        self.assertEqual(result.calls, [])
        events = [r.event for r in detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.MALFORMED_JSON_DROPPED])

    def test_detector_local_malformed_fallback_is_recorded(self):
        detector = HermesDetector()
        text = "<tool_call>not valid json</tool_call>"

        result = detector.detect_and_parse(text, TOOLS)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)
        events = [r.event for r in detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.MALFORMED_JSON_DROPPED])

    def test_detector_local_unknown_tool_drop_is_recorded(self):
        cases = [
            (
                MiMoDetector(),
                "<tool_call><function=no_such_tool>"
                "<parameter=x>1</parameter></function></tool_call>",
            ),
            (
                Apertus2509Detector(),
                '<|tools_prefix|>[{"no_such_tool": {"x": 1}}]<|tools_suffix|>',
            ),
        ]

        for detector, text in cases:
            with self.subTest(detector=type(detector).__name__):
                result = detector.detect_and_parse(text, TOOLS)
                self.assertEqual(result.calls, [])
                events = [r.event for r in detector.compatibility_records]
                self.assertEqual(events, [CompatibilityEvent.UNKNOWN_TOOL_DROPPED])

    def test_detector_local_unknown_tool_strict_fails_open(self):
        text = (
            "<tool_call><function=no_such_tool>"
            "<parameter=x>1</parameter></function></tool_call>"
        )
        parser = _make_parser(MiMoDetector(), enable_compatibility_mode=False)

        normal_text, calls = parser.parse_non_stream(text)

        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)
        events = [r.event for r in parser.detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.FAIL_OPEN])

    def test_streaming_exception_flushes_buffered_text(self):
        """A raising detector loses no buffered text: fail-open flushes it."""
        parser = _make_parser(_BufferedRaisingDetector())
        normal1, calls1 = parser.parse_stream_chunk("hello ")
        self.assertEqual((normal1, calls1), ("", []))
        normal2, calls2 = parser.parse_stream_chunk("X world")
        self.assertEqual(normal2, "hello X world")
        self.assertEqual(calls2, [])
        normal3, _ = parser.parse_stream_chunk(" after")
        self.assertEqual(normal3, " after")
        events = [r.event for r in parser.detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.FAIL_OPEN])


class TestParamTypeConversion(CustomTestCase):
    def _convert(self, value, schema):
        return FunctionCallParameterDataType.from_property(schema).convert(value)

    def test_float_text_is_preserved(self):
        schema = {"type": "number"}
        self.assertEqual(self._convert("3", schema), 3)
        result = self._convert("3.0", schema)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 3.0)
        result = self._convert("1e2", schema)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 100.0)

    def test_integer_schema_unchanged(self):
        schema = {"type": "integer"}
        self.assertEqual(self._convert("5", schema), 5)


class TestCompatibilityModeOption(CustomTestCase):
    """The serving-level switch: compatibility mode heals; default strict fails open."""

    def test_compatibility_mode_heals_default_strict_fails_open(self):
        text = TestCompatibilityEvents.DUPLICATE_TAG_TEXT

        compatible = FunctionCallParser(
            TOOLS, "minimax-m3-nom", enable_compatibility_mode=True
        )
        _, calls = compatible.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0].parameters), {"location": {"city": ["a", "b"]}}
        )
        # The healing is on the audit trail.
        events = [r.event for r in compatible.detector.compatibility_records]
        self.assertEqual(events, [CompatibilityEvent.DUPLICATE_TAG_AS_LIST])

        strict = FunctionCallParser(TOOLS, "minimax-m3-nom")
        normal_text, calls = strict.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)
        events = [r.event for r in strict.detector.compatibility_records]
        self.assertIn(CompatibilityEvent.FAIL_OPEN, events)


ARRAY_TOOLS = TOOLS + [
    Tool(
        type="function",
        function=Function(
            name="set_flags",
            parameters={
                "type": "object",
                "properties": {
                    "flags": {"type": "array", "items": {"type": "string"}},
                },
            },
        ),
    ),
]


class TestRemainingCompatibilityEvents(CustomTestCase):
    """One test per compatibility event not covered elsewhere, plus the
    strict-mode fail-open behavior at the boundary for each."""

    def _events(self, parser):
        return [r.event for r in parser.detector.compatibility_records]

    def test_dropped_invoke_tail(self):
        text = (
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">GARBAGE WITH NO TAG{NS}</invoke>\n'
            f"{NS}</tool_call>"
        )
        parser = _make_parser()
        normal_text, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {})
        self.assertIn(CompatibilityEvent.DROPPED_INVOKE_TAIL, self._events(parser))

        strict = _make_parser(
            FunctionCallParser(TOOLS, "minimax-m3-nom").detector,
            enable_compatibility_mode=False,
        )
        strict.detector.compatibility.strict = True
        normal_text, calls = strict.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)

    def test_mismatched_closing_tag(self):
        text = (
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>Paris{NS}</city>"
            f"{NS}<location>{NS}<lat>1{NS}</wrong>{NS}</location>"
            f"{NS}</invoke>\n{NS}</tool_call>"
        )
        parser = _make_parser()
        _, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertIn(CompatibilityEvent.MISMATCHED_CLOSING_TAG, self._events(parser))

    def test_unclosed_tags_at_end(self):
        text = (
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>Paris{NS}</city>"
            f"{NS}<location>{NS}<lat>1"
            f"{NS}</location>"
            f"{NS}</invoke>\n{NS}</tool_call>"
        )
        parser = _make_parser()
        _, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertIn(CompatibilityEvent.UNCLOSED_TAGS_AT_END, self._events(parser))

    def test_non_item_array_child(self):
        text = (
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="set_flags">'
            f"{NS}<flags>{NS}<item>a{NS}</item>{NS}<element>b{NS}</element>{NS}</flags>"
            f"{NS}</invoke>\n{NS}</tool_call>"
        )
        parser = FunctionCallParser(
            ARRAY_TOOLS, "minimax-m3-nom", enable_compatibility_mode=True
        )
        _, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"flags": ["a", "b"]})
        self.assertIn(CompatibilityEvent.NON_ITEM_ARRAY_CHILD, self._events(parser))

    def test_structure_overrode_schema(self):
        # `city` is string-typed, but the namespace token asserts nesting.
        text = (
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>{NS}<x>1{NS}</x>{NS}</city>"
            f"{NS}</invoke>\n{NS}</tool_call>"
        )
        parser = _make_parser()
        _, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertIn(
            CompatibilityEvent.STRUCTURE_OVERRODE_SCHEMA, self._events(parser)
        )

    def test_invalid_schema_ignored_never_raises_in_strict(self):
        data_type = FunctionCallParameterDataType.get_schema_of_parameter(
            {
                "parameters": {
                    "type": "object",
                    # multipleOf must be > 0: the schema itself is invalid.
                    "properties": {"x": {"type": "integer", "multipleOf": 0}},
                }
            },
            "x",
        )
        mode = CompatibilityMode(strict=True)
        value = data_type.convert("5", compatibility=mode)
        self.assertEqual(value, 5)
        self.assertEqual(
            [r.event for r in mode.records],
            [CompatibilityEvent.INVALID_SCHEMA_IGNORED],
        )

    def test_truncated_call_dropped_identical_in_both_modes(self):
        from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

        text = "Check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nBei"
        for enable_compatibility_mode in (True, False):
            parser = FunctionCallParser(
                TOOLS,
                "qwen3_coder",
                enable_compatibility_mode=enable_compatibility_mode,
            )
            normal_text, calls = parser.parse_non_stream(text)
            self.assertEqual(calls, [], enable_compatibility_mode)
            self.assertEqual(normal_text, text, enable_compatibility_mode)
        compatible = FunctionCallParser(
            TOOLS, "qwen3_coder", enable_compatibility_mode=True
        )
        compatible.parse_non_stream(text)
        self.assertIn(
            CompatibilityEvent.TRUNCATED_CALL_DROPPED, self._events(compatible)
        )

    def test_skipped_non_function_entry(self):
        from sglang.srt.function_call.step3_detector import Step3Detector

        text = (
            "<｜tool_calls_begin｜>\n"
            "<｜tool_call_begin｜>thought<｜tool_sep｜>hmm<｜tool_call_end｜>\n"
            "<｜tool_call_begin｜>function<｜tool_sep｜>"
            '<steptml:invoke name="get_weather">\n'
            '<steptml:parameter name="city">Paris</steptml:parameter>\n'
            "</steptml:invoke><｜tool_call_end｜>\n"
            "<｜tool_calls_end｜>"
        )
        parser = FunctionCallParser(
            TOOLS, "step3", enable_compatibility_mode=True
        )
        _, calls = parser.parse_non_stream(text)
        self.assertEqual([c.name for c in calls], ["get_weather"])
        self.assertIn(
            CompatibilityEvent.SKIPPED_NON_FUNCTION_ENTRY, self._events(parser)
        )

    def test_missing_close_tag(self):
        text = (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n"
            "<parameter=days>\n3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        parser = FunctionCallParser(
            TOOLS, "qwen3_coder", enable_compatibility_mode=True
        )
        _, calls = parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris", "days": 3})
        self.assertIn(CompatibilityEvent.MISSING_CLOSE_TAG, self._events(parser))


class TestComplexSchemaCompatibility(CustomTestCase):
    """Compatibility mode over nested arrays, objects, unions, and conversions."""

    COMPLEX_TEXT = (
        f"{NS}<tool_call>\n"
        f'{NS}<invoke name="plan_trip">'
        f"{NS}<destination>Tokyo{NS}</destination>"
        f"{NS}<days>5{NS}</days>"
        f"{NS}<budget>1250.5{NS}</budget>"
        f"{NS}<travelers>"
        f"{NS}<item>"
        f"{NS}<name>Ada{NS}</name>"
        f"{NS}<age>34{NS}</age>"
        f"{NS}<preferences>"
        f"{NS}<vegetarian>true{NS}</vegetarian>"
        f"{NS}<tags>{NS}<item>museum{NS}</item>{NS}<item>ramen{NS}</item>{NS}</tags>"
        f"{NS}</preferences>"
        f"{NS}</item>"
        f"{NS}<item>"
        f"{NS}<name>Lin{NS}</name>"
        f"{NS}<age>old{NS}</age>"
        f"{NS}</item>"
        f"{NS}</travelers>"
        f"{NS}<itinerary>"
        f"{NS}<item>{NS}<day>1{NS}</day>{NS}<city>Tokyo{NS}</city>"
        f"{NS}<activities>{NS}<item>arrival{NS}</item>{NS}</activities>{NS}</item>"
        f"{NS}<entry>{NS}<day>2{NS}</day>{NS}<city>Tokyo{NS}</city>"
        f"{NS}<activities>{NS}<item>museum{NS}</item>{NS}</activities>{NS}</entry>"
        f"{NS}</itinerary>"
        f"{NS}<contact>null{NS}</contact>"
        f"{NS}<metadata>{NS}<season>spring{NS}</season>{NS}<pace>moderate{NS}</pace>{NS}</metadata>"
        f"{NS}<transport>{NS}<mode>train{NS}</mode>{NS}<priority>2{NS}</priority>{NS}</transport>"
        f"{NS}</invoke>\n"
        f"{NS}</tool_call>"
    )

    def _parser(self, enable_compatibility_mode=True):
        return FunctionCallParser(
            COMPLEX_TOOLS,
            "minimax-m3-nom",
            enable_compatibility_mode=enable_compatibility_mode,
        )

    def _events(self, parser):
        return [r.event for r in parser.detector.compatibility_records]

    def test_complex_schema_heals_nested_deviations_and_records(self):
        parser = self._parser(enable_compatibility_mode=True)
        normal_text, calls = parser.parse_non_stream(self.COMPLEX_TEXT)
        self.assertEqual(normal_text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "plan_trip")

        args = json.loads(calls[0].parameters)
        self.assertEqual(args["destination"], "Tokyo")
        self.assertEqual(args["days"], 5)
        self.assertEqual(args["budget"], 1250.5)
        self.assertEqual(args["contact"], None)
        self.assertEqual(args["metadata"], {"season": "spring", "pace": "moderate"})
        self.assertEqual(args["transport"], {"mode": "train", "priority": 2})
        self.assertEqual(args["travelers"][0]["preferences"]["tags"], ["museum", "ramen"])
        self.assertEqual(args["travelers"][1]["age"], "old")
        self.assertEqual(args["itinerary"][1]["day"], 2)

        events = self._events(parser)
        self.assertIn(CompatibilityEvent.NON_ITEM_ARRAY_CHILD, events)
        self.assertIn(CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW, events)

    def test_complex_schema_strict_parser_fails_open(self):
        parser = self._parser(enable_compatibility_mode=False)
        normal_text, calls = parser.parse_non_stream(self.COMPLEX_TEXT)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, self.COMPLEX_TEXT)
        events = self._events(parser)
        self.assertEqual(events, [CompatibilityEvent.FAIL_OPEN])
        self.assertIn(
            CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW.value,
            parser.detector.compatibility_records[0].detail,
        )


class TestTagStreamingGates(CustomTestCase):
    """Streaming-only behaviors of the tag adapter: the unknown-tool gate and
    the fail-open raw-text accounting."""

    def _stream_m2(self, text, chunk_size, enable_compatibility_mode=True):
        parser = FunctionCallParser(
            TOOLS,
            "minimax-m2",
            enable_compatibility_mode=enable_compatibility_mode,
        )
        return parser, _stream(parser, text, chunk_size)

    def test_unknown_tool_dropped_in_streaming_with_dense_indices(self):
        text = (
            "<minimax:tool_call>\n"
            '<invoke name="hack_tool">\n<parameter name="city">x</parameter>\n</invoke>\n'
            '<invoke name="get_weather">\n<parameter name="city">Paris</parameter>\n</invoke>\n'
            "</minimax:tool_call>"
        )
        for chunk_size in (1, 7, len(text)):
            parser, (normal_text, calls) = self._stream_m2(text, chunk_size)
            names = [c.name for c in calls if c.name]
            self.assertEqual(names, ["get_weather"], chunk_size)
            args = "".join(c.parameters for c in calls if c.parameters)
            self.assertEqual(json.loads(args), {"city": "Paris"}, chunk_size)
            # Indices stay dense: the grammar never assigns an ordinal to the
            # dropped call in the first place.
            self.assertEqual({c.tool_index for c in calls}, {0}, chunk_size)
            events = [r.event for r in parser.detector.compatibility_records]
            self.assertIn(CompatibilityEvent.UNKNOWN_TOOL_DROPPED, events)
            # Consistent with the non-streaming path.
            nonstream = FunctionCallParser(
                TOOLS, "minimax-m2", enable_compatibility_mode=True
            )
            _, ns_calls = nonstream.parse_non_stream(text)
            self.assertEqual([c.name for c in ns_calls], ["get_weather"])

    def test_unknown_tool_fails_open_in_strict_streaming(self):
        text = (
            "<minimax:tool_call>\n"
            '<invoke name="hack_tool">\n<parameter name="city">x</parameter>\n</invoke>\n'
            "</minimax:tool_call> after"
        )
        for chunk_size in (1, 9, len(text)):
            parser, (normal_text, calls) = self._stream_m2(
                text, chunk_size, enable_compatibility_mode=False
            )
            self.assertEqual(calls, [], chunk_size)
            # The gate fires at name-parse time, before anything for the call
            # is emitted or committed, so fail-open re-surfaces the full raw
            # text deterministically — independent of chunking.
            self.assertEqual(normal_text, text, chunk_size)
            # Strict-mode convention: the violation raises instead of being
            # recorded; the FAIL_OPEN record carries it in its detail.
            records = parser.detector.compatibility_records
            self.assertEqual([r.event for r in records], [CompatibilityEvent.FAIL_OPEN])
            self.assertIn(CompatibilityEvent.UNKNOWN_TOOL_DROPPED.value, records[0].detail)

    def test_strict_fail_open_after_completed_block_is_not_corrupted(self):
        # Regression: emitted content is not a contiguous prefix of the raw
        # stream once a no-call block completes; fail-open used to duplicate
        # text and clip markers here.
        text = (
            "A<minimax:tool_call> </minimax:tool_call>"
            'B<minimax:tool_call> junk <invoke name="get_weather">'
        )
        for chunk_size in (1, 5, len(text)):
            parser, (normal_text, calls) = self._stream_m2(
                text, chunk_size, enable_compatibility_mode=False
            )
            self.assertEqual(calls, [], chunk_size)
            self.assertEqual(
                normal_text,
                'AB<minimax:tool_call> junk <invoke name="get_weather">',
                chunk_size,
            )


if __name__ == "__main__":
    unittest.main()
