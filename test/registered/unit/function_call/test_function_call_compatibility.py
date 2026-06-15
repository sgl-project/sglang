"""Unit tests for the function_call compatibility mode (see the ``compatibility`` package).

Covers the fail-open ladder at the FunctionCallParser boundary, the
synthesized JSON close for mid-stream calls, compatibility-event recording, strict
mode, and schema-driven value conversion fallbacks.
"""

import json
import unittest
from dataclasses import dataclass

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.compatibility import (
    CompatibilityContext,
    CompatibilityEvent,
    CompatibilityViolation,
    synthesize_json_close,
)
from sglang.srt.function_call.compatibility.param_types import (
    FunctionCallParameterDataType,
)
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.minimax_m3 import M3TextParser, MinimaxM3Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.parsing import (
    GeneratorParser,
    default_tool_call_output_key,
)
from sglang.srt.function_call.poolside_v1_detector import PoolsideV1Detector
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

BOOLEAN_TOOLS = TOOLS + [
    Tool(
        type="function",
        function=Function(
            name="set_enabled",
            parameters={
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
            },
        ),
    )
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


def _make_parser(detector=None, name="minimax-m3", enable_compatibility_mode=True):
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
        chunk_normal, chunk_calls = parser.parse_stream_chunk(text[i : i + chunk_size])
        normal_text += chunk_normal
        calls.extend(chunk_calls)
    return _drain_stream(parser, normal_text, calls, ticks=4)


def _drain_stream(parser, normal_text="", calls=None, ticks=8):
    """Allow deferred emission, mirroring the serving loop's trailing ticks."""
    if calls is None:
        calls = []
    for _ in range(ticks):
        chunk_normal, chunk_calls = parser.parse_stream_chunk("")
        if not chunk_normal and not chunk_calls:
            break
        normal_text += chunk_normal
        calls.extend(chunk_calls)
    return normal_text, calls


def _stream_chunks(parser, chunks, ticks=8):
    normal_text = ""
    calls = []
    for chunk in chunks:
        chunk_normal, chunk_calls = parser.parse_stream_chunk(chunk)
        normal_text += chunk_normal
        calls.extend(chunk_calls)
    return _drain_stream(parser, normal_text, calls, ticks=ticks)


@dataclass(frozen=True)
class ParserTextCase:
    parser_name: str
    text: str


@dataclass(frozen=True)
class ParserChunksCase:
    parser_name: str
    case_name: str
    chunks: tuple
    expected_event: CompatibilityEvent


@dataclass(frozen=True)
class ParserNamedTextCase:
    parser_name: str
    case_name: str
    text: str
    expected_event: CompatibilityEvent


@dataclass(frozen=True)
class ParserDropCase(ParserTextCase):
    expected_event: CompatibilityEvent
    expected_normal_text: str = ""


@dataclass(frozen=True)
class DetectorTextCase:
    detector_factory: object
    text: str


@dataclass(frozen=True)
class DetectorInstanceCase:
    detector: object
    text: str


@dataclass(frozen=True)
class DetectorArgsCase(DetectorInstanceCase):
    expected_arguments: object


NONSTREAM_DROP_CASES = (
    ParserDropCase(
        "qwen25",
        '<tool_call>\n{"name": "no_such_tool", "arguments": {}}\n</tool_call>',
        CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
        "",
    ),
    ParserDropCase(
        "hermes",
        "<tool_call>{broken}</tool_call>",
        CompatibilityEvent.MALFORMED_JSON_DROPPED,
        "<tool_call>{broken}</tool_call>",
    ),
    ParserDropCase(
        "minimax-m3",
        f"{NS}<tool_call>\n"
        f'{NS}<invoke name="no_such_tool">'
        f"{NS}<city>Paris{NS}</city>"
        f"{NS}</invoke>\n"
        f"{NS}</tool_call>",
        CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
        "",
    ),
    ParserDropCase(
        "mimo",
        "<tool_call><function=no_such_tool>"
        "<parameter=city>Paris</parameter>"
        "</function></tool_call>",
        CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
        "",
    ),
    ParserDropCase(
        "minicpm5",
        '<function name="no_such_tool">'
        '<param name="city">Paris</param>'
        "</function>",
        CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
        '<function name="no_such_tool">'
        '<param name="city">Paris</param>'
        "</function>",
    ),
    ParserDropCase(
        "gigachat3",
        'function call<|role_sep|>\n{"name":"no_such_tool",' '"arguments":{}}</s>',
        CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
        "",
    ),
    ParserDropCase(
        "kimi_k2",
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0"
        "<|tool_call_argument_begin|>{broken<|tool_call_end|>"
        "<|tool_calls_section_end|>",
        CompatibilityEvent.MALFORMED_JSON_DROPPED,
        "",
    ),
    ParserDropCase(
        "gemma4",
        "<|tool_call>not-a-call<tool_call|>",
        CompatibilityEvent.MALFORMED_JSON_DROPPED,
        "",
    ),
    ParserDropCase(
        "hunyuan",
        "<tool_calls>not-a-call</tool_calls>",
        CompatibilityEvent.MALFORMED_JSON_DROPPED,
        "",
    ),
)


class CompatibilityTestCase(CustomTestCase):
    def records_of(self, owner):
        if hasattr(owner, "detector"):
            return owner.detector.compatibility_records
        if hasattr(owner, "compatibility_records"):
            return owner.compatibility_records
        if hasattr(owner, "compatibility"):
            return owner.compatibility.records
        return owner.records

    def events_of(self, owner):
        return [record.event for record in self.records_of(owner)]

    def assert_events_equal(self, owner, expected):
        self.assertEqual(self.events_of(owner), list(expected))

    def assert_event_in(self, owner, event):
        self.assertIn(event, self.events_of(owner))

    def assert_fail_open_with_detail(self, owner, inner_event):
        records = self.records_of(owner)
        self.assertEqual(
            [record.event for record in records], [CompatibilityEvent.FAIL_OPEN]
        )
        self.assertIn(inner_event.value, records[0].detail)

    def call_names(self, calls):
        return [call.name for call in calls if call.name]

    def streamed_parameters(self, calls):
        return "".join(call.parameters for call in calls if call.parameters)

    def streamed_parameters_by_tool(self, calls):
        grouped = {}
        for call in calls:
            if call.parameters:
                grouped.setdefault(call.tool_index, "")
                grouped[call.tool_index] += call.parameters
        return {idx: json.loads(params) for idx, params in grouped.items()}

    def assert_single_call(self, calls, *, name=None, arguments=None, tool_index=None):
        self.assertEqual(len(calls), 1)
        call = calls[0]
        if name is not None:
            self.assertEqual(call.name, name)
        if tool_index is not None:
            self.assertEqual(call.tool_index, tool_index)
        if arguments is not None:
            self.assertEqual(json.loads(call.parameters), arguments)
        return call

    def assert_streamed_arguments(self, calls, expected):
        self.assertEqual(json.loads(self.streamed_parameters(calls)), expected)


class TestSynthesizeJsonClose(CompatibilityTestCase):
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
            default_tool_call_output_key(0, {"name": "get_weather", "arguments": "{"})
        )
        self._append_delta(
            default_tool_call_output_key(0, {"arguments": '"city": "Par'})
        )
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


class TestFailOpenLadder(CompatibilityTestCase):
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
            self.assert_event_in(parser, CompatibilityEvent.FAIL_OPEN)

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
            self.assertEqual(self.call_names(calls), ["get_weather"], chunk_size)
            self.assert_streamed_arguments(calls, {"city": "Paris"})

    def test_mid_call_failure_closes_streamed_arguments(self):
        parser = _make_parser(_MidCallFailingDetector())
        normal1, calls1 = parser.parse_stream_chunk("abc")
        self.assertEqual(self.call_names(calls1), ["get_weather"])

        normal2, calls2 = parser.parse_stream_chunk("X")
        # The synthesized close makes the streamed fragments valid JSON.
        self.assert_streamed_arguments(calls1 + calls2, {"city": "Par"})
        # The failing text is flushed as content, later text passes through.
        self.assertEqual(normal2, "X")
        normal3, calls3 = parser.parse_stream_chunk("after")
        self.assertEqual(normal3, "after")
        self.assertEqual(calls3, [])

    def test_non_stream_error_falls_back_to_full_text(self):
        text = f"Hi {NS}<tool_call>\nnot a valid invoke"
        # Detectors are pure may-raise parsers; recovery is the boundary's.
        detector = MinimaxM3Detector()
        with self.assertRaises(Exception):
            detector.detect_and_parse(text, TOOLS)
        parser = FunctionCallParser(TOOLS, "minimax-m3")
        normal_text, calls = parser.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)
        self.assert_event_in(parser, CompatibilityEvent.FAIL_OPEN)

    def test_non_stream_error_salvages_complete_calls(self):
        """Complete calls parsed before the error survive; the tail flows."""
        text = (
            f"Lead. {NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>Paris{NS}</city>"
            f"{NS}</invoke>\n"
            f"{NS}</tool_call>TRAILING GARBAGE"
        )
        parser = FunctionCallParser(TOOLS, "minimax-m3")
        normal_text, calls = parser.parse_non_stream(text)
        self.assert_single_call(calls, name="get_weather", arguments={"city": "Paris"})
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


class TestGenericDetectorFailOpen(CompatibilityTestCase):
    """The fail-open ladder applies to every detector, not only tag formats."""

    def test_stream_chunk_fails_open_and_latches(self):
        parser = _make_parser(_RaisingDetector())
        self.assertEqual(parser.parse_stream_chunk("chunk one"), ("chunk one", []))
        # Latched: the detector is not called again.
        self.assertEqual(parser.parse_stream_chunk("chunk two"), ("chunk two", []))
        self.assert_events_equal(parser, [CompatibilityEvent.FAIL_OPEN])

    def test_non_stream_fails_open(self):
        parser = _make_parser(_RaisingDetector())
        self.assertEqual(parser.parse_non_stream("full text"), ("full text", []))
        self.assert_events_equal(parser, [CompatibilityEvent.FAIL_OPEN])

    def test_mid_call_close_synthesis_from_base_bookkeeping(self):
        detector = _RaisingDetector()
        detector.has_streamed_partial_args = True
        parser = _make_parser(detector)
        _, calls1 = parser.parse_stream_chunk("first")
        normal2, calls2 = parser.parse_stream_chunk("second")
        self.assert_streamed_arguments(calls1 + calls2, {"city": "Par"})
        self.assertEqual(detector.prev_tool_call_arr[0]["arguments"], {"city": "Par"})
        self.assertEqual(normal2, "second")


def _m3_parser(strict=False):
    return M3TextParser(
        functions=M3_FUNCTIONS,
        compatibility=CompatibilityContext(strict=strict),
    )


class TestCompatibilityEvents(CompatibilityTestCase):
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
        self.assert_events_equal(grammar, [CompatibilityEvent.DUPLICATE_TAG_AS_LIST])

    def test_strict_mode_raises_instead_of_recovering(self):
        grammar = _m3_parser(strict=True)
        with self.assertRaises(CompatibilityViolation) as ctx:
            grammar.update(self.DUPLICATE_TAG_TEXT)
        self.assertEqual(
            ctx.exception.record.event, CompatibilityEvent.DUPLICATE_TAG_AS_LIST
        )

    def test_unconvertible_value_kept_raw_and_recorded(self):
        detector = MinimaxM3Detector()
        result = detector.detect_and_parse(
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<city>Paris{NS}</city>{NS}<days>tomorrow{NS}</days>"
            f"{NS}</invoke>\n"
            f"{NS}</tool_call>",
            TOOLS,
        )
        self.assert_single_call(
            result.calls,
            arguments={"city": "Paris", "days": "tomorrow"},
        )
        self.assert_events_equal(
            detector, [CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW]
        )

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
        self.assert_single_call(result.calls, arguments={"city": "Paris"})
        self.assert_events_equal(detector, [CompatibilityEvent.SKIPPED_GARBAGE])

    def test_custom_streaming_fsm_skipped_garbage_records(self):
        cases = [
            DetectorArgsCase(
                PoolsideV1Detector(),
                "<tool_call>get_weather\n"
                "junk<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                "</tool_call>",
                {"city": "Paris"},
            ),
            DetectorArgsCase(
                Gemma4Detector(),
                '<|tool_call>junkcall:get_weather{city:<|"|>Paris<|"|>}' "<tool_call|>",
                {"city": "Paris"},
            ),
        ]

        for case in cases:
            with self.subTest(detector=type(case.detector).__name__):
                parser = _make_parser(case.detector)

                normal_text, calls = parser.parse_stream_chunk(case.text)

                self.assertEqual(normal_text, "")
                self.assertEqual(calls[0].name, "get_weather")
                self.assert_streamed_arguments(calls, case.expected_arguments)
                self.assert_event_in(parser, CompatibilityEvent.SKIPPED_GARBAGE)

    def test_custom_streaming_fsm_skipped_garbage_strict_fails_open(self):
        cases = [
            DetectorInstanceCase(
                PoolsideV1Detector(),
                "<tool_call>get_weather\n"
                "junk<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                "</tool_call>",
            ),
            DetectorInstanceCase(
                Gemma4Detector(),
                '<|tool_call>junkcall:get_weather{city:<|"|>Paris<|"|>}' "<tool_call|>",
            ),
        ]

        for case in cases:
            with self.subTest(detector=type(case.detector).__name__):
                parser = _make_parser(case.detector, enable_compatibility_mode=False)

                normal_text, calls = parser.parse_stream_chunk(case.text)

                self.assertEqual(calls, [])
                self.assertEqual(normal_text, case.text)
                self.assert_fail_open_with_detail(
                    parser, CompatibilityEvent.SKIPPED_GARBAGE
                )

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
        self.assert_event_in(grammar, CompatibilityEvent.MIXED_TEXT_CAPTURED)


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


class TestJsonDetectorCompatibility(CompatibilityTestCase):
    """Compatibility events and fail-open for JSON-wrapper (non-tag) detectors."""

    def test_nonstream_compatibility_drop_records_parser_fallback(self):
        for case in NONSTREAM_DROP_CASES:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = parser.parse_non_stream(case.text)

                self.assertEqual(normal_text, case.expected_normal_text)
                self.assertEqual(calls, [])
                self.assert_events_equal(parser, [case.expected_event])

    def test_nonstream_compatibility_drop_strict_fails_open(self):
        for case in NONSTREAM_DROP_CASES:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=False
                )

                normal_text, calls = parser.parse_non_stream(case.text)

                self.assertEqual(calls, [])
                self.assertEqual(normal_text, case.text)
                self.assert_fail_open_with_detail(parser, case.expected_event)

    def test_deepseek_malformed_block_dropped_others_survive(self):
        cases = [
            (
                DeepSeekV3Detector(),
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁calls▁end｜>",
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
                '```json\n{"city": "Paris"}\n```<｜tool▁call▁end｜>',
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
                "```json\n{broken\n```<｜tool▁call▁end｜>",
            ),
            (
                DeepSeekV31Detector(),
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁calls▁end｜>",
                "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
                '{"city": "Paris"}<｜tool▁call▁end｜>',
                "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
                "{broken<｜tool▁call▁end｜>",
            ),
            (
                DeepSeekV32Detector(),
                "<｜DSML｜function_calls>",
                "</｜DSML｜function_calls>",
                '<｜DSML｜invoke name="get_weather">'
                '{"city": "Paris"}</｜DSML｜invoke>',
                '<｜DSML｜invoke name="get_weather">{broken</｜DSML｜invoke>',
            ),
            (
                DeepSeekV4Detector(),
                "<｜DSML｜tool_calls>",
                "</｜DSML｜tool_calls>",
                '<｜DSML｜invoke name="get_weather">'
                '{"city": "Paris"}</｜DSML｜invoke>',
                '<｜DSML｜invoke name="get_weather">{broken</｜DSML｜invoke>',
            ),
        ]

        for detector, begin, end, good, bad in cases:
            with self.subTest(detector=type(detector).__name__):
                result = detector.detect_and_parse(f"{begin}{good}{bad}{end}", TOOLS)

                self.assert_single_call(result.calls, arguments={"city": "Paris"})
                self.assert_events_equal(
                    detector, [CompatibilityEvent.MALFORMED_JSON_DROPPED]
                )

    def _unknown_tool_stream_cases(self):
        return [
            DetectorTextCase(
                DeepSeekV3Detector,
                "<｜tool▁calls▁begin｜>"
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>no_such_tool\n"
                "```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            ),
            DetectorTextCase(
                DeepSeekV31Detector,
                "<｜tool▁calls▁begin｜>"
                "<｜tool▁call▁begin｜>no_such_tool<｜tool▁sep｜>{}"
                "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            ),
            DetectorTextCase(
                DeepSeekV32Detector,
                "<｜DSML｜function_calls>"
                '<｜DSML｜invoke name="no_such_tool">{}</｜DSML｜invoke>'
                "</｜DSML｜function_calls>",
            ),
            DetectorTextCase(
                DeepSeekV4Detector,
                "<｜DSML｜tool_calls>"
                '<｜DSML｜invoke name="no_such_tool">{}</｜DSML｜invoke>'
                "</｜DSML｜tool_calls>",
            ),
            DetectorTextCase(
                Gemma4Detector, "<|tool_call>call:no_such_tool{}<tool_call|>"
            ),
            DetectorTextCase(
                GigaChat3Detector,
                'function call<|role_sep|>\n{"name": "no_such_tool", '
                '"arguments": {}}</s>',
            ),
            DetectorTextCase(Glm4MoeDetector, "<tool_call>no_such_tool\n</tool_call>"),
            DetectorTextCase(Glm47MoeDetector, "<tool_call>no_such_tool</tool_call>"),
            DetectorTextCase(
                KimiK2Detector,
                "<|tool_calls_section_begin|>"
                "<|tool_call_begin|>functions.no_such_tool:0"
                "<|tool_call_argument_begin|>{}<|tool_call_end|>"
                "<|tool_calls_section_end|>",
            ),
            DetectorTextCase(MistralDetector, "[TOOL_CALLS]no_such_tool[ARGS]{}"),
            DetectorTextCase(
                PoolsideV1Detector,
                "<tool_call>no_such_tool\n"
                "<arg_key>x</arg_key><arg_value>1</arg_value>"
                "</tool_call>",
            ),
        ]

    def test_unknown_tool_streaming_recorded_across_custom_parsers(self):
        for case in self._unknown_tool_stream_cases():
            with self.subTest(detector=case.detector_factory.__name__):
                parser = FunctionCallParser.with_detector(
                    case.detector_factory(),
                    TOOLS,
                    enable_compatibility_mode=True,
                )

                normal_text, calls = parser.parse_stream_chunk(case.text)

                self.assertEqual(normal_text, "")
                self.assertEqual(calls, [])
                self.assert_events_equal(
                    parser, [CompatibilityEvent.UNKNOWN_TOOL_DROPPED]
                )

    def test_unknown_tool_streaming_strict_fails_open(self):
        for case in self._unknown_tool_stream_cases():
            with self.subTest(detector=case.detector_factory.__name__):
                parser = FunctionCallParser.with_detector(
                    case.detector_factory(),
                    TOOLS,
                    enable_compatibility_mode=False,
                )

                normal_text, calls = parser.parse_stream_chunk(case.text)

                self.assertEqual(calls, [])
                self.assertEqual(normal_text, case.text)
                self.assert_fail_open_with_detail(
                    parser, CompatibilityEvent.UNKNOWN_TOOL_DROPPED
                )

    def test_streaming_unknown_then_known_preserves_following_call(self):
        cases = [
            ParserTextCase(
                "qwen25",
                '<tool_call>\n{"name":"no_such_tool","arguments":{"city":"Ignored"}}\n'
                '</tool_call><tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n'
                "</tool_call>",
            ),
            ParserTextCase(
                "hermes",
                '<tool_call>{"name":"no_such_tool","arguments":{"city":"Ignored"}}</tool_call>'
                '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
            ),
            ParserTextCase(
                "llama3",
                '<|python_tag|>{"name":"no_such_tool","arguments":{}};'
                '{"name":"get_weather","arguments":{"city":"Paris"}}',
            ),
            ParserTextCase(
                "mistral",
                '[TOOL_CALLS] [{"name":"no_such_tool","arguments":{"city":"Ignored"}},'
                '{"name":"get_weather","arguments":{"city":"Paris"}}]',
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, [case.text])

                self.assertEqual(normal_text, "")
                self.assertEqual(self.call_names(calls), ["get_weather"])
                self.assert_streamed_arguments(calls, {"city": "Paris"})
                self.assert_events_equal(
                    parser, [CompatibilityEvent.UNKNOWN_TOOL_DROPPED]
                )

    def test_streaming_split_separator_after_dropped_call_preserves_next_call(self):
        cases = [
            ParserChunksCase(
                "mistral",
                "unknown",
                (
                    '[TOOL_CALLS] [{"name":"no_such_tool","arguments":{}}',
                    ",",
                    '{"name":"get_weather","arguments":{"city":"Paris"}}]',
                ),
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
            ParserChunksCase(
                "mistral",
                "malformed",
                (
                    "[TOOL_CALLS] [{broken}",
                    ",",
                    '{"name":"get_weather","arguments":{"city":"Paris"}}]',
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserChunksCase(
                "llama3",
                "unknown",
                (
                    '<|python_tag|>{"name":"no_such_tool","arguments":{}}',
                    ";",
                    '{"name":"get_weather","arguments":{"city":"Paris"}}',
                ),
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
            ParserChunksCase(
                "llama3",
                "malformed",
                (
                    "<|python_tag|>{broken}",
                    ";",
                    '{"name":"get_weather","arguments":{"city":"Paris"}}',
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name, case_name=case.case_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, case.chunks)

                self.assertEqual(normal_text, "")
                self.assertEqual(self.call_names(calls), ["get_weather"])
                self.assert_streamed_arguments(calls, {"city": "Paris"})
                self.assert_events_equal(parser, [case.expected_event])

    def test_streaming_split_eot_after_dropped_wrapped_call_preserves_next_call(self):
        cases = [
            ParserChunksCase(
                "qwen25",
                "unknown",
                (
                    '<tool_call>\n{"name":"no_such_tool","arguments":{}}',
                    "\n</tool_call>",
                    '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>',
                ),
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
            ParserChunksCase(
                "qwen25",
                "malformed",
                (
                    "<tool_call>\n{broken}",
                    "\n</tool_call>",
                    '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>',
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserChunksCase(
                "hermes",
                "unknown",
                (
                    '<tool_call>{"name":"no_such_tool","arguments":{}}',
                    "</tool_call>",
                    '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
                ),
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
            ParserChunksCase(
                "hermes",
                "malformed",
                (
                    "<tool_call>{broken}",
                    "</tool_call>",
                    '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name, case_name=case.case_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, case.chunks)

                self.assertEqual(normal_text, "")
                self.assertEqual(self.call_names(calls), ["get_weather"])
                self.assert_streamed_arguments(calls, {"city": "Paris"})
                self.assert_events_equal(parser, [case.expected_event])

    def test_streaming_malformed_then_known_preserves_following_call(self):
        cases = [
            ParserTextCase(
                "qwen25",
                "<tool_call>\n{broken}\n</tool_call>"
                '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n'
                "</tool_call>",
            ),
            ParserTextCase(
                "hermes",
                "<tool_call>{broken}</tool_call>"
                '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}'
                "</tool_call>",
            ),
            ParserTextCase(
                "llama3",
                "<|python_tag|>{broken};"
                '{"name":"get_weather","arguments":{"city":"Paris"}}',
            ),
            ParserTextCase(
                "mistral",
                '[TOOL_CALLS] [{broken},{"name":"get_weather",'
                '"arguments":{"city":"Paris"}}]',
            ),
            ParserTextCase(
                "apertus2509",
                "<|tools_prefix|>[broken]<|tools_suffix|>"
                '<|tools_prefix|>[{"get_weather":{"city":"Paris"}}]<|tools_suffix|>',
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, [case.text])

                self.assertEqual(normal_text, "")
                self.assertEqual(self.call_names(calls), ["get_weather"])
                self.assert_streamed_arguments(calls, {"city": "Paris"})
                self.assert_events_equal(
                    parser, [CompatibilityEvent.MALFORMED_JSON_DROPPED]
                )

    def test_streaming_bad_middle_call_preserves_dense_indices(self):
        cases = [
            ParserNamedTextCase(
                "mistral",
                "unknown",
                '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"London"}},'
                '{"name":"no_such_tool","arguments":{}},'
                '{"name":"get_weather","arguments":{"city":"Paris"}}]',
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
            ParserNamedTextCase(
                "llama3",
                "unknown",
                '<|python_tag|>{"name":"get_weather","arguments":{"city":"London"}};'
                '{"name":"no_such_tool","arguments":{}};'
                '{"name":"get_weather","arguments":{"city":"Paris"}}',
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
            ParserNamedTextCase(
                "mistral",
                "malformed",
                '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"London"}},'
                '{broken},{"name":"get_weather","arguments":{"city":"Paris"}}]',
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserNamedTextCase(
                "llama3",
                "malformed",
                '<|python_tag|>{"name":"get_weather","arguments":{"city":"London"}};'
                '{broken};{"name":"get_weather","arguments":{"city":"Paris"}}',
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name, case_name=case.case_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, [case.text])

                self.assertEqual(normal_text, "")
                self.assertEqual(
                    [(call.tool_index, call.name) for call in calls if call.name],
                    [(0, "get_weather"), (1, "get_weather")],
                )
                streamed_args = [
                    json.loads(call.parameters) for call in calls if call.parameters
                ]
                self.assertEqual(streamed_args, [{"city": "London"}, {"city": "Paris"}])
                self.assert_events_equal(parser, [case.expected_event])

    def test_glm_streaming_malformed_arg_tail_closes_valid_json(self):
        cases = [
            ParserTextCase(
                "glm",
                "<tool_call>get_weather\n"
                "<arg_key>city</arg_key></tool_call>"
                "<tool_call>get_weather\n"
                "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
                "</tool_call>",
            ),
            ParserTextCase(
                "glm47",
                "<tool_call>get_weather"
                "<arg_key>city</arg_key></tool_call>"
                "<tool_call>get_weather"
                "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                "</tool_call>",
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, [case.text])

                self.assertEqual(normal_text, "")
                self.assertEqual(self.call_names(calls), ["get_weather", "get_weather"])
                self.assertEqual(
                    self.streamed_parameters_by_tool(calls),
                    {0: {"city": None}, 1: {"city": "Paris"}},
                )
                self.assert_events_equal(
                    parser, [CompatibilityEvent.MALFORMED_JSON_DROPPED]
                )

    def test_streaming_partial_malformed_arg_tail_closes_valid_json(self):
        cases = [
            (
                "deepseekv31",
                (
                    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
                    '<｜tool▁sep｜>{"city": ',
                    "",
                    "<｜tool▁call▁end｜>",
                    "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
                    '{"city":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
                {0: {"city": None}, 1: {"city": "Paris"}},
            ),
            (
                "deepseekv32",
                (
                    '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
                    '{"city": ',
                    '"Par',
                    "}</｜DSML｜invoke>",
                    '<｜DSML｜invoke name="get_weather">{"city":"Paris"}'
                    "</｜DSML｜invoke></｜DSML｜function_calls>",
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
                {0: {"city": None}, 1: {"city": "Paris"}},
            ),
            (
                "deepseekv4",
                (
                    '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
                    '{"city": ',
                    '"Par',
                    "}</｜DSML｜invoke>",
                    '<｜DSML｜invoke name="get_weather">{"city":"Paris"}'
                    "</｜DSML｜invoke></｜DSML｜tool_calls>",
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
                {0: {"city": None}, 1: {"city": "Paris"}},
            ),
            (
                "kimi_k2",
                (
                    "<|tool_calls_section_begin|><|tool_call_begin|>"
                    "functions.get_weather:0<|tool_call_argument_begin|>"
                    '{"city": ',
                    "",
                    "<|tool_call_end|>",
                    "<|tool_call_begin|>functions.get_weather:1"
                    '<|tool_call_argument_begin|>{"city":"Paris"}'
                    "<|tool_call_end|><|tool_calls_section_end|>",
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
                {0: {"city": None}, 1: {"city": "Paris"}},
            ),
            (
                "gigachat3",
                (
                    'function call<|role_sep|>\n{"name":"get_weather",'
                    '"arguments":{"city": ',
                    "}</s>",
                    'function call<|role_sep|>\n{"name":"get_weather",'
                    '"arguments":{"city":"Paris"}}</s>',
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
                {0: {"city": None}, 1: {"city": "Paris"}},
            ),
            (
                "hunyuan",
                (
                    "<tool_calls><tool_call>get_weather<tool_sep>"
                    "<arg_key>city</arg_key><arg_value>Par",
                    "</tool_call>",
                    "<tool_call>get_weather<tool_sep><arg_key>city</arg_key>"
                    "<arg_value>Paris</arg_value></tool_call></tool_calls>",
                ),
                CompatibilityEvent.MISSING_CLOSE_TAG,
                {0: {"city": "Par"}, 1: {"city": "Paris"}},
            ),
            (
                "poolside_v1",
                (
                    "<tool_call>get_weather\n<arg_key>city</arg_key>" "<arg_value>Par",
                    "</tool_call>",
                    "<tool_call>get_weather\n<arg_key>city</arg_key>"
                    "<arg_value>Paris</arg_value></tool_call>",
                ),
                CompatibilityEvent.MISSING_CLOSE_TAG,
                {0: {"city": "Par"}, 1: {"city": "Paris"}},
            ),
        ]

        for parser_name, chunks, expected_event, expected_args in cases:
            with self.subTest(parser_name=parser_name):
                parser = FunctionCallParser(
                    TOOLS, parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, chunks)

                self.assertEqual(normal_text, "")
                self.assertEqual(self.call_names(calls), ["get_weather", "get_weather"])
                self.assertEqual(
                    self.streamed_parameters_by_tool(calls),
                    expected_args,
                )
                self.assert_events_equal(parser, [expected_event])

    def test_streaming_complete_malformed_entries_are_consumed_and_recorded(self):
        cases = [
            ParserTextCase(
                "kimi_k2",
                "<|tool_calls_section_begin|><|tool_call_begin|>badid"
                "<|tool_call_argument_begin|>{}<|tool_call_end|>"
                "<|tool_calls_section_end|>TAIL",
            ),
            ParserTextCase(
                "gigachat3",
                'function call<|role_sep|>\n{"arguments":{}}</s>TAIL',
            ),
            ParserTextCase(
                "gemma4",
                "<|tool_call>call:get_weather<tool_call|>TAIL",
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, [case.text])

                self.assertEqual(normal_text, "TAIL")
                self.assertEqual(calls, [])
                self.assert_events_equal(
                    parser, [CompatibilityEvent.MALFORMED_JSON_DROPPED]
                )

    def test_nonstream_malformed_blocks_preserve_trailing_text(self):
        cases = [
            ParserTextCase(
                "cohere_command4",
                "pre <|START_ACTION|>[{bad}]<|END_ACTION|> post",
            ),
            ParserTextCase("qwen25", "pre <tool_call>\n{bad}\n</tool_call> post"),
            ParserTextCase(
                "deepseekv3",
                "pre <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
                "<｜tool▁sep｜>get_weather\n```json\n{bad}\n```"
                "<｜tool▁call▁end｜><｜tool▁calls▁end｜> post",
            ),
            ParserTextCase(
                "deepseekv31",
                "pre <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
                "<｜tool▁sep｜>{bad}<｜tool▁call▁end｜><｜tool▁calls▁end｜> post",
            ),
            ParserTextCase(
                "deepseekv32",
                'pre <｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
                "{bad}</｜DSML｜invoke></｜DSML｜function_calls> post",
            ),
            ParserTextCase(
                "deepseekv4",
                'pre <｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
                "{bad}</｜DSML｜invoke></｜DSML｜tool_calls> post",
            ),
            ParserTextCase(
                "kimi_k2",
                "pre <|tool_calls_section_begin|><|tool_call_begin|>"
                "functions.get_weather:0<|tool_call_argument_begin|>{bad}"
                "<|tool_call_end|><|tool_calls_section_end|> post",
            ),
            ParserTextCase("glm", "pre <tool_call>get_weather\n{bad}</tool_call> post"),
            ParserTextCase(
                "hunyuan",
                "pre <tool_calls><tool_call>get_weather<tool_sep>{bad}"
                "</tool_call></tool_calls> post",
            ),
            ParserTextCase(
                "interns1",
                "pre <|action_start|> <|plugin|>{bad}<|action_end|> post",
            ),
            ParserTextCase(
                "lfm2",
                "pre <|tool_call_start|>[get_weather(city=some_var)]"
                "<|tool_call_end|> post",
            ),
            ParserTextCase(
                "mimo",
                "pre <tool_call><function=get_weather>{bad}</tool_call> post",
            ),
            ParserTextCase("gemma4", "pre <|tool_call>not-a-call<tool_call|> post"),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = parser.parse_non_stream(case.text)

                self.assertEqual(calls, [])
                self.assertIn("pre", normal_text)
                self.assertIn("post", normal_text)
                self.assert_event_in(parser, CompatibilityEvent.MALFORMED_JSON_DROPPED)

    def test_streaming_started_malformed_call_closes_with_empty_arguments(self):
        cases = [
            ParserChunksCase(
                "deepseekv3",
                "started-json-codeblock",
                (
                    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
                    "<｜tool▁sep｜>get_weather\n```json\n{broken\n```",
                    "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserChunksCase(
                "deepseekv31",
                "started-json",
                (
                    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
                    "<｜tool▁sep｜>{broken",
                    "}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserChunksCase(
                "deepseekv32",
                "started-dsml",
                (
                    '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
                    "{broken",
                    "}</｜DSML｜invoke></｜DSML｜function_calls>",
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserChunksCase(
                "deepseekv4",
                "started-dsml",
                (
                    '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">' "{broken",
                    "}</｜DSML｜invoke></｜DSML｜tool_calls>",
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserChunksCase(
                "kimi_k2",
                "started-json",
                (
                    "<|tool_calls_section_begin|><|tool_call_begin|>"
                    "functions.get_weather:0<|tool_call_argument_begin|>{broken",
                    "}<|tool_call_end|><|tool_calls_section_end|>",
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserChunksCase(
                "gigachat3",
                "started-json",
                (
                    'function call<|role_sep|>\n{"name":"get_weather",',
                    '"arguments":{"city":}}</s>',
                ),
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name, case_name=case.case_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = _stream_chunks(parser, case.chunks)

                self.assertEqual(normal_text, "")
                self.assertEqual(self.call_names(calls), ["get_weather"])
                self.assertEqual(self.streamed_parameters_by_tool(calls), {0: {}})
                self.assert_events_equal(parser, [case.expected_event])

    def test_streaming_drop_retries_same_chunk_tail(self):
        cases = [
            ParserNamedTextCase(
                "deepseekv3",
                "unknown-tool-before-valid",
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
                "<｜tool▁sep｜>bad\n```json\n{}\n```<｜tool▁call▁end｜>"
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
                '```json\n{"city":"Paris"}\n```<｜tool▁call▁end｜>'
                "<｜tool▁calls▁end｜>",
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
            ParserNamedTextCase(
                "deepseekv31",
                "unknown-tool-before-valid",
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>bad"
                "<｜tool▁sep｜>{}<｜tool▁call▁end｜>"
                "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
                '{"city":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
            ParserNamedTextCase(
                "kimi_k2",
                "bad-id-before-valid",
                "<|tool_calls_section_begin|><|tool_call_begin|>badid"
                "<|tool_call_argument_begin|>{}<|tool_call_end|>"
                "<|tool_call_begin|>functions.get_weather:0"
                '<|tool_call_argument_begin|>{"city":"Paris"}'
                "<|tool_call_end|><|tool_calls_section_end|>",
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
            ),
            ParserNamedTextCase(
                "mistral",
                "unknown-tool-before-valid",
                '[TOOL_CALLS]bad[ARGS]{}[TOOL_CALLS]get_weather[ARGS]{"city":"Paris"}',
                CompatibilityEvent.UNKNOWN_TOOL_DROPPED,
            ),
        ]

        for case in cases:
            with self.subTest(parser_name=case.parser_name, case_name=case.case_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                normal_text, calls = parser.parse_stream_chunk(case.text)

                self.assertEqual(normal_text, "")
                self.assertEqual(self.call_names(calls), ["get_weather"])
                self.assert_event_in(parser, case.expected_event)

    def _streaming_malformed_complete_json_cases(self):
        return [
            ParserTextCase(
                "deepseekv3",
                "<｜tool▁calls▁begin｜>"
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
                "```json\n{broken}\n```"
                "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            ),
            ParserTextCase(
                "deepseekv31",
                "<｜tool▁calls▁begin｜>"
                "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{broken}"
                "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            ),
            ParserTextCase(
                "deepseekv32",
                '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
                "{broken}</｜DSML｜invoke></｜DSML｜function_calls>",
            ),
            ParserTextCase(
                "deepseekv4",
                '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
                "{broken}</｜DSML｜invoke></｜DSML｜tool_calls>",
            ),
            ParserTextCase(
                "kimi_k2",
                "<|tool_calls_section_begin|>"
                "<|tool_call_begin|>functions.get_weather:0"
                "<|tool_call_argument_begin|>{broken}<|tool_call_end|>"
                "<|tool_calls_section_end|>",
            ),
            ParserTextCase(
                "gigachat3",
                'function call<|role_sep|>\n{"name":"get_weather",'
                '"arguments": {broken}}</s>',
            ),
        ]

    def test_streaming_complete_malformed_json_args_recorded(self):
        for case in self._streaming_malformed_complete_json_cases():
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS,
                    case.parser_name,
                    enable_compatibility_mode=True,
                )

                normal_text, calls = parser.parse_stream_chunk(case.text)

                self.assertEqual(normal_text, "")
                self.assertEqual(calls, [])
                self.assert_events_equal(
                    parser, [CompatibilityEvent.MALFORMED_JSON_DROPPED]
                )

    def test_streaming_complete_malformed_json_args_strict_fails_open(self):
        for case in self._streaming_malformed_complete_json_cases():
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS,
                    case.parser_name,
                    enable_compatibility_mode=False,
                )

                normal_text, calls = parser.parse_stream_chunk(case.text)

                self.assertEqual(normal_text, case.text)
                self.assertEqual(calls, [])
                self.assert_fail_open_with_detail(
                    parser, CompatibilityEvent.MALFORMED_JSON_DROPPED
                )

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
        self.assert_events_equal(parser, [CompatibilityEvent.FAIL_OPEN])


class TestParamTypeConversion(CompatibilityTestCase):
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


class TestCompatibilityContextOption(CompatibilityTestCase):
    """The serving-level switch: compatibility mode heals; default strict fails open."""

    def test_compatibility_mode_heals_default_strict_fails_open(self):
        text = TestCompatibilityEvents.DUPLICATE_TAG_TEXT

        compatible = FunctionCallParser(
            TOOLS, "minimax-m3", enable_compatibility_mode=True
        )
        _, calls = compatible.parse_non_stream(text)
        self.assert_single_call(calls, arguments={"location": {"city": ["a", "b"]}})
        # The healing is on the audit trail.
        self.assert_events_equal(compatible, [CompatibilityEvent.DUPLICATE_TAG_AS_LIST])

        strict = FunctionCallParser(TOOLS, "minimax-m3")
        normal_text, calls = strict.parse_non_stream(text)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)
        self.assert_event_in(strict, CompatibilityEvent.FAIL_OPEN)


DETECTOR_COMPATIBILITY_CASES = (
    ParserTextCase("apertus2509", '<|tools_prefix|>[{"get_weather":{"city":"Paris"}}'),
    ParserTextCase(
        "cohere_command4",
        '<|START_ACTION|>[{"tool_name":"get_weather",' '"parameters":{"city":"Paris"}}',
    ),
    ParserTextCase(
        "deepseekv3",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
        "<｜tool▁sep｜>get_weather\n```json\n{broken\n```"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ),
    ParserTextCase(
        "deepseekv31",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
        "<｜tool▁sep｜>{broken<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ),
    ParserTextCase(
        "deepseekv32",
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
        "{broken</｜DSML｜invoke></｜DSML｜function_calls>",
    ),
    ParserTextCase(
        "deepseekv4",
        '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
        "{broken</｜DSML｜invoke></｜DSML｜tool_calls>",
    ),
    ParserTextCase("glm", "<tool_call>get_weather\n<city>Paris"),
    ParserTextCase("glm45", "<tool_call>get_weather\n<city>Paris"),
    ParserTextCase("glm47", "<tool_call>get_weather<city>Paris"),
    ParserTextCase(
        "gpt-oss",
        "<|start|>assistant<|channel|>commentary to=functions.get_weather"
        "<|constrain|>json<|message|>{broken<|call|>",
    ),
    ParserTextCase(
        "kimi_k2",
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
        "<|tool_call_argument_begin|>{broken<|tool_call_end|>"
        "<|tool_calls_section_end|>",
    ),
    ParserTextCase(
        "lfm2",
        "<|tool_call_start|>[get_weather(city=some_var)]<|tool_call_end|>",
    ),
    ParserTextCase(
        "llama3",
        '<|python_tag|>[{"name":"get_weather",'
        '"parameters":{"city":"Paris"}}, garbage]',
    ),
    ParserTextCase("mimo", "<tool_call><function=get_weather>{broken}</tool_call>"),
    ParserTextCase(
        "minicpm5",
        '<function name="get_weather"><param name="city">Paris</param>',
    ),
    ParserTextCase(
        "mistral",
        '[TOOL_CALLS] [{"name":"get_weather",' '"arguments":{"city":"Paris"}}',
    ),
    ParserTextCase(
        "poolside_v1",
        "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Paris",
    ),
    ParserTextCase("pythonic", "[get_weather(city=some_var)]"),
    ParserTextCase(
        "qwen", '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}'
    ),
    ParserTextCase(
        "qwen25", '<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}'
    ),
    ParserTextCase(
        "qwen3_coder", "<tool_call>\n<function=get_weather>\n<parameter=city>Paris"
    ),
    ParserTextCase(
        "step3",
        "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
        '<steptml:invoke name="get_weather">'
        '<steptml:parameter name="city">Paris',
    ),
    ParserTextCase(
        "step3p5", "<tool_call>\n<function=get_weather>\n<parameter=city>Paris"
    ),
    ParserTextCase(
        "minimax-m2",
        '<minimax:tool_call><invoke name="get_weather">' '<parameter name="city">Paris',
    ),
    ParserTextCase(
        "minimax-m3",
        f'{NS}<tool_call>\n{NS}<invoke name="get_weather">{NS}<city>Paris',
    ),
    ParserTextCase(
        "trinity",
        '<think>x</think><tool_call>\n{"name":"get_weather",'
        '"arguments":{"city":"Paris"}}',
    ),
    ParserTextCase(
        "interns1",
        '<|action_start|> <|plugin|>{"name":"get_weather",'
        '"parameters":{"city":"Paris"}}',
    ),
    ParserTextCase(
        "hermes", '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}'
    ),
    ParserTextCase(
        "hunyuan", "<tool_calls><tool_call><function=get_weather><parameter=city>Paris"
    ),
    ParserTextCase(
        "gigachat3",
        'function call<|role_sep|>\n{"name":"get_weather",' '"arguments":[]}</s>',
    ),
    ParserTextCase("gemma4", '<|tool_call>call:get_weather{"city":"Paris"<tool_call|>'),
)

_EXPECT_TRUNCATED = (CompatibilityEvent.TRUNCATED_CALL_DROPPED,)
_EXPECT_MALFORMED = (CompatibilityEvent.MALFORMED_JSON_DROPPED,)

DETECTOR_COMPATIBILITY_EXPECTED_EVENTS = {
    "apertus2509": _EXPECT_TRUNCATED,
    "cohere_command4": _EXPECT_TRUNCATED,
    "deepseekv3": _EXPECT_MALFORMED,
    "deepseekv31": _EXPECT_MALFORMED,
    "deepseekv32": _EXPECT_MALFORMED,
    "deepseekv4": _EXPECT_MALFORMED,
    "gemma4": _EXPECT_MALFORMED,
    "gigachat3": _EXPECT_MALFORMED,
    "glm": _EXPECT_TRUNCATED,
    "glm45": _EXPECT_TRUNCATED,
    "glm47": _EXPECT_TRUNCATED,
    "gpt-oss": _EXPECT_MALFORMED,
    "hermes": _EXPECT_TRUNCATED,
    "hunyuan": _EXPECT_TRUNCATED,
    "interns1": _EXPECT_TRUNCATED,
    "kimi_k2": _EXPECT_MALFORMED,
    "lfm2": _EXPECT_MALFORMED,
    "llama3": (
        CompatibilityEvent.MALFORMED_JSON_DROPPED,
        CompatibilityEvent.MALFORMED_JSON_DROPPED,
    ),
    "mimo": _EXPECT_MALFORMED,
    "minicpm5": _EXPECT_TRUNCATED,
    "minimax-m2": _EXPECT_TRUNCATED,
    "minimax-m3": _EXPECT_TRUNCATED,
    "mistral": _EXPECT_TRUNCATED,
    "poolside_v1": _EXPECT_TRUNCATED,
    "pythonic": _EXPECT_MALFORMED,
    "qwen": _EXPECT_TRUNCATED,
    "qwen25": _EXPECT_TRUNCATED,
    "qwen3_coder": _EXPECT_TRUNCATED,
    "step3": _EXPECT_TRUNCATED,
    "step3p5": _EXPECT_TRUNCATED,
    "trinity": _EXPECT_TRUNCATED,
}


UNCONVERTIBLE_VALUE_CASES = (
    ParserTextCase(
        "glm",
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
        "<arg_key>days</arg_key>\n<arg_value>soon</arg_value>\n"
        "</tool_call>",
    ),
    ParserTextCase(
        "glm47",
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>soon</arg_value>"
        "</tool_call>",
    ),
    ParserTextCase(
        "poolside_v1",
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>soon</arg_value>"
        "</tool_call>",
    ),
    ParserTextCase(
        "mimo",
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter>"
        "<parameter=days>soon</parameter>"
        "</function></tool_call>",
    ),
    ParserTextCase(
        "minicpm5",
        '<function name="get_weather">'
        '<param name="city">Paris</param>'
        '<param name="days">soon</param>'
        "</function>",
    ),
    ParserTextCase(
        "hunyuan",
        "<tool_calls><tool_call>get_weather<tool_sep>"
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>soon</arg_value>"
        "</tool_call></tool_calls>",
    ),
)


class TestEveryRegisteredDetectorCompatibility(CompatibilityTestCase):
    """Every registered detector reports local compatibility fallback events."""

    def test_compatibility_mode_records_detector_local_fallbacks(self):
        for case in DETECTOR_COMPATIBILITY_CASES:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                parser.parse_non_stream(case.text)

                self.assert_events_equal(
                    parser, DETECTOR_COMPATIBILITY_EXPECTED_EVENTS[case.parser_name]
                )

    def test_strict_mode_fails_open_for_detector_local_fallbacks(self):
        for case in DETECTOR_COMPATIBILITY_CASES:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=False
                )

                normal_text, calls = parser.parse_non_stream(case.text)

                self.assertEqual(calls, [])
                self.assertEqual(normal_text, case.text)
                self.assert_fail_open_with_detail(
                    parser,
                    DETECTOR_COMPATIBILITY_EXPECTED_EVENTS[case.parser_name][0],
                )

    def test_unconvertible_value_kept_raw_used_by_local_converters(self):
        for case in UNCONVERTIBLE_VALUE_CASES:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=True
                )

                parser.parse_non_stream(case.text)

                self.assert_event_in(
                    parser, CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW
                )

    def test_unconvertible_value_strict_fails_open(self):
        for case in UNCONVERTIBLE_VALUE_CASES:
            with self.subTest(parser_name=case.parser_name):
                parser = FunctionCallParser(
                    TOOLS, case.parser_name, enable_compatibility_mode=False
                )

                normal_text, calls = parser.parse_non_stream(case.text)

                self.assertEqual(calls, [])
                self.assertEqual(normal_text, case.text)
                self.assert_fail_open_with_detail(
                    parser, CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW
                )

    def test_mimo_invalid_boolean_keeps_raw_and_records(self):
        text = (
            "<tool_call><function=set_enabled>"
            "<parameter=enabled>maybe</parameter>"
            "</function></tool_call>"
        )
        parser = FunctionCallParser(
            BOOLEAN_TOOLS, "mimo", enable_compatibility_mode=True
        )

        _, calls = parser.parse_non_stream(text)

        self.assert_single_call(calls, arguments={"enabled": "maybe"})
        self.assert_events_equal(
            parser, [CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW]
        )

    def test_mimo_invalid_boolean_strict_fails_open(self):
        text = (
            "<tool_call><function=set_enabled>"
            "<parameter=enabled>maybe</parameter>"
            "</function></tool_call>"
        )
        parser = FunctionCallParser(
            BOOLEAN_TOOLS, "mimo", enable_compatibility_mode=False
        )

        normal_text, calls = parser.parse_non_stream(text)

        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)
        self.assert_fail_open_with_detail(
            parser, CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW
        )


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


class TestRemainingCompatibilityEvents(CompatibilityTestCase):
    """One test per compatibility event not covered elsewhere, plus the
    strict-mode fail-open behavior at the boundary for each."""

    def test_dropped_invoke_tail(self):
        text = (
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">GARBAGE WITH NO TAG{NS}</invoke>\n'
            f"{NS}</tool_call>"
        )
        parser = _make_parser()
        normal_text, calls = parser.parse_non_stream(text)
        self.assert_single_call(calls, name="get_weather", arguments={})
        self.assert_event_in(parser, CompatibilityEvent.DROPPED_INVOKE_TAIL)

        strict = _make_parser(
            FunctionCallParser(TOOLS, "minimax-m3").detector,
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
        self.assert_event_in(parser, CompatibilityEvent.MISMATCHED_CLOSING_TAG)

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
        self.assert_event_in(parser, CompatibilityEvent.UNCLOSED_TAGS_AT_END)

    def test_non_item_array_child(self):
        text = (
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="set_flags">'
            f"{NS}<flags>{NS}<item>a{NS}</item>{NS}<element>b{NS}</element>{NS}</flags>"
            f"{NS}</invoke>\n{NS}</tool_call>"
        )
        parser = FunctionCallParser(
            ARRAY_TOOLS, "minimax-m3", enable_compatibility_mode=True
        )
        _, calls = parser.parse_non_stream(text)
        self.assert_single_call(calls, arguments={"flags": ["a", "b"]})
        self.assert_event_in(parser, CompatibilityEvent.NON_ITEM_ARRAY_CHILD)

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
        self.assert_event_in(parser, CompatibilityEvent.STRUCTURE_OVERRODE_SCHEMA)

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
        context = CompatibilityContext(strict=True)
        value = data_type.convert("5", compatibility=context)
        self.assertEqual(value, 5)
        self.assert_events_equal(context, [CompatibilityEvent.INVALID_SCHEMA_IGNORED])

    def test_truncated_call_dropped_identical_in_both_modes(self):

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
        self.assert_event_in(compatible, CompatibilityEvent.TRUNCATED_CALL_DROPPED)

    def test_skipped_non_function_entry(self):

        text = (
            "<｜tool_calls_begin｜>\n"
            "<｜tool_call_begin｜>thought<｜tool_sep｜>hmm<｜tool_call_end｜>\n"
            "<｜tool_call_begin｜>function<｜tool_sep｜>"
            '<steptml:invoke name="get_weather">\n'
            '<steptml:parameter name="city">Paris</steptml:parameter>\n'
            "</steptml:invoke><｜tool_call_end｜>\n"
            "<｜tool_calls_end｜>"
        )
        parser = FunctionCallParser(TOOLS, "step3", enable_compatibility_mode=True)
        _, calls = parser.parse_non_stream(text)
        self.assertEqual(self.call_names(calls), ["get_weather"])
        self.assert_event_in(parser, CompatibilityEvent.SKIPPED_NON_FUNCTION_ENTRY)

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
        self.assert_single_call(calls, arguments={"city": "Paris", "days": 3})
        self.assert_event_in(parser, CompatibilityEvent.MISSING_CLOSE_TAG)


class TestComplexSchemaCompatibility(CompatibilityTestCase):
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
            "minimax-m3",
            enable_compatibility_mode=enable_compatibility_mode,
        )

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
        self.assertEqual(
            args["travelers"][0]["preferences"]["tags"], ["museum", "ramen"]
        )
        self.assertEqual(args["travelers"][1]["age"], "old")
        self.assertEqual(args["itinerary"][1]["day"], 2)

        events = self.events_of(parser)
        self.assertIn(CompatibilityEvent.NON_ITEM_ARRAY_CHILD, events)
        self.assertIn(CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW, events)

    def test_complex_schema_strict_parser_fails_open(self):
        parser = self._parser(enable_compatibility_mode=False)
        normal_text, calls = parser.parse_non_stream(self.COMPLEX_TEXT)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, self.COMPLEX_TEXT)
        self.assert_fail_open_with_detail(
            parser, CompatibilityEvent.UNCONVERTIBLE_VALUE_KEPT_RAW
        )


class TestTagStreamingGates(CompatibilityTestCase):
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
            self.assertEqual(self.call_names(calls), ["get_weather"], chunk_size)
            self.assert_streamed_arguments(calls, {"city": "Paris"})
            # Indices stay dense: the grammar never assigns an ordinal to the
            # dropped call in the first place.
            self.assertEqual({c.tool_index for c in calls}, {0}, chunk_size)
            self.assert_event_in(parser, CompatibilityEvent.UNKNOWN_TOOL_DROPPED)
            # Consistent with the non-streaming path.
            nonstream = FunctionCallParser(
                TOOLS, "minimax-m2", enable_compatibility_mode=True
            )
            _, ns_calls = nonstream.parse_non_stream(text)
            self.assertEqual(self.call_names(ns_calls), ["get_weather"])

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
            self.assert_fail_open_with_detail(
                parser, CompatibilityEvent.UNKNOWN_TOOL_DROPPED
            )


if __name__ == "__main__":
    unittest.main()
