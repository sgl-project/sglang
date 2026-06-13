"""Deterministic fuzz-style tests for streaming tool-call parsers."""

import json
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.minicpm5_detector import MiniCPM5Detector
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.minimax_m3_nom import MinimaxM3NomDetector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "base-a-test-cpu")


DetectorFactory = Callable[[], BaseFormatDetector]


@dataclass(frozen=True)
class StreamingFuzzCase:
    name: str
    detector_factory: DetectorFactory
    tools: List[Tool]
    text: str
    markers: List[str]
    forbidden_normal_text_markers: Optional[List[str]] = None
    check_normal_text: bool = True
    exact_normal_text: bool = False
    #: True when the text follows the wire format exactly: parsing it must
    #: apply zero tolerances, so strict mode must behave identically to
    #: compatibility mode (see assert_clean_input_strict_invariant).
    clean: bool = True


@dataclass(frozen=True)
class NoCrashFuzzCase:
    name: str
    detector_factory: DetectorFactory
    tools: List[Tool]
    text: str
    markers: List[str]


@dataclass
class CollectedCall:
    name: str = ""
    parameters: str = ""
    first_name_pos: Optional[int] = None
    first_param_pos: Optional[int] = None


class JsonTagDetector(BaseFormatDetector):
    """Small test detector that exercises BaseFormatDetector streaming logic."""

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_separator = ", "

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = text[:idx]
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        calls = []
        for body in re.findall(pattern, text, re.DOTALL):
            calls.extend(self.parse_base_json(json.loads(body.strip()), tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'{self.bot_token}{{"name":"{name}", "arguments":',
            end=f"}}{self.eot_token}",
            trigger=self.bot_token,
        )


def _tool(name: str, properties: dict, required: Optional[List[str]] = None) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} test tool",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        ),
    )


def make_common_tools() -> List[Tool]:
    return [
        _tool(
            "get_weather",
            {
                "city": {"type": "string"},
                "unit": {"type": "string"},
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "flags": {"type": "array", "items": {"type": "string"}},
                "options": {
                    "type": "object",
                    "properties": {"detailed": {"type": "boolean"}},
                },
                "note": {"type": "string"},
            },
            ["city"],
        ),
        _tool(
            "search",
            {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "filters": {
                    "type": "object",
                    "properties": {"fresh": {"type": "boolean"}},
                },
            },
            ["query"],
        ),
        _tool("get_current_date", {}, []),
        _tool(
            "sum_values",
            {"nums": {"type": "array", "items": {"type": "integer"}}},
            ["nums"],
        ),
        _tool(
            "plan_trip",
            {
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
            ["destination", "travelers"],
        ),
    ]


def collect_streamed_tool_calls(calls: Iterable[ToolCallItem]) -> List[CollectedCall]:
    collected: dict[int, CollectedCall] = {}
    for pos, call in enumerate(calls):
        assert call.tool_index >= 0
        current = collected.setdefault(call.tool_index, CollectedCall())
        if call.name:
            current.name += call.name
            if current.first_name_pos is None:
                current.first_name_pos = pos
        if call.parameters:
            current.parameters += call.parameters
            if current.first_param_pos is None:
                current.first_param_pos = pos
    return [collected[idx] for idx in sorted(collected)]


def chunk_strategies(text: str, markers: List[str]) -> List[tuple[str, List[str]]]:
    strategies: List[tuple[str, List[str]]] = [
        ("all", [text]),
        ("char", list(text)),
    ]

    for size in (2, 3, 5, 8, 13):
        strategies.append(
            (f"fixed-{size}", [text[i : i + size] for i in range(0, len(text), size)])
        )

    cut_points = {0, len(text)}
    for marker in markers + ["{", "}", "[", "]", '"', "\\", "</", "<", ">"]:
        if not marker:
            continue
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos == -1:
                break
            for offset in (-1, 0, 1, len(marker) - 1, len(marker), len(marker) + 1):
                cut = pos + offset
                if 0 < cut < len(text):
                    cut_points.add(cut)
            start = pos + 1
    cuts = sorted(cut_points)
    strategies.append(("marker-boundary", [text[a:b] for a, b in zip(cuts, cuts[1:])]))

    for seed in (0, 7, 13):
        rng = random.Random(seed)
        chunks = []
        idx = 0
        while idx < len(text):
            size = rng.randint(1, 17)
            chunks.append(text[idx : idx + size])
            idx += size
        strategies.append((f"seed-{seed}", chunks))

    return strategies


def _load_parameters(parameters: str) -> object:
    return json.loads(parameters or "{}")


def _drain_streaming_detector(
    detector: BaseFormatDetector, tools: List[Tool], max_ticks: int
) -> tuple[str, List[ToolCallItem]]:
    normal_text = ""
    calls: List[ToolCallItem] = []
    for _ in range(max_ticks):
        result = detector.parse_streaming_increment("", tools)
        if not result.normal_text and not result.calls:
            break
        normal_text += result.normal_text
        calls.extend(result.calls)
    return normal_text, calls


def test_base_streaming_holds_back_only_partial_bot_token() -> None:
    detector = JsonTagDetector()
    result = detector.parse_streaming_increment("Hello <tool_", make_common_tools())

    assert result.normal_text == "Hello "
    assert result.calls == []
    assert detector._buffer == "<tool_"


def _forbidden_normal_text_markers(markers: Iterable[str]) -> List[str]:
    return [
        marker
        for marker in markers
        if len(marker) > 1
        and any(token in marker.lower() for token in ("tool", "invoke", "function"))
    ]


def assert_stream_matches_non_stream(case: StreamingFuzzCase) -> None:
    reference = case.detector_factory().detect_and_parse(case.text, case.tools)
    assert reference.calls, f"{case.name}: reference parser found no tool calls"
    reference_names = [call.name for call in reference.calls]
    reference_args = [_load_parameters(call.parameters) for call in reference.calls]

    for strategy_name, chunks in chunk_strategies(case.text, case.markers):
        detector = case.detector_factory()
        all_calls: List[ToolCallItem] = []
        normal_text = ""

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, case.tools)
            normal_text += result.normal_text
            all_calls.extend(result.calls)

        # Some detectors emit the function name and argument JSON on separate
        # parser ticks even when the model output arrived in one chunk.
        drained_text, drained_calls = _drain_streaming_detector(
            detector, case.tools, 2 * len(reference.calls) + 2
        )
        normal_text += drained_text
        all_calls.extend(drained_calls)

        collected = collect_streamed_tool_calls(all_calls)
        assert len(collected) == len(reference.calls), (
            f"{case.name}/{strategy_name}: expected {len(reference.calls)} calls, "
            f"got {len(collected)} from {all_calls}"
        )
        assert [call.name for call in collected] == reference_names
        assert [_load_parameters(call.parameters) for call in collected] == reference_args

        if case.check_normal_text:
            if case.exact_normal_text:
                assert normal_text == reference.normal_text
            else:
                assert normal_text.strip() == reference.normal_text.strip()

        forbidden_markers = (
            case.forbidden_normal_text_markers
            if case.forbidden_normal_text_markers is not None
            else _forbidden_normal_text_markers(case.markers)
        )
        for marker in forbidden_markers:
            assert marker not in normal_text

        for call in collected:
            if call.first_name_pos is not None and call.first_param_pos is not None:
                assert call.first_name_pos <= call.first_param_pos


def assert_clean_input_strict_invariant(case: StreamingFuzzCase) -> None:
    """Clean input must apply zero tolerances, so strict == compatibility mode.

    Without this invariant, accidental healing on valid input goes unnoticed
    in compatibility mode and surfaces only as spurious strict-mode failures.
    """
    detector = case.detector_factory()
    reference = detector.detect_and_parse(case.text, case.tools)
    assert detector.compatibility_records == [], (
        f"{case.name}: clean input applied tolerances (compatibility, non-stream): "
        f"{detector.compatibility_records}"
    )
    reference_names = [call.name for call in reference.calls]
    reference_args = [_load_parameters(call.parameters) for call in reference.calls]

    strict_detector = case.detector_factory()
    strict_detector.compatibility.strict = True
    result = strict_detector.detect_and_parse(case.text, case.tools)
    assert [call.name for call in result.calls] == reference_names
    assert [
        _load_parameters(call.parameters) for call in result.calls
    ] == reference_args
    assert result.normal_text == reference.normal_text

    def parse_stream(detector: BaseFormatDetector) -> tuple[str, List[CollectedCall]]:
        normal_text = ""
        all_calls: List[ToolCallItem] = []
        for i in range(0, len(case.text), 7):
            result = detector.parse_streaming_increment(
                case.text[i : i + 7], case.tools
            )
            normal_text += result.normal_text
            all_calls.extend(result.calls)
        drained_text, drained_calls = _drain_streaming_detector(
            detector, case.tools, 2 * len(reference.calls) + 2
        )
        normal_text += drained_text
        all_calls.extend(drained_calls)
        return normal_text, collect_streamed_tool_calls(all_calls)

    compatibility_stream_detector = case.detector_factory()
    compatibility_normal_text, compatibility_collected = parse_stream(
        compatibility_stream_detector
    )
    assert [call.name for call in compatibility_collected] == reference_names
    assert [
        _load_parameters(call.parameters) for call in compatibility_collected
    ] == reference_args
    assert compatibility_stream_detector.compatibility_records == [], (
        f"{case.name}: clean input applied tolerances (compatibility, streaming): "
        f"{compatibility_stream_detector.compatibility_records}"
    )

    strict_detector = case.detector_factory()
    strict_detector.compatibility.strict = True
    strict_normal_text, collected = parse_stream(strict_detector)
    assert [call.name for call in collected] == reference_names
    assert [_load_parameters(call.parameters) for call in collected] == reference_args
    assert strict_normal_text == compatibility_normal_text
    assert strict_detector.compatibility_records == [], (
        f"{case.name}: clean input applied tolerances (strict, streaming): "
        f"{strict_detector.compatibility_records}"
    )


def mutate_text(text: str, markers: List[str]) -> List[str]:
    """Seeded, deterministic mutants: truncations at marker boundaries,
    deleted / duplicated markers, garbage injections, random truncations."""
    rng = random.Random(42)
    mutants: List[str] = []
    for marker in dict.fromkeys(m for m in markers if m):
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos == -1:
                break
            mutants.append(text[: pos + max(1, len(marker) // 2)])
            mutants.append(text[:pos] + text[pos + len(marker) :])
            mutants.append(text[:pos] + marker + text[pos:])
            start = pos + 1
    for _ in range(4):
        mutants.append(text[: rng.randrange(1, len(text))])
    for _ in range(4):
        at = rng.randrange(0, len(text))
        mutants.append(text[:at] + " GARBAGE\U0001f92a " + text[at:])
    return list(dict.fromkeys(mutants))


def assert_mutants_never_crash(
    case: StreamingFuzzCase, enable_compatibility_mode: bool
) -> None:
    """Mutated wire text must never escape the FunctionCallParser boundary,
    whether compatibility mode is enabled or not, streaming or not — and the
    streaming latch must keep passing text through after a failure."""

    def make_parser() -> FunctionCallParser:
        return FunctionCallParser.with_detector(
            case.detector_factory(),
            case.tools,
            enable_compatibility_mode=enable_compatibility_mode,
        )

    for mutant in mutate_text(case.text, case.markers):
        parser = make_parser()
        normal_text, calls = parser.parse_non_stream(mutant)
        for call in calls:
            # Non-streaming calls carry complete argument JSON by contract.
            _load_parameters(call.parameters)

        parser = make_parser()
        all_calls: List[ToolCallItem] = []
        for i in range(0, len(mutant), 7):
            _, calls = parser.parse_stream_chunk(mutant[i : i + 7])
            all_calls.extend(calls)
        for _ in range(4):
            text, calls = parser.parse_stream_chunk("")
            if not text and not calls:
                break
            all_calls.extend(calls)
        for call in all_calls:
            assert call.tool_index >= 0
        # The latch: after any failure, later chunks flow through as text.
        detector_failed = parser._detector_failed
        text_after, calls_after = parser.parse_stream_chunk(" tail-after-mutant")
        if detector_failed:
            assert text_after == " tail-after-mutant"
            assert calls_after == []
        else:
            assert isinstance(text_after, str)
            assert isinstance(calls_after, list)


def assert_stream_no_crash(case: NoCrashFuzzCase) -> None:
    # The never-crash property belongs to the FunctionCallParser boundary:
    # detectors are pure may-raise parsers (see the compatibility package scope 4), so the
    # adversarial sweep drives the boundary, not the detector directly.
    for _, chunks in chunk_strategies(case.text, case.markers):
        parser = FunctionCallParser.with_detector(case.detector_factory(), case.tools)
        all_calls: List[ToolCallItem] = []
        for chunk in chunks:
            _, calls = parser.parse_stream_chunk(chunk)
            all_calls.extend(calls)

        for call in all_calls:
            assert call.tool_index >= 0


def _json_tag_text(detector: BaseFormatDetector, name: str, args: dict) -> str:
    info = detector.structure_info()(name)
    return info.begin + json.dumps(args, ensure_ascii=False) + info.end


def _hunyuan_text(*tool_bodies: str, leading: str = "") -> str:
    return f"{leading}<tool_calls>{''.join(tool_bodies)}</tool_calls>"


def _hunyuan_call(name: str, args: dict) -> str:
    parts = [f"<tool_call>{name}<tool_sep>"]
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value = "true" if value else "false"
        elif value is None:
            value = "null"
        parts.append(f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>")
    parts.append("</tool_call>")
    return "".join(parts)


def _minicpm_call(name: str, args: dict) -> str:
    params = []
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        params.append(f'<param name="{key}">{value}</param>')
    return f'<function name="{name}">{"".join(params)}</function>'


def _m2_call(name: str, args: dict) -> str:
    params = []
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value = "true" if value else "false"
        elif value is None:
            value = "null"
        params.append(f'<parameter name="{key}">{value}</parameter>')
    return f'<invoke name="{name}">\n' + "\n".join(params) + "\n</invoke>"


def _m2_text(*calls: str, leading: str = "", trailing: str = "") -> str:
    inner = "\n".join(calls)
    return f"{leading}<minimax:tool_call>\n{inner}\n</minimax:tool_call>{trailing}"


def _m3_call(name: str, args: dict) -> str:
    ns = "]<]minimax[>["
    parts = [f'{ns}<invoke name="{name}">']
    for key, value in args.items():
        parts.append(_m3_param(key, value))
    parts.append(f"{ns}</invoke>")
    return "".join(parts)


def _m3_text(*calls: str, leading: str = "") -> str:
    ns = "]<]minimax[>["
    return f"{leading}{ns}<tool_call>\n" + "\n".join(calls) + f"\n{ns}</tool_call>"


def _m3_param(key: str, value) -> str:
    ns = "]<]minimax[>["
    if isinstance(value, list):
        return (
            f"{ns}<{key}>"
            + "".join(_m3_param("item", item) for item in value)
            + f"{ns}</{key}>"
        )
    if isinstance(value, dict):
        return (
            f"{ns}<{key}>"
            + "".join(
                _m3_param(child_key, child_value)
                for child_key, child_value in value.items()
            )
            + f"{ns}</{key}>"
        )
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif value is None:
        value = "null"
    return f"{ns}<{key}>{value}{ns}</{key}>"


def _qwen3_call(name: str, args: dict) -> str:
    params = []
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value = "true" if value else "false"
        elif value is None:
            value = "null"
        params.append(f"<parameter={key}>\n{value}\n</parameter>")
    return (
        f"<tool_call>\n<function={name}>\n"
        + "\n".join(params)
        + "\n</function>\n</tool_call>"
    )


def _step3_entry(name: str, args: dict) -> str:
    params = []
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value = "true" if value else "false"
        elif value is None:
            value = "null"
        params.append(f'<steptml:parameter name="{key}">{value}</steptml:parameter>')
    return (
        f'<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="{name}">\n'
        + "\n".join(params)
        + "\n</steptml:invoke><｜tool_call_end｜>"
    )


def _step3_text(*entries: str, leading: str = "", trailing: str = "") -> str:
    return (
        f"{leading}<｜tool_calls_begin｜>\n"
        + "\n".join(entries)
        + f"\n<｜tool_calls_end｜>{trailing}"
    )


COMMON_TOOLS = make_common_tools()

PLAN_TRIP_ARGS = {
    "destination": "Tokyo",
    "days": 5,
    "budget": 1250.5,
    "travelers": [
        {
            "name": "Ada",
            "age": 34,
            "preferences": {"vegetarian": True, "tags": ["museum", "ramen"]},
        },
        {
            "name": "Lin",
            "age": 8,
            "preferences": {"vegetarian": False, "tags": ["park"]},
        },
    ],
    "itinerary": [
        {"day": 1, "city": "Tokyo", "activities": ["arrival", "ramen"]},
        {"day": 2, "city": "Tokyo", "activities": ["museum"]},
    ],
    "contact": None,
    "metadata": {"season": "spring", "pace": "moderate"},
    "transport": {"mode": "train", "priority": 2},
}


def _extend_marker_candidates(markers: List[str], value) -> None:
    if isinstance(value, str):
        if value:
            markers.append(value)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _extend_marker_candidates(markers, item)


def _marker_candidates_for_detector(detector: BaseFormatDetector) -> List[str]:
    markers: List[str] = []

    try:
        info = detector.structure_info()("get_weather")
        _extend_marker_candidates(markers, (info.trigger, info.begin, info.end))
    except Exception:
        pass

    for cls in reversed(type(detector).mro()):
        for attr_name, value in vars(cls).items():
            if attr_name.startswith("__"):
                continue
            _extend_marker_candidates(markers, value)

    for value in vars(detector).values():
        _extend_marker_candidates(markers, value)

    if isinstance(detector, PythonicDetector):
        for tool in COMMON_TOOLS:
            if tool.function.name:
                markers.append(f"[{tool.function.name}(")
                markers.append(f"{tool.function.name}(")

    return list(dict.fromkeys(markers))


def _is_primary_tool_marker(marker: str) -> bool:
    stripped = marker.strip()
    if stripped in {"", ",", ";", "[", "]", "{", "}"}:
        return False
    if re.match(r"^\[?[A-Za-z_]\w*\(", stripped):
        return True
    lower = stripped.lower()
    return any(
        token in lower
        for token in ("tool", "function", "invoke", "action", "call", "dsml")
    )


def _preferred_partial_marker(markers: List[str]) -> str:
    for marker in markers:
        if _is_primary_tool_marker(marker):
            return marker
    return markers[0] if markers else "<tool_call>"


VALID_CASES: List[StreamingFuzzCase] = [
    StreamingFuzzCase(
        name="base-json-hard-markers-in-string",
        detector_factory=JsonTagDetector,
        tools=COMMON_TOOLS,
        text=(
            'Lead. <tool_call>{"name": "get_weather", "arguments": '
            '{"city": "Paris", "note": "literal <tool_call> marker"}}'
            "</tool_call>"
        ),
        markers=["<tool_call>", "</tool_call>", '"arguments"'],
        forbidden_normal_text_markers=[],
        check_normal_text=False,
    ),
    StreamingFuzzCase(
        name="hermes-json-unicode-nested",
        detector_factory=HermesDetector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. "
            + _json_tag_text(
                HermesDetector(),
                "get_weather",
                {
                    "city": "杭州",
                    "unit": "celsius",
                    "options": {"detailed": True},
                    "note": 'quote " slash \\ newline\n<tool_call>',
                },
            )
        ),
        markers=["<tool_call>", "</tool_call>", '"arguments"'],
    ),
    StreamingFuzzCase(
        name="qwen-two-calls",
        detector_factory=Qwen25Detector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. "
            + _json_tag_text(Qwen25Detector(), "get_weather", {"city": "北京"})
            + "\n"
            + _json_tag_text(
                Qwen25Detector(), "search", {"query": "restaurants", "limit": 3}
            )
        ),
        markers=["<tool_call>\n", "\n</tool_call>", '"arguments"'],
        check_normal_text=False,
    ),
    StreamingFuzzCase(
        name="qwen-complex-schema-plan-trip",
        detector_factory=Qwen25Detector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. "
            + _json_tag_text(Qwen25Detector(), "plan_trip", PLAN_TRIP_ARGS)
            + " After."
        ),
        markers=["<tool_call>\n", "\n</tool_call>", '"arguments"'],
        check_normal_text=False,
    ),
    StreamingFuzzCase(
        name="llama-python-tag-array-object",
        detector_factory=Llama32Detector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. "
            + _json_tag_text(
                Llama32Detector(),
                "get_weather",
                {
                    "city": "Tokyo",
                    "flags": ["rain", "wind"],
                    "options": {"detailed": False},
                },
            )
        ),
        markers=["<|python_tag|>", '"arguments"', "{", "}"],
        check_normal_text=False,
    ),
    StreamingFuzzCase(
        name="pythonic-multiple-calls",
        detector_factory=PythonicDetector,
        tools=COMMON_TOOLS,
        text=(
            'Lead. [get_weather(city="Paris", count=3, enabled=True), '
            'search(query="hotels", limit=2)] After.'
        ),
        markers=["[", "]", "get_weather(", "search("],
    ),
    StreamingFuzzCase(
        name="mistral-compact-hard-json",
        detector_factory=MistralDetector,
        tools=COMMON_TOOLS,
        text=(
            'Lead. [TOOL_CALLS]get_weather[ARGS]{"city":"San Francisco",'
            '"flags":["fog","wind"],"note":"[TOOL_CALLS] literal"}'
        ),
        markers=["[TOOL_CALLS]", "[ARGS]", "{", "}"],
    ),
    StreamingFuzzCase(
        name="hunyuan-multiple-typed",
        detector_factory=HunyuanDetector,
        tools=COMMON_TOOLS,
        text=_hunyuan_text(
            _hunyuan_call("get_current_date", {}),
            _hunyuan_call(
                "get_weather",
                {"city": "上海", "count": 7, "enabled": True},
            ),
            leading="Lead. ",
        ),
        markers=[
            "<tool_calls>",
            "</tool_calls>",
            "<tool_call>",
            "</tool_call>",
            "<tool_sep>",
            "<arg_key>",
            "<arg_value>",
        ],
    ),
    StreamingFuzzCase(
        name="minicpm-multiple-complete-blocks",
        detector_factory=MiniCPM5Detector,
        tools=COMMON_TOOLS,
        text=(
            "Lead.\n"
            + _minicpm_call("get_weather", {"city": "北京", "note": "A & B"})
            + _minicpm_call("sum_values", {"nums": [1, 2, 3]})
        ),
        markers=[
            "<function",
            "</function>",
            "<param",
            "</param>",
            'name="get_weather"',
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="deepseekv3-two-fenced-json-calls",
        detector_factory=DeepSeekV3Detector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. <｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Shanghai", "count": 3}\n```<｜tool▁call▁end｜>\n'
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n"
            '```json\n{"query": "noodles"}\n```<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        ),
        markers=[
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁sep｜>",
            "```json",
            "```",
            "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ],
        check_normal_text=False,
    ),
    StreamingFuzzCase(
        name="kimik2-two-calls",
        detector_factory=KimiK2Detector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. <|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>"
            '{"city": "Paris", "enabled": true}<|tool_call_end|>'
            "<|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>"
            '{"query": "ramen", "limit": 5}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        ),
        markers=[
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ],
        check_normal_text=False,
    ),
    StreamingFuzzCase(
        name="minimax-m3-multiple-invokes-nested",
        detector_factory=MinimaxM3NomDetector,
        tools=COMMON_TOOLS,
        text=_m3_text(
            _m3_call(
                "get_weather",
                {
                    "city": "杭州",
                    "count": 5,
                    "enabled": True,
                    "flags": ["rain", "wind"],
                    "options": {"detailed": "true"},
                },
            ),
            _m3_call("search", {"query": "pizza", "limit": 3}),
            leading="Lead. ",
        ),
        markers=[
            "]<]minimax[>[",
            "]<]minimax[>[<tool_call>",
            '<invoke name="',
            "]<]minimax[>[</invoke>",
            "]<]minimax[>[</tool_call>",
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="minimax-m3-complex-schema-plan-trip",
        detector_factory=MinimaxM3NomDetector,
        tools=COMMON_TOOLS,
        text=_m3_text(_m3_call("plan_trip", PLAN_TRIP_ARGS), leading="Lead. "),
        markers=[
            "]<]minimax[>[",
            "]<]minimax[>[<tool_call>",
            '<invoke name="',
            "]<]minimax[>[<travelers>",
            "]<]minimax[>[<item>",
            "]<]minimax[>[</tool_call>",
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="minimax-m2-two-calls-typed",
        detector_factory=MinimaxM2Detector,
        tools=COMMON_TOOLS,
        text=_m2_text(
            _m2_call(
                "get_weather",
                {
                    "city": "東京",
                    "count": 7,
                    "enabled": True,
                    "flags": ["rain", "wind"],
                    "options": {"detailed": True},
                },
            ),
            _m2_call("search", {"query": "ramen", "limit": 5}),
            leading="Let me check. ",
        ),
        markers=[
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            '<invoke name="',
            "</invoke>",
            '<parameter name="',
            "</parameter>",
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="minimax-m2-two-blocks-text-between",
        detector_factory=MinimaxM2Detector,
        tools=COMMON_TOOLS,
        text=(
            _m2_text(_m2_call("get_weather", {"city": "Paris"}), leading="Lead. ")
            + "\nBetween blocks.\n"
            + _m2_text(_m2_call("search", {"query": "pizza", "limit": 3}))
            + " After."
        ),
        markers=[
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            '<invoke name="',
            "</invoke>",
            '<parameter name="',
            "</parameter>",
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="minimax-m2-junk-between-invokes",
        detector_factory=MinimaxM2Detector,
        tools=COMMON_TOOLS,
        text=_m2_text(
            _m2_call("get_weather", {"city": "Paris"}),
            "OOPS stray text",
            _m2_call("search", {"query": "pizza"}),
            leading="Lead. ",
        ),
        markers=[
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            '<invoke name="',
            "</invoke>",
            '<parameter name="',
            "</parameter>",
        ],
        check_normal_text=False,
        clean=False,
    ),
    StreamingFuzzCase(
        name="minimax-m2-null-and-empty-values",
        detector_factory=MinimaxM2Detector,
        tools=COMMON_TOOLS,
        text=_m2_text(
            '<invoke name="get_weather">\n'
            '<parameter name="city">null</parameter>\n'
            '<parameter name="note"></parameter>\n'
            "</invoke>",
            leading="Lead. ",
        ),
        markers=[
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            '<invoke name="',
            "</invoke>",
            '<parameter name="',
            "</parameter>",
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="minimax-m2-complex-schema-plan-trip",
        detector_factory=MinimaxM2Detector,
        tools=COMMON_TOOLS,
        text=_m2_text(_m2_call("plan_trip", PLAN_TRIP_ARGS), leading="Lead. "),
        markers=[
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            '<invoke name="',
            "</invoke>",
            '<parameter name="travelers">',
            "</parameter>",
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="qwen3-coder-two-blocks-typed",
        detector_factory=Qwen3CoderDetector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. "
            + _qwen3_call(
                "get_weather",
                {
                    "city": "東京",
                    "count": 7,
                    "enabled": True,
                    "flags": ["rain", "wind"],
                    "options": {"detailed": True},
                },
            )
            + "\nBetween.\n"
            + _qwen3_call("search", {"query": "ramen", "limit": 5})
            + " After."
        ),
        markers=[
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "</function>",
            "<parameter=",
            "</parameter>",
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="qwen3-coder-missing-parameter-close",
        detector_factory=Qwen3CoderDetector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. <tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n"
            "<parameter=unit>\ncelsius\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        markers=[
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "</function>",
            "<parameter=",
            "</parameter>",
        ],
        clean=False,
    ),
    StreamingFuzzCase(
        name="qwen3-coder-complex-schema-plan-trip",
        detector_factory=Qwen3CoderDetector,
        tools=COMMON_TOOLS,
        text="Lead. " + _qwen3_call("plan_trip", PLAN_TRIP_ARGS) + " After.",
        markers=[
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "</function>",
            "<parameter=travelers>",
            "</parameter>",
        ],
        exact_normal_text=True,
    ),
    StreamingFuzzCase(
        name="step3-two-entries-typed",
        detector_factory=Step3Detector,
        tools=COMMON_TOOLS,
        text=_step3_text(
            _step3_entry(
                "get_weather",
                {"city": "上海", "count": 5, "enabled": True},
            ),
            _step3_entry("search", {"query": "pizza", "limit": 3}),
            leading="Lead. ",
            trailing=" After.",
        ),
        markers=[
            "<｜tool_calls_begin｜>",
            "<｜tool_calls_end｜>",
            "<｜tool_call_begin｜>",
            "<｜tool_call_end｜>",
            "<｜tool_sep｜>",
            '<steptml:invoke name="',
            "</steptml:invoke>",
            '<steptml:parameter name="',
            "</steptml:parameter>",
        ],
    ),
    StreamingFuzzCase(
        name="step3-complex-schema-plan-trip",
        detector_factory=Step3Detector,
        tools=COMMON_TOOLS,
        text=_step3_text(
            _step3_entry("plan_trip", PLAN_TRIP_ARGS),
            leading="Lead. ",
            trailing=" After.",
        ),
        markers=[
            "<｜tool_calls_begin｜>",
            "<｜tool_calls_end｜>",
            "<｜tool_call_begin｜>",
            "<｜tool_call_end｜>",
            "<｜tool_sep｜>",
            '<steptml:invoke name="',
            '<steptml:parameter name="travelers">',
            "</steptml:parameter>",
        ],
    ),
    StreamingFuzzCase(
        name="step3-non-function-entry-skipped",
        detector_factory=Step3Detector,
        tools=COMMON_TOOLS,
        text=_step3_text(
            "<｜tool_call_begin｜>thought<｜tool_sep｜>pondering...<｜tool_call_end｜>",
            _step3_entry("search", {"query": "pizza"}),
            leading="Lead. ",
        ),
        markers=[
            "<｜tool_calls_begin｜>",
            "<｜tool_calls_end｜>",
            "<｜tool_call_begin｜>",
            "<｜tool_call_end｜>",
            "<｜tool_sep｜>",
            '<steptml:invoke name="',
            "</steptml:invoke>",
            '<steptml:parameter name="',
            "</steptml:parameter>",
        ],
        clean=False,
    ),
]


def _registered_no_crash_cases() -> List[NoCrashFuzzCase]:
    cases: List[NoCrashFuzzCase] = []
    seen_classes = set()
    for parser_name, detector_cls in sorted(FunctionCallParser.ToolCallParserEnum.items()):
        if detector_cls in seen_classes:
            continue
        seen_classes.add(detector_cls)
        detector = detector_cls()

        markers = _marker_candidates_for_detector(detector)
        marker = _preferred_partial_marker(markers)
        partial_marker = marker[: max(1, len(marker) - 1)]
        cases.append(
            NoCrashFuzzCase(
                name=f"{parser_name}-partial-marker",
                detector_factory=detector_cls,
                tools=COMMON_TOOLS,
                text=f"normal before {partial_marker}",
                markers=markers or [partial_marker],
            )
        )

    return cases


NO_CRASH_CASES: List[NoCrashFuzzCase] = [
    NoCrashFuzzCase(
        name="minimax-m3-truncated-namespace",
        detector_factory=MinimaxM3NomDetector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. ]<]minimax[>[<tool_call>\n"
            ']<]minimax[>[<invoke name="get_weather">]<]minimax[>[<city>Par'
        ),
        markers=["]<]minimax[>[", "<tool_call>", '<invoke name="', "<city>"],
    ),
    NoCrashFuzzCase(
        name="hunyuan-truncated-arg-value",
        detector_factory=HunyuanDetector,
        tools=COMMON_TOOLS,
        text=(
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>San Fran"
        ),
        markers=["<tool_calls>", "<tool_call>", "<tool_sep>", "<arg_value>"],
    ),
    NoCrashFuzzCase(
        name="mistral-truncated-compact-json",
        detector_factory=MistralDetector,
        tools=COMMON_TOOLS,
        text='[TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "flags": ["a"',
        markers=["[TOOL_CALLS]", "[ARGS]", "{", "["],
    ),
    NoCrashFuzzCase(
        name="base-truncated-json",
        detector_factory=JsonTagDetector,
        tools=COMMON_TOOLS,
        text='<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"',
        markers=["<tool_call>", '"arguments"', "{"],
    ),
    NoCrashFuzzCase(
        name="qwen3-coder-truncated-mid-value",
        detector_factory=Qwen3CoderDetector,
        tools=COMMON_TOOLS,
        text="Check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nBei",
        markers=[
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "</function>",
            "<parameter=",
            "</parameter>",
        ],
    ),
    NoCrashFuzzCase(
        name="step3-truncated-mid-entry",
        detector_factory=Step3Detector,
        tools=COMMON_TOOLS,
        text=(
            "Lead. <｜tool_calls_begin｜>\n<｜tool_call_begin｜>function<｜tool_sep｜>"
            '<steptml:invoke name="get_weather">\n<steptml:parameter name="city">Par'
        ),
        markers=[
            "<｜tool_calls_begin｜>",
            "<｜tool_calls_end｜>",
            "<｜tool_call_begin｜>",
            "<｜tool_call_end｜>",
            "<｜tool_sep｜>",
            '<steptml:invoke name="',
            "</steptml:invoke>",
            '<steptml:parameter name="',
            "</steptml:parameter>",
        ],
    ),
    NoCrashFuzzCase(
        name="minimax-m2-truncated-mid-parameter",
        detector_factory=MinimaxM2Detector,
        tools=COMMON_TOOLS,
        text=(
            'Lead. <minimax:tool_call>\n<invoke name="get_weather">\n'
            '<parameter name="city">Par'
        ),
        markers=[
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            '<invoke name="',
            "</invoke>",
            '<parameter name="',
            "</parameter>",
        ],
    ),
    NoCrashFuzzCase(
        name="minimax-m2-garbage-inside-block",
        detector_factory=MinimaxM2Detector,
        tools=COMMON_TOOLS,
        text="<minimax:tool_call>GARBAGE NO TAG HERE</minimax:tool_call> after",
        markers=[
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            '<invoke name="',
            "</invoke>",
            '<parameter name="',
            "</parameter>",
        ],
    ),
    NoCrashFuzzCase(
        name="minimax-m2-missing-invoke-end",
        detector_factory=MinimaxM2Detector,
        tools=COMMON_TOOLS,
        text=(
            '<minimax:tool_call>\n<invoke name="get_weather">\n'
            '<parameter name="city">Paris</parameter>\n'
        ),
        markers=[
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            '<invoke name="',
            "</invoke>",
            '<parameter name="',
            "</parameter>",
        ],
    ),
] + _registered_no_crash_cases()


@pytest.mark.parametrize("case", VALID_CASES, ids=lambda case: case.name)
def test_streaming_matches_non_stream_under_fuzzed_chunking(
    case: StreamingFuzzCase,
):
    assert_stream_matches_non_stream(case)


@pytest.mark.parametrize("case", NO_CRASH_CASES, ids=lambda case: case.name)
def test_streaming_hard_mode_no_crash(case: NoCrashFuzzCase):
    assert_stream_no_crash(case)


@pytest.mark.parametrize(
    "case",
    [case for case in VALID_CASES if case.clean],
    ids=lambda case: case.name,
)
def test_clean_input_strict_matches_compatibility(case: StreamingFuzzCase):
    assert_clean_input_strict_invariant(case)


@pytest.mark.parametrize(
    "enable_compatibility_mode",
    [True, False],
    ids=["compatibility-enabled", "compatibility-disabled"],
)
@pytest.mark.parametrize("case", VALID_CASES, ids=lambda case: case.name)
def test_mutation_sweep_never_crashes(
    case: StreamingFuzzCase, enable_compatibility_mode: bool
):
    assert_mutants_never_crash(case, enable_compatibility_mode)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
