import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.spark3_detector import Spark3Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _xml(name: str, arguments: list[tuple[str, str]]) -> str:
    pairs = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in arguments
    )
    return f"<tool_call>{name}{pairs}</tool_call>"


def _tools() -> list[Tool]:
    return [
        Tool(
            type="function",
            function=Function(
                name="set_state",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "active": {"type": "boolean"},
                        "items": {"type": "array"},
                        "metadata": {"type": "object"},
                    },
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="now",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]


def test_spark3_parser_is_registered():
    assert FunctionCallParser.ToolCallParserEnum["spark"] is Spark3Detector


def test_nonstream_parses_multiple_calls_and_preserves_normal_text():
    text = (
        "before"
        + _xml(
            "set_state",
            [
                ("name", "上海"),
                ("count", "42"),
                ("ratio", "2.5"),
                ("active", "1"),
                ("items", '["a", "b"]'),
                ("metadata", '{"source":"spark"}'),
            ],
        )
        + "middle"
        + _xml("now", [])
        + "after"
    )

    result = Spark3Detector().detect_and_parse(text, _tools())

    assert result.normal_text == "beforemiddleafter"
    assert [call.tool_index for call in result.calls] == [0, 1]
    assert [call.name for call in result.calls] == ["set_state", "now"]
    assert json.loads(result.calls[0].parameters) == {
        "name": "上海",
        "count": 42,
        "ratio": 2.5,
        "active": True,
        "items": ["a", "b"],
        "metadata": {"source": "spark"},
    }
    assert json.loads(result.calls[1].parameters) == {}


def test_null_and_conversion_fallbacks_match_spark3_protocol():
    text = _xml(
        "set_state",
        [
            ("name", "null"),
            ("count", "not-an-int"),
            ("active", "false"),
            ("undeclared", "42"),
        ],
    )

    result = Spark3Detector().detect_and_parse(text, _tools())

    assert json.loads(result.calls[0].parameters) == {
        "name": None,
        "count": "not-an-int",
        "active": False,
        "undeclared": "42",
    }


def test_streaming_character_chunks_match_nonstream_result():
    text = (
        "answer:"
        + _xml("set_state", [("count", "42"), ("active", "0")])
        + _xml("now", [])
        + "done"
    )
    detector = Spark3Detector()
    normal_parts = []
    calls = []

    for character in text:
        result = detector.parse_streaming_increment(character, _tools())
        normal_parts.append(result.normal_text)
        calls.extend(result.calls)
    end = detector.finish(_tools())
    normal_parts.append(end.normal_text)
    calls.extend(end.calls)

    assert "".join(normal_parts) == "answer:done"
    assert [call.tool_index for call in calls] == [0, 1]
    assert json.loads(calls[0].parameters) == {"count": 42, "active": False}
    assert json.loads(calls[1].parameters) == {}
    assert detector.prev_tool_call_arr == [
        {"name": "set_state", "arguments": {"count": 42, "active": False}},
        {"name": "now", "arguments": {}},
    ]
    assert detector.streamed_args_for_tool == [
        '{"count":42,"active":false}',
        "{}",
    ]


def test_malformed_block_is_text_and_unknown_tool_honors_policy():
    malformed = "x<tool_call></tool_call>y"
    result = Spark3Detector().detect_and_parse(malformed, _tools())
    assert result.normal_text == malformed
    assert result.calls == []

    unknown = _xml("missing", [("value", "1")])
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        result = Spark3Detector().detect_and_parse(unknown, _tools())
        assert result.normal_text == ""
        assert result.calls == []
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        result = Spark3Detector().detect_and_parse(unknown, _tools())
        assert result.calls[0].name == "missing"
        assert json.loads(result.calls[0].parameters) == {"value": "1"}


def test_stream_end_flushes_partial_marker_and_required_stays_native():
    detector = Spark3Detector()
    result = detector.parse_streaming_increment("plain<tool_", _tools())
    assert result.normal_text == "plain"
    assert detector.finish(_tools()).normal_text == "<tool_"

    truncated = Spark3Detector()
    result = truncated.parse_streaming_increment(
        "plain<tool_call>set_state<arg_key>count</arg_key>", _tools()
    )
    assert result.normal_text == "plain"
    assert truncated.finish(_tools()).normal_text == ""

    assert detector.supports_structural_tag() is False
    assert detector.parses_required_natively() is True
    assert (
        FunctionCallParser(_tools(), "spark").get_structure_constraint("required")
        is None
    )
