import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.dots_detector import DotsToolDetector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import Qwen3Detector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tool(name: str, properties: dict) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description="test tool",
            parameters={"type": "object", "properties": properties},
        ),
    )


def test_dots_parsers_are_registered():
    assert ReasoningParser.DetectorMap["dots"] is Qwen3Detector
    assert FunctionCallParser.ToolCallParserEnum["dots"] is DotsToolDetector


def test_dots_reasoning_uses_qwen3_format():
    parser = ReasoningParser("dots", stream_reasoning=False, force_reasoning=True)
    reasoning, content = parser.parse_non_stream(
        "Need to inspect inputs.</think>Final answer"
    )
    assert reasoning == "Need to inspect inputs."
    assert content == "Final answer"


def test_non_stream_xml_converts_schema_types_and_resolves_ref():
    tool = Tool(
        type="function",
        function=Function(
            name="set_location",
            description="Set location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"$ref": "#/$defs/Location"},
                    "days": {"type": "integer"},
                    "include_weather": {"type": "boolean"},
                },
                "$defs": {
                    "Location": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }
                },
            },
        ),
    )
    parser = FunctionCallParser([tool], "dots")
    text = (
        "ok<dots_function_call>"
        '<invoke name="set_location">'
        '<parameter name="location">{"city": "Shanghai"}</parameter>'
        '<parameter name="days">3</parameter>'
        '<parameter name="include_weather">true</parameter>'
        "</invoke>"
        "</dots_function_call>"
    )

    normal_text, calls = parser.parse_non_stream(text)

    assert normal_text == "ok"
    assert len(calls) == 1
    assert calls[0].name == "set_location"
    assert json.loads(calls[0].parameters) == {
        "location": {"city": "Shanghai"},
        "days": 3,
        "include_weather": True,
    }


def test_non_stream_supports_multiple_invokes_and_json_fallback():
    tools = [
        _tool("search", {"query": {"type": "string"}}),
        _tool("open", {"id": {"type": "integer"}}),
    ]
    parser = FunctionCallParser(tools, "dots")
    text = (
        "<dots_function_call>"
        '<invoke name="search"><parameter name="query">chairs</parameter></invoke>'
        '<invoke name="open"><parameter name="id">7</parameter></invoke>'
        "</dots_function_call>"
        '<dots_function_call>{"name":"search","arguments":{"query":"tables"}}'
        "</dots_function_call>"
    )

    _, calls = parser.parse_non_stream(text)

    assert [call.name for call in calls] == ["search", "open", "search"]
    assert [json.loads(call.parameters) for call in calls] == [
        {"query": "chairs"},
        {"id": 7},
        {"query": "tables"},
    ]


def test_streaming_buffers_partial_marker_and_emits_all_complete_calls():
    tools = [_tool("search", {"query": {"type": "string"}})]
    detector = DotsToolDetector()
    chunks = [
        "visible<dots_func",
        (
            "tion_call>"
            '<invoke name="search"><parameter name="query">chairs</parameter></invoke>'
            "</dots_function_call>"
            "<dots_function_call>"
            '<invoke name="search"><parameter name="query">tables</parameter></invoke>'
            "</dots_function_call>"
        ),
    ]

    results = [detector.parse_streaming_increment(chunk, tools) for chunk in chunks]

    assert "".join(result.normal_text for result in results) == "visible"
    calls = [call for result in results for call in result.calls]
    assert [call.tool_index for call in calls] == [0, 1]
    assert [json.loads(call.parameters) for call in calls] == [
        {"query": "chairs"},
        {"query": "tables"},
    ]
