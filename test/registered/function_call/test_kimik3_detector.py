import json
import sys

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.function_call.kimik3_native import KimiK3ParseError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _make_tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
            },
        ),
    )


def _call_block(tool: str, index: int, args: dict[str, tuple[str, str]]) -> str:
    parts = [f'<|open|>call tool="{tool}" index="{index}"<|sep|>']
    for key, (arg_type, value) in args.items():
        parts.append(
            f'<|open|>argument key="{key}" type="{arg_type}"<|sep|>'
            f"{value}<|close|>argument<|sep|>"
        )
    parts.append("<|close|>call<|sep|>")
    return "".join(parts)


def _chunks(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def _stream(
    detector: KimiK3Detector, chunks: list[str], tools: list[Tool]
) -> tuple[str, list[ToolCallItem]]:
    text = ""
    calls = {}
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        text += result.normal_text
        for item in result.calls:
            if item.tool_index not in calls:
                calls[item.tool_index] = ToolCallItem(
                    tool_index=item.tool_index, name=item.name, parameters=""
                )
            calls[item.tool_index].parameters += item.parameters or ""
    return text, list(calls.values())


def test_detect_and_parse_single_call() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        f"{RESPONSE_OPEN}Let me run it.{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block(
            "python",
            1,
            {"code": ("string", "print(1)"), "opts": ("object", '{"a": 1}')},
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, tools)
    assert result.normal_text == "Let me run it."
    assert len(result.calls) == 1
    assert result.calls[0].name == "python"
    assert json.loads(result.calls[0].parameters) == {
        "code": "print(1)",
        "opts": {"a": 1},
    }


def test_detect_and_parse_no_tools_channel() -> None:
    detector = KimiK3Detector()
    result = detector.detect_and_parse(
        f"{RESPONSE_OPEN}hi there{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
        [_make_tool("python")],
    )
    assert result.normal_text == "hi there"
    assert result.calls == []


def test_detect_and_parse_multiple_calls() -> None:
    detector = KimiK3Detector()
    text = (
        TOOLS_OPEN
        + _call_block("python", 1, {"code": ("string", "a")})
        + _call_block("python", 2, {"code": ("string", "b")})
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert [call.tool_index for call in result.calls] == [0, 1]
    assert json.loads(result.calls[1].parameters) == {"code": "b"}


def test_detect_and_parse_unclosed_tools_section() -> None:
    detector = KimiK3Detector()
    text = TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "x")})
    with pytest.raises(KimiK3ParseError, match="unclosed_channel"):
        detector.detect_and_parse(text, [_make_tool("python")])
    result = detector.detect_and_parse(
        text, [_make_tool("python")], allow_incomplete=True
    )
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {"code": "x"}


def test_attr_unescaping_and_raw_string_args() -> None:
    detector = KimiK3Detector()
    text = (
        f"{TOOLS_OPEN}"
        '<|open|>call tool="a&amp;b" index="1"<|sep|>'
        '<|open|>argument key="q" type="string"<|sep|>'
        "say &quot;hi&quot;<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        f"{TOOLS_CLOSE}"
    )
    result = detector.detect_and_parse(text, [_make_tool("a&b")])
    assert result.calls[0].name == "a&b"
    assert json.loads(result.calls[0].parameters) == {"q": "say &quot;hi&quot;"}


def test_non_string_arg_json_decoding() -> None:
    detector = KimiK3Detector()
    text = (
        TOOLS_OPEN
        + _call_block(
            "python",
            1,
            {
                "n": ("number", "42"),
                "flag": ("boolean", "true"),
                "data": ("object", '{"valid": true}'),
            },
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert json.loads(result.calls[0].parameters) == {
        "n": 42,
        "flag": True,
        "data": {"valid": True},
    }


@pytest.mark.parametrize("chunk_size", [1, 7, 23])
def test_streaming_split_markers(chunk_size: int) -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        f"{RESPONSE_OPEN}Hello!{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block("python", 1, {"code": ("string", "print(2)")})
        + TOOLS_CLOSE
    )
    normal_text, calls = _stream(detector, _chunks(text, chunk_size), tools)
    assert normal_text == "Hello!"
    assert len(calls) == 1
    assert calls[0].name == "python"
    assert json.loads(calls[0].parameters) == {"code": "print(2)"}


def test_streaming_two_calls() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN
        + _call_block("python", 1, {"code": ("string", "a")})
        + _call_block("python", 2, {"code": ("string", "b")})
        + TOOLS_CLOSE
    )
    _, calls = _stream(detector, _chunks(text, 7), tools)
    assert [call.tool_index for call in calls] == [0, 1]
    assert [json.loads(call.parameters) for call in calls] == [
        {"code": "a"},
        {"code": "b"},
    ]


def test_streaming_plain_text_only() -> None:
    detector = KimiK3Detector()
    text, calls = _stream(
        detector, ["just a ", "plain ", "reply"], [_make_tool("python")]
    )
    assert text == "just a plain reply"
    assert calls == []


def test_streaming_bookkeeping_for_serving_layer() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "a")}) + TOOLS_CLOSE
    )
    _stream(detector, _chunks(text, 9), tools)
    assert detector.current_tool_id == 0
    assert detector.prev_tool_call_arr[0] == {
        "name": "python",
        "arguments": {"code": "a"},
    }
    assert json.loads(detector.streamed_args_for_tool[0]) == {"code": "a"}


def test_stream_end_rejects_unclosed_call() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>'
    _, calls = _stream(detector, _chunks(truncated, 7), tools)
    assert calls and calls[0].name == "python"
    with pytest.raises(json.JSONDecodeError):
        json.loads(calls[0].parameters)
    with pytest.raises(KimiK3ParseError, match="unclosed_call"):
        detector.finish(tools)


def test_stream_end_releases_held_back_text() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text, _ = _stream(detector, ["all done", "<"], tools)
    assert text == "all done"
    result = detector.finish(tools)
    assert text + (result.normal_text or "") == "all done<"


def test_stream_end_drops_truncated_marker() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text, _ = _stream(detector, ["all done", "<|open|>"], tools)
    with pytest.raises(KimiK3ParseError, match="unclosed_channel"):
        detector.finish(tools)
    length_detector = KimiK3Detector()
    _stream(length_detector, ["all done", "<|open|>"], tools)
    assert length_detector.finish(tools, allow_incomplete=True).normal_text == ""


def test_detector_capabilities_and_registration() -> None:
    detector = KimiK3Detector()
    assert detector.supports_structural_tag()
    assert not detector.parses_required_natively()
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    assert isinstance(parser.detector, KimiK3Detector)
    assert parser.get_structure_constraint("required") is not None


@pytest.mark.parametrize("size", [1, 7, 1000])
def test_string_delivery_precedes_argument_and_call_close(size):
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    prefix = TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>'
    prefix += '<|open|>argument key="code" type="string"<|sep|>'
    pieces = detector.parse_streaming_increment(prefix, tools).calls
    assert pieces[0].name == "python"
    body = 'line\\quote"\n中文' * 100
    for chunk in _chunks(body, size):
        result = detector.parse_streaming_increment(chunk, tools)
        assert result.calls
        assert len(detector._native.pending) < 32
        pieces.extend(result.calls)
    partial = "".join(item.parameters or "" for item in pieces)
    assert "中文" in partial
    with pytest.raises(json.JSONDecodeError):
        json.loads(partial)
    result = detector.parse_streaming_increment(
        "<|close|>argument<|sep|><|close|>call<|sep|>" + TOOLS_CLOSE, tools
    )
    complete = partial + "".join(item.parameters or "" for item in result.calls)
    assert json.loads(complete) == {"code": body}
    assert not detector.finish(tools).calls


def test_undeclared_call_rejected_before_name_or_arguments():
    detector = KimiK3Detector()
    prefix = TOOLS_OPEN + '<|open|>call tool="unknown" index="1"<|sep|>'
    for char in prefix[:-1]:
        assert not detector.parse_streaming_increment(
            char, [_make_tool("python")]
        ).calls
    with pytest.raises(KimiK3ParseError, match="unknown_tool"):
        detector.parse_streaming_increment(prefix[-1], [_make_tool("python")])


def test_typed_value_never_streamed_before_validation():
    detector = KimiK3Detector()
    prefix = TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>'
    prefix += '<|open|>argument key="code" type="object"<|sep|>'
    initial = detector.parse_streaming_increment(prefix, [_make_tool("python")])
    assert initial.calls
    assert not detector.parse_streaming_increment(
        '"wrong"', [_make_tool("python")]
    ).calls
    with pytest.raises(KimiK3ParseError, match="invalid_argument_type"):
        detector.parse_streaming_increment(
            "<|close|>argument<|sep|>", [_make_tool("python")]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
