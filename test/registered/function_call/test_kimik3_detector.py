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
    calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        text += result.normal_text
        calls.extend(result.calls)
    return text, calls


def _assemble(calls: list[ToolCallItem]) -> dict[int, dict]:
    """Reassemble incremental streaming deltas per tool call.

    Returns {tool_index: {"name": ..., "parameters": <concatenated deltas>}}.
    """
    by_id: dict[int, dict] = {}
    for call in calls:
        entry = by_id.setdefault(call.tool_index, {"name": None, "parameters": ""})
        if call.name:
            entry["name"] = call.name
        entry["parameters"] += call.parameters
    return by_id


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
    result = detector.detect_and_parse(text, [_make_tool("python")])
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
    result = detector.detect_and_parse(text, [_make_tool("python")])
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
                "bad": ("object", "{not json"),
            },
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert json.loads(result.calls[0].parameters) == {
        "n": 42,
        "flag": True,
        "bad": "{not json",
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
    assembled = _assemble(calls)
    assert list(assembled) == [0]
    assert assembled[0]["name"] == "python"
    # Concatenated deltas must be byte-identical to the non-streaming output.
    assert assembled[0]["parameters"] == json.dumps(
        {"code": "print(2)"}, ensure_ascii=False
    )


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
    assembled = _assemble(calls)
    assert sorted(assembled) == [0, 1]
    assert [assembled[i]["parameters"] for i in (0, 1)] == [
        json.dumps({"code": "a"}, ensure_ascii=False),
        json.dumps({"code": "b"}, ensure_ascii=False),
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


def test_streaming_name_emitted_at_call_open() -> None:
    """The tool name must be emitted as soon as the call open tag completes,
    before any argument bytes arrive."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    open_tag = '<|open|>call tool="python" index="1"<|sep|>'
    text, calls = _stream(detector, _chunks(TOOLS_OPEN + open_tag, 7), tools)
    assert text == ""
    assert len(calls) == 1
    assert calls[0].tool_index == 0
    assert calls[0].name == "python"
    assert calls[0].parameters == ""


def test_streaming_arguments_flow_before_call_close() -> None:
    """Regression test for the buffered-until-close bug: argument deltas must
    arrive while the call is still open, not only after <|close|>call."""
    detector = KimiK3Detector()
    tools = [_make_tool("write_file")]
    body = "x" * 500  # long enough that several chunks carry argument text
    block = _call_block("write_file", 1, {"content": ("string", body)})
    call_close = "<|close|>call<|sep|>"
    head, _ = block.split(call_close)
    # Stream everything except the final <|close|>call<|sep|>.
    text, calls = _stream(detector, _chunks(TOOLS_OPEN + head, 11), tools)
    assert text == ""
    streamed_params = "".join(call.parameters for call in calls)
    assert streamed_params != "", "no argument deltas arrived before call close"
    assert streamed_params.startswith('{"content": "')
    # Now close the call and check final assembly.
    _, tail_calls = _stream(detector, [call_close, TOOLS_CLOSE], tools)
    calls.extend(tail_calls)
    assembled = _assemble(calls)
    assert assembled[0]["name"] == "write_file"
    assert assembled[0]["parameters"] == json.dumps(
        {"content": body}, ensure_ascii=False
    )


@pytest.mark.parametrize("chunk_size", [1, 7, 23])
def test_streaming_deltas_byte_identical_with_tricky_content(
    chunk_size: int,
) -> None:
    """Concatenated deltas must equal json.dumps(arguments) exactly, even with
    quotes, newlines, unicode, and text that looks like a partial marker."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    args = {
        "code": 'print("héllo\\nworld") 😀',
        "tricky": "ends with partial <|close|>argument plus more",
        "n": ("number", "42"),
        "flag": ("boolean", "true"),
        "obj": ("object", '{"a": [1, 2]}'),
        "bad": ("object", "{not json"),
    }
    block = _call_block(
        "python",
        1,
        {
            "code": ("string", args["code"]),
            "tricky": ("string", args["tricky"]),
            "n": args["n"],
            "flag": args["flag"],
            "obj": args["obj"],
            "bad": args["bad"],
        },
    )
    expected = {
        "code": args["code"],
        "tricky": args["tricky"],
        "n": 42,
        "flag": True,
        "obj": {"a": [1, 2]},
        "bad": "{not json",
    }
    text = TOOLS_OPEN + block + TOOLS_CLOSE
    _, calls = _stream(detector, _chunks(text, chunk_size), tools)
    assembled = _assemble(calls)
    assert assembled[0]["parameters"] == json.dumps(expected, ensure_ascii=False)
    # ... and it matches the non-streaming path exactly.
    non_streaming = KimiK3Detector().detect_and_parse(text, tools)
    assert assembled[0]["parameters"] == non_streaming.calls[0].parameters


def test_streaming_non_string_arg_buffered_until_close() -> None:
    """Non-string argument values are decoded at their close tag; the value
    must not be streamed before it."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    block = _call_block("python", 1, {"code": ("string", "a"), "n": ("number", "42")})
    arg_close = "<|close|>argument<|sep|>"
    parts = block.split(arg_close)
    assert len(parts) == 3  # two arguments, each with a close tag
    head = parts[0] + arg_close + parts[1]  # up to, excluding, n's close tag
    tail = arg_close + parts[2]
    # Stream up to (excluding) the number argument's close tag.
    _, calls = _stream(detector, _chunks(TOOLS_OPEN + head, 9), tools)
    assembled = _assemble(calls)
    assert assembled[0]["parameters"].endswith(', "n": ')
    # After the close tag the decoded value streams as one delta.
    _, tail_calls = _stream(detector, [arg_close + tail], tools)
    calls.extend(tail_calls)
    assembled = _assemble(calls)
    assert json.loads(assembled[0]["parameters"]) == {"code": "a", "n": 42}


def test_streaming_zero_arg_call() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("ping")]
    text = (
        TOOLS_OPEN
        + '<|open|>call tool="ping" index="1"<|sep|><|close|>call<|sep|>'
        + TOOLS_CLOSE
    )
    _, calls = _stream(detector, _chunks(text, 5), tools)
    assembled = _assemble(calls)
    assert assembled[0]["name"] == "ping"
    assert assembled[0]["parameters"] == "{}"


def test_stream_end_reports_truncated_tools_section(caplog) -> None:
    """A tools section cut off before its closing tag used to vanish at
    end-of-stream: no call, no text, no log. It must at least be reported."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>'
    text, calls = _stream(detector, _chunks(truncated, 7), tools)
    # The name is emitted as soon as the call open tag completes, even though
    # the call itself never completes.
    assert [call.name for call in calls] == ["python"]
    assert [call.parameters for call in calls] == [""]
    with caplog.at_level("WARNING", logger="sglang.srt.function_call.kimik3_detector"):
        result = detector.finish(tools)
    assert result.calls == []
    assert TOOLS_OPEN not in (result.normal_text or "")
    assert "no complete tool call" in caplog.text


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
    result = detector.finish(tools)
    assert text + (result.normal_text or "") == "all done"


def test_detector_capabilities_and_registration() -> None:
    detector = KimiK3Detector()
    assert detector.supports_structural_tag()
    assert not detector.parses_required_natively()
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    assert isinstance(parser.detector, KimiK3Detector)
    assert parser.get_structure_constraint("required") is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
