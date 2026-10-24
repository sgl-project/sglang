import json
import sys

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    CALL_CLOSE,
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


def _reassemble(calls: list[ToolCallItem]) -> list[tuple[int, str, str]]:
    """Fold streaming deltas back into one entry per tool call.

    This is what a client does: concatenate every ``parameters`` delta for a
    tool index and use the name from whichever delta carried it.
    """
    order: list[int] = []
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    for call in calls:
        if call.tool_index not in args:
            order.append(call.tool_index)
            args[call.tool_index] = ""
        if call.name:
            names[call.tool_index] = call.name
        args[call.tool_index] += call.parameters or ""
    return [(index, names.get(index, ""), args[index]) for index in order]


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
    assert _reassemble(calls) == [
        (0, "python", json.dumps({"code": "print(2)"}, ensure_ascii=False))
    ]


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
    assert _reassemble(calls) == [
        (0, "python", json.dumps({"code": "a"}, ensure_ascii=False)),
        (1, "python", json.dumps({"code": "b"}, ensure_ascii=False)),
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


def test_stream_end_reports_truncated_tools_section(caplog) -> None:
    """A tools section cut off before its closing tag must still be reported.

    The tool name is emitted as soon as the call header is complete -- that is
    the point of streaming them incrementally -- but the truncated arguments
    are never invented, and ``finish`` still flags the section.
    """
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>'
    text, calls = _stream(detector, _chunks(truncated, 7), tools)
    assert _reassemble(calls) == [(0, "python", "{")]
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


def test_streaming_emits_name_before_call_completes() -> None:
    """The name must not wait for the whole call.

    Before the fix, ``parse_streaming_increment`` only matched a *complete*
    call, so a client received nothing at all -- not even the tool name --
    until the closing tag arrived.
    """
    detector = KimiK3Detector()
    tools = [_make_tool("write_file")]
    partial = TOOLS_OPEN + '<|open|>call tool="write_file" index="1"<|sep|>'
    result = detector.parse_streaming_increment(partial, tools)
    assert [(c.name, c.parameters) for c in result.calls] == [
        ("write_file", ""),
        (None, "{"),
    ]


def test_streaming_arguments_are_not_buffered_until_call_close() -> None:
    """A long argument must stream out while it is being generated."""
    detector = KimiK3Detector()
    tools = [_make_tool("write_file")]
    body = TOOLS_OPEN + '<|open|>call tool="write_file" index="1"<|sep|>'
    body += '<|open|>argument key="content" type="string"<|sep|>'
    body += "line of source code\n" * 200

    seen = 0
    step = 40
    for start in range(0, len(body), step):
        result = detector.parse_streaming_increment(body[start : start + step], tools)
        seen += sum(len(c.parameters or "") for c in result.calls)

    # The call is still open, yet most of the payload has already shipped.
    assert seen > len(body) * 0.8


@pytest.mark.parametrize("chunk_size", [1, 3, 17, 512])
@pytest.mark.parametrize(
    "value",
    [
        "plain",
        'quotes " and \\ backslash',
        "new\nline\ttab\r\n",
        "unicode: 中文 é ü 🚀",
        'a literal "<|close|>argument<|sep|>" inside text',
        '<|open|>argument key="evil" type="string"<|sep|>',
        "",
    ],
)
def test_streamed_deltas_match_non_streaming_bytes(chunk_size: int, value: str) -> None:
    """Reassembled deltas must equal the one-shot output byte for byte.

    The serving layer slices ``streamed_args_for_tool`` against
    ``json.dumps(prev_tool_call_arr[i]["arguments"])``, so any divergence in key
    order, separators or escaping would corrupt the final tool call.
    """
    detector = KimiK3Detector()
    tools = [_make_tool("write_file")]
    text = (
        TOOLS_OPEN
        + _call_block(
            "write_file",
            1,
            {"content": ("string", value), "path": ("string", "a/b.py")},
        )
        + TOOLS_CLOSE
    )
    expected = detector.detect_and_parse(text, tools).calls[0].parameters

    detector = KimiK3Detector()
    _, calls = _stream(detector, _chunks(text, chunk_size), tools)

    streamed = _reassemble(calls)[0][2]
    assert streamed == expected
    # The detector's own bookkeeping must agree with what it emitted.
    assert detector.streamed_args_for_tool[0] == expected
    assert (
        json.dumps(detector.prev_tool_call_arr[0]["arguments"], ensure_ascii=False)
        == expected
    )


@pytest.mark.parametrize("chunk_size", [1, 5, 29])
def test_streaming_mixed_argument_types(chunk_size: int) -> None:
    """Non-string arguments are JSON-decoded and emitted at their close tag."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN
        + _call_block(
            "python",
            1,
            {
                "code": ("string", "print(2)"),
                "n": ("number", "42"),
                "flag": ("boolean", "true"),
                "opts": ("object", '{"deep": [1, 2]}'),
            },
        )
        + TOOLS_CLOSE
    )
    expected = detector.detect_and_parse(text, tools).calls[0].parameters

    detector = KimiK3Detector()
    _, calls = _stream(detector, _chunks(text, chunk_size), tools)
    streamed = _reassemble(calls)[0][2]
    assert streamed == expected
    assert json.loads(streamed) == {
        "code": "print(2)",
        "n": 42,
        "flag": True,
        "opts": {"deep": [1, 2]},
    }


def test_streaming_call_without_arguments() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("ping")]
    text = (
        TOOLS_OPEN
        + '<|open|>call tool="ping" index="1"<|sep|>'
        + CALL_CLOSE
        + TOOLS_CLOSE
    )
    expected = detector.detect_and_parse(text, tools).calls[0].parameters

    detector = KimiK3Detector()
    _, calls = _stream(detector, _chunks(text, 4), tools)
    streamed = _reassemble(calls)[0][2]
    assert streamed == expected == "{}"


def test_detector_capabilities_and_registration() -> None:
    detector = KimiK3Detector()
    assert detector.supports_structural_tag()
    assert not detector.parses_required_natively()
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    assert isinstance(parser.detector, KimiK3Detector)
    assert parser.get_structure_constraint("required") is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
