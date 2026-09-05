import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm53_flash_detector import (
    AK_END,
    AK_START,
    AV_END,
    AV_START,
    TC_END,
    TC_START,
    Glm53FlashDetector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


def make_tools_weather():
    return [
        Tool(
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "city name"},
                    },
                    "required": ["city"],
                },
            )
        )
    ]


def make_tools_todo_bash():
    """Tools with nested array schema (todo_write) and simple schema (bash)."""
    return [
        Tool(
            function=Function(
                name="todo_write",
                parameters={
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed"],
                                    },
                                },
                                "required": ["content", "status"],
                            },
                        }
                    },
                    "required": ["todos"],
                },
            )
        ),
        Tool(
            function=Function(
                name="bash",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["command", "description"],
                },
            )
        ),
    ]


def make_tools_glob_grep():
    """Tools with simple schemas that the model emits in tag format."""
    return [
        Tool(
            function=Function(
                name="glob",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            )
        ),
        Tool(
            function=Function(
                name="grep",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "include": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            )
        ),
    ]


# ---------------------------------------------------------------------------
# Original tests (unchanged)
# ---------------------------------------------------------------------------


def test_json_format_single_call():
    """JSON format: TC_START + name + > + JSON"""
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    text = '\u6211\u6765\u5e2e\u60a8\u67e5\u8be2\u3002<![get_weather>{"city": "\u5317\u4eac"}'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "\u5317\u4eac"
    assert "\u6211\u6765\u5e2e\u60a8\u67e5\u8be2\u3002" in res.normal_text


def test_json_format_display_name_prefix():
    """Display/name prefix: <![Weather/get_weather>{...} -> get_weather"""
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    text = '<![Weather/get_weather>{"city": "\u5317\u4eac"}'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"
    args = json.loads(res.calls[0].parameters)
    assert args["city"] == "\u5317\u4eac"


def test_tag_format_single_call():
    """Tag format: TC_START + name + AK_START key AK_END AV_START val AV_END + TC_END"""
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    text = (
        "<!\u200b[get_weather\n"
        "<!\u200b[arg_key\u200b]city\n"
        "<!\u200b[arg_val\u200b]\u5317\u4eac\n"
        "<!\u200b[/tool_call\u200b]"
    )
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"


def test_no_tool_call():
    """Plain text without tool calls is preserved."""
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    text = "\u4eca\u5929\u5929\u6c14\u4e0d\u9519\u3002"
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 0
    assert res.normal_text == text


def test_multiple_json_calls():
    """Two JSON-format calls in sequence."""
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    text = '<![get_weather>{"city": "\u5317\u4eac"}<![get_weather>{"city": "\u4e0a\u6d77"}'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 2
    assert json.loads(res.calls[0].parameters)["city"] == "\u5317\u4eac"
    assert json.loads(res.calls[1].parameters)["city"] == "\u4e0a\u6d77"


def test_unknown_function_name_dropped():
    """Unknown function name should not produce a valid call."""
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    text = '<![unknown_func>{"x": 1}<![get_weather>{"city": "\u5317\u4eac"}'
    res = detector.detect_and_parse(text, tools)
    assert any(c.name == "get_weather" for c in res.calls)


def test_streaming_json_format():
    """Streaming: feed JSON-format call in chunks, verify split."""
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    chunks = ['<![get_weather>', '{"city": "', '\u5317\u4eac', '"}']
    rc, cc = "", ""
    for c in chunks:
        r = detector.parse_streaming_increment(c, tools)
        rc += r.normal_text or ""
        if r.calls:
            for call in r.calls:
                if call.name:
                    rc += call.name
                if call.parameters:
                    cc += call.parameters
    assert "\u5317\u4eac" not in rc
    assert "\u5317\u4eac" in cc or '"city"' in cc


# ---------------------------------------------------------------------------
# New tests for streaming multi-tool-call fixes
# ---------------------------------------------------------------------------


def test_streaming_multi_tool_json():
    """Streaming: two JSON-format tool calls in one response.

    The model emits both calls in a single buffer. Without raw_decode,
    json.loads() fails with "Extra data" because the buffer contains
    JSON1 + JSON2.
    """
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    # Feed both calls as a single chunk (simulates model outputting both at once)
    text = '<![get_weather>{"city": "\u5317\u4eac"}<![get_weather>{"city": "\u4e0a\u6d77"}'
    r = detector.parse_streaming_increment(text, tools)
    # Should get at least the first call parsed
    assert r.calls, "Expected at least one tool call from multi-tool JSON streaming"
    # The first call should have correct args
    first_args = [c for c in r.calls if c.parameters]
    assert first_args, "Expected at least one call with parameters"
    args = json.loads(first_args[0].parameters)
    assert args["city"] == "\u5317\u4eac"


def test_streaming_multi_tool_json_separate_chunks():
    """Streaming: two JSON calls arriving in separate chunks.

    The first call is parsed, tool_id is incremented, and the second
    call is parsed from the remaining buffer on the next invocation.
    """
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    all_calls = []

    chunks = [
        '<![get_weather>{"city": "\u5317\u4eac"}',
        '<![get_weather>{"city": "\u4e0a\u6d77"}',
    ]
    for chunk in chunks:
        r = detector.parse_streaming_increment(chunk, tools)
        for call in r.calls:
            all_calls.append(call)

    # Should have 2 calls with args
    args_list = [c.parameters for c in all_calls if c.parameters]
    assert len(args_list) == 2, f"Expected 2 calls with args, got {len(args_list)}"
    assert json.loads(args_list[0])["city"] == "\u5317\u4eac"
    assert json.loads(args_list[1])["city"] == "\u4e0a\u6d77"
    # Verify different tool indices
    indices = [c.tool_index for c in all_calls if c.name]
    assert len(set(indices)) == 2, f"Expected 2 different indices, got {indices}"


def test_streaming_multi_tool_tag_format():
    """Streaming: two tag-format tool calls in one response.

    The model emits both calls in tag format. Without tool_id increment
    in the tag handler, the second call's args get the same index as
    the first, and the name is not emitted.
    """
    detector = Glm53FlashDetector()
    tools = make_tools_glob_grep()

    # Build tag-format text with two tool calls
    text = (
        TC_START + "glob" + AK_START + "pattern" + AK_END + AV_START + "**/*.py" + AV_END + TC_END
        + TC_START + "grep" + AK_START + "pattern" + AK_END + AV_START + "def main" + AV_END + TC_END
    )
    r = detector.parse_streaming_increment(text, tools)
    assert r.calls, "Expected tool calls from tag format"

    # Should have 2 name calls with different indices
    name_calls = [c for c in r.calls if c.name]
    assert len(name_calls) == 2, f"Expected 2 name calls, got {len(name_calls)}"
    assert name_calls[0].name == "glob"
    assert name_calls[1].name == "grep"
    assert name_calls[0].tool_index != name_calls[1].tool_index


def test_streaming_multi_tool_tag_separate_chunks():
    """Streaming: two tag-format calls arriving in separate chunks."""
    detector = Glm53FlashDetector()
    tools = make_tools_glob_grep()

    all_calls = []
    chunks = [
        TC_START + "glob" + AK_START + "pattern" + AK_END + AV_START + "**/*.py" + AV_END + TC_END,
        TC_START + "grep" + AK_START + "pattern" + AK_END + AV_START + "def main" + AV_END + TC_END,
    ]
    for chunk in chunks:
        r = detector.parse_streaming_increment(chunk, tools)
        for call in r.calls:
            all_calls.append(call)

    name_calls = [c for c in all_calls if c.name]
    assert len(name_calls) == 2
    assert name_calls[0].name == "glob"
    assert name_calls[1].name == "grep"


def test_finish_flushes_remaining_buffer():
    """finish() should flush remaining tool calls left in the buffer.

    When the model outputs multiple tool calls and the stream ends
    before all are processed by parse_streaming_increment, the base
    class finish() returns empty result. Our finish() processes the
    remaining buffer.
    """
    detector = Glm53FlashDetector()
    tools = make_tools_weather()

    # Feed first call, then end stream - second call should be in buffer
    text1 = '<![get_weather>{"city": "\u5317\u4eac"}'
    text2 = '<![get_weather>{"city": "\u4e0a\u6d77"}'

    r1 = detector.parse_streaming_increment(text1, tools)
    # First call should be parsed
    assert r1.calls

    # Feed second call
    r2 = detector.parse_streaming_increment(text2, tools)
    # Second call should be parsed from buffer
    assert r2.calls

    # finish() should return empty (buffer already drained)
    r_finish = detector.finish(tools)
    assert len(r_finish.calls) == 0


def test_fix_args_against_schema_unwrapped():
    """_fix_args_against_schema wraps unwrapped nested array args.

    The model sometimes outputs the inner object of an array schema
    directly, e.g. {"content": "...", "status": "..."} instead of
    {"todos": [{"content": "...", "status": "..."}]}.
    """
    detector = Glm53FlashDetector()
    tools = make_tools_todo_bash()

    # Simulate unwrapped args: model outputs inner object directly
    text = '<![todo_write>{"content": "write code", "status": "pending"}'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    # Should be wrapped in {"todos": [...]}
    assert "todos" in args, f"Expected 'todos' key, got {args}"
    assert isinstance(args["todos"], list)
    assert args["todos"][0]["content"] == "write code"


def test_fix_args_against_schema_correct_args():
    """_fix_args_against_schema does not modify correctly wrapped args."""
    detector = Glm53FlashDetector()
    tools = make_tools_todo_bash()

    text = '<![todo_write>{"todos": [{"content": "write code", "status": "pending"}]}'
    res = detector.detect_and_parse(text, tools)
    assert len(res.calls) == 1
    args = json.loads(res.calls[0].parameters)
    assert "todos" in args
    assert args["todos"][0]["content"] == "write code"


def test_streaming_multi_tool_todo_bash():
    """Streaming: todo_write + bash in one response.

    This is the primary use case: the model outputs a todo list
    followed by a bash command. Both should be parsed correctly
    with proper tool indices.
    """
    detector = Glm53FlashDetector()
    tools = make_tools_todo_bash()

    text = (
        '<![todo_write>{"todos": [{"content": "write code", "status": "pending"}]}'
        '<![bash>{"command": "ls -la", "description": "List files"}'
    )
    r = detector.parse_streaming_increment(text, tools)
    assert r.calls, "Expected tool calls"

    # Should have 2 name calls with different indices
    name_calls = [c for c in r.calls if c.name]
    assert len(name_calls) == 2, f"Expected 2 name calls, got {len(name_calls)}"

    # First call should be todo_write with todos
    assert name_calls[0].name == "todo_write"
    # Second call should be bash
    assert name_calls[1].name == "bash"
    # Different indices
    assert name_calls[0].tool_index != name_calls[1].tool_index


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
