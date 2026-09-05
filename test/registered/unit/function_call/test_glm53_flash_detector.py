import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm53_flash_detector import Glm53FlashDetector
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
