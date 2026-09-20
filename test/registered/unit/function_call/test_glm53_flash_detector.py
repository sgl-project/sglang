"""Unit tests for the GLM-5.3-Flash tool call detector.

Marker ground truth, read from zai-org/GLM-5.3-Flash ``tokenizer.json``
(``added_tokens``)::

    154843 '<tool_call>'    154844 '</tool_call>'
    154847 '<arg_key>'      154848 '</arg_key>'
    154849 '<arg_value>'    154850 '</arg_value>'

The 3-char ``<!['` is *base-vocab* token 75459 and is **not** a tool-call
marker. Earlier revisions of this test file used ``<!['` as the start marker,
which never matches the detector and made every assertion fail.
"""

import json
import sys

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
from sglang.srt.parser.template_detection import (
    TOOL_CALL_PARSER_RULES,
    ReasoningToggleConfig,
    detect_tool_call_parser,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


def make_tools_two():
    return [
        make_tools_weather()[0],
        Tool(
            function=Function(
                name="get_time",
                parameters={
                    "type": "object",
                    "properties": {"tz": {"type": "string"}},
                },
            )
        ),
    ]


def make_tools_search():
    return [
        Tool(
            function=Function(
                name="search",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "fuzzy": {"type": "boolean"},
                    },
                },
            )
        )
    ]


def _collect(res, names, params):
    """Accumulate a StreamingParseResult into the (names, params) strings."""
    for call in res.calls:
        if call.name:
            names += call.name
        if call.parameters:
            params += call.parameters
    return names, params


# ---------------------------------------------------------------------------
# Marker constants
# ---------------------------------------------------------------------------


def test_marker_constants_match_tokenizer():
    """The constants must be the real added tokens of GLM-5.3-Flash."""
    assert TC_START == "<tool_call>"
    assert TC_END == "</tool_call>"
    assert AK_START == "<arg_key>"
    assert AK_END == "</arg_key>"
    assert AV_START == "<arg_value>"
    assert AV_END == "</arg_value>"


def test_detector_uses_tool_call_tokens():
    detector = Glm53FlashDetector()
    assert detector.bot_token == TC_START
    assert detector.eot_token == TC_END


def test_has_tool_call():
    detector = Glm53FlashDetector()
    assert detector.has_tool_call('<tool_call>get_weather>{"city": "Beijing"}')
    assert not detector.has_tool_call('<![get_weather>{"city": "Beijing"}')


def test_three_char_marker_is_not_a_tool_call():
    """The 3-char '<![' (base-vocab token 75459) must not trigger a call."""
    detector = Glm53FlashDetector()
    text = '<![get_weather>{"city": "Beijing"}'
    res = detector.detect_and_parse(text, make_tools_weather())
    assert res.calls == []
    assert res.normal_text == text


# ---------------------------------------------------------------------------
# Non-streaming: detect_and_parse
# ---------------------------------------------------------------------------


def test_no_tool_call_preserves_text():
    detector = Glm53FlashDetector()
    text = "今天天气不错。"
    res = detector.detect_and_parse(text, make_tools_weather())
    assert res.calls == []
    assert res.normal_text == text


def test_json_format_single_call():
    """JSON format: <tool_call>name>{"key": "value"}"""
    detector = Glm53FlashDetector()
    text = '我来帮您查询。<tool_call>get_weather>{"city": "北京"}'
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"
    assert res.calls[0].tool_index == 0
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}
    assert res.normal_text == "我来帮您查询。"


def test_json_format_display_name_prefix():
    """`Weather/get_weather` resolves to the registered `get_weather`."""
    detector = Glm53FlashDetector()
    text = '<tool_call>Weather/get_weather>{"city": "北京"}'
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}


def test_json_format_case_insensitive_name():
    detector = Glm53FlashDetector()
    text = '<tool_call>GET_WEATHER>{"city": "北京"}'
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"


def test_multiple_json_calls():
    detector = Glm53FlashDetector()
    text = (
        '<tool_call>get_weather>{"city": "北京"}<tool_call>get_weather>{"city": "上海"}'
    )
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 2
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}
    assert json.loads(res.calls[1].parameters) == {"city": "上海"}
    assert res.normal_text == ""


def test_unknown_function_dropped_when_ambiguous():
    """An unregistered name is dropped rather than mapped onto some other tool."""
    detector = Glm53FlashDetector()
    text = '<tool_call>not_a_tool>{"x": 1}<tool_call>get_time>{"tz": "UTC"}'
    res = detector.detect_and_parse(text, make_tools_two())
    assert [c.name for c in res.calls] == ["get_time"]


def test_unknown_function_falls_back_to_single_tool():
    """With exactly one tool registered, an unmatched name falls back to it."""
    detector = Glm53FlashDetector()
    text = '<tool_call>Weather>{"city": "北京"}'
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"


def test_json_args_wrapped_in_prose():
    """Malformed args are recovered by extracting the outermost {...}."""
    detector = Glm53FlashDetector()
    text = '<tool_call>get_weather>here: {"city": "北京"} thanks'
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}


# ---------------------------------------------------------------------------
# Non-streaming: tag format
# ---------------------------------------------------------------------------


def test_tag_format_single_call():
    detector = Glm53FlashDetector()
    text = (
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>北京</arg_value>"
        "</tool_call>"
    )
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}


def test_tag_format_multiline():
    """The model emits the name and each arg pair on separate lines."""
    detector = Glm53FlashDetector()
    text = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n"
        "<arg_value>北京</arg_value>\n"
        "</tool_call>"
    )
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 1
    assert res.calls[0].name == "get_weather"
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}


def test_tag_format_typed_values():
    """JSON-scalar values are decoded to their Python type."""
    detector = Glm53FlashDetector()
    text = (
        "<tool_call>search"
        "<arg_key>query</arg_key><arg_value>北京</arg_value>"
        "<arg_key>limit</arg_key><arg_value>3</arg_value>"
        "<arg_key>fuzzy</arg_key><arg_value>true</arg_value>"
        "</tool_call>"
    )
    res = detector.detect_and_parse(text, make_tools_search())
    assert len(res.calls) == 1
    assert json.loads(res.calls[0].parameters) == {
        "query": "北京",
        "limit": 3,
        "fuzzy": True,
    }


def test_tag_format_multiple_calls():
    detector = Glm53FlashDetector()
    text = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>北京</arg_value></tool_call>"
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>上海</arg_value></tool_call>"
    )
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 2
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}
    assert json.loads(res.calls[1].parameters) == {"city": "上海"}


def test_tag_format_then_json_format():
    """A tag-format block followed by a JSON-format call in the same response."""
    detector = Glm53FlashDetector()
    text = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>北京</arg_value></tool_call>"
        '<tool_call>get_weather>{"city": "上海"}'
    )
    res = detector.detect_and_parse(text, make_tools_weather())
    assert len(res.calls) == 2
    assert json.loads(res.calls[0].parameters) == {"city": "北京"}
    assert json.loads(res.calls[1].parameters) == {"city": "上海"}
    assert res.normal_text == ""


def test_structural_tag_is_disabled():
    detector = Glm53FlashDetector()
    assert detector.supports_structural_tag() is False
    assert detector.get_structural_tag() is None
    assert detector.structure_info() is None
    assert detector.get_structural_tag_name() == "glm5_3_flash"


# ---------------------------------------------------------------------------
# Streaming: parse_streaming_increment
# ---------------------------------------------------------------------------


def test_streaming_json_format_split_across_chunks():
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    chunks = ["<tool_call>get_weather>", '{"city": ', '"北京"}']

    normal, names, params = "", "", ""
    for chunk in chunks:
        res = detector.parse_streaming_increment(chunk, tools)
        normal += res.normal_text or ""
        names, params = _collect(res, names, params)

    assert normal == ""
    assert names == "get_weather"
    assert json.loads(params) == {"city": "北京"}


def test_streaming_tag_format_split_across_chunks():
    detector = Glm53FlashDetector()
    tools = make_tools_weather()
    chunks = [
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>",
        "<arg_value>北京</arg_value>",
        "</tool_call>",
    ]

    normal, names, params = "", "", ""
    for chunk in chunks:
        res = detector.parse_streaming_increment(chunk, tools)
        normal += res.normal_text or ""
        names, params = _collect(res, names, params)

    assert normal == ""
    assert names == "get_weather"
    assert json.loads(params) == {"city": "北京"}


def test_streaming_buffers_partial_marker():
    """A chunk that ends mid-marker must be held back, not emitted as text."""
    detector = Glm53FlashDetector()
    tools = make_tools_weather()

    first = detector.parse_streaming_increment("你好", tools)
    assert first.normal_text == "你好"
    assert first.calls == []

    second = detector.parse_streaming_increment("<tool_c", tools)
    assert second.normal_text == ""
    assert second.calls == []

    third = detector.parse_streaming_increment(
        'all>get_weather>{"city": "北京"}', tools
    )
    assert third.normal_text == ""
    names, params = _collect(third, "", "")
    assert names == "get_weather"
    assert json.loads(params) == {"city": "北京"}


def test_streaming_preserves_text_before_call():
    detector = Glm53FlashDetector()
    res = detector.parse_streaming_increment(
        '前缀<tool_call>get_weather>{"city": "北京"}', make_tools_weather()
    )
    assert res.normal_text == "前缀"


# ---------------------------------------------------------------------------
# Auto-detection routing (--tool-call-parser auto)
# ---------------------------------------------------------------------------

# Decisive lines excerpted from zai-org/GLM-5.3-Flash chat_template.jinja
# (lines 1, 3, 162-163). Note `~` (not `+`) in the tool-call line: main's
# `_is_glm47` regex accepts both. `_is_glm53` additionally requires the
# `<arg_key>` / `<arg_value>` markers to be present in the template.
GLM53_FLASH_TEMPLATE = (
    "[gMASK]<sop>\n"
    "{%- if effective_reasoning_effort is not none -%}"
    "<|system|>Reasoning Effort: "
    "{{ effective_reasoning_effort | capitalize }}{%- endif -%}\n"
    "{% if m.tool_calls %}\n"
    "{% for tc in m.tool_calls %}\n"
    "{{- '<tool_call>' ~ tc.name -}}\n"
    "{% set _args = tc.arguments %}{% for k, v in _args.items() %}"
    "<arg_key>{{ k }}</arg_key><arg_value>{{ v }}</arg_value>"
    "{% endfor %}</tool_call>{% endfor %}\n"
    "{% endif %}\n"
)

GLM53_FLASH_VOCAB = [
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<|user|>",
    "<|system|>",
    "<|endoftext|>",
]


class _FakeTokenizer:
    """Minimal stand-in: template detection only needs get_vocab()."""

    def get_vocab(self):
        return {token: i for i, token in enumerate(GLM53_FLASH_VOCAB)}


def test_glm53_rule_precedes_glm47():
    """match_rules returns the first hit, so glm53 must be listed before glm47."""
    names = [rule.name for rule in TOOL_CALL_PARSER_RULES]
    assert "glm53" in names
    assert names.index("glm53") < names.index("glm47")


def test_auto_detects_glm53_from_glm53_flash_template():
    parser = detect_tool_call_parser(GLM53_FLASH_TEMPLATE, _FakeTokenizer())
    assert parser == "glm53"


def test_auto_does_not_route_glm45_template_to_glm53():
    """A GLM-4.5-style template must not be hijacked by the glm53 rule."""
    template = "[gMASK]<sop>{%- if enable_thinking -%}<|system|>thinking{%- endif -%}"
    config = ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)
    assert detect_tool_call_parser(template, _FakeTokenizer(), config) != "glm53"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
