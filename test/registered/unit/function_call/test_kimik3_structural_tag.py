import json
import sys

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    THINK_CLOSE,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
    get_kimik3_structural_tag,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_CLOSE_TOKEN = "<|close|>"
_CLOSE_TOKEN_ID = 256
_TOKENIZER_INFO = xgr.TokenizerInfo(
    [bytes([token_id]) for token_id in range(256)] + [_CLOSE_TOKEN.encode()]
)
_TOKEN_COMPILER = xgr.GrammarCompiler(_TOKENIZER_INFO, cache_enabled=True)


def _tool(name="weather", strict=True):
    return Tool(
        type="function",
        function=Function(
            name=name,
            strict=strict,
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "pattern": "[A-Z][A-Za-z ]+",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "unit": {
                        "type": ["string", "null"],
                        "enum": ["celsius", "fahrenheit", None],
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}},
                        "required": ["source"],
                        "additionalProperties": False,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["city", "days"],
                "additionalProperties": False,
            },
        ),
    )


def _argument(key, argument_type, value):
    return (
        f'<|open|>argument key="{key}" type="{argument_type}"<|sep|>'
        f"{value}{ARGUMENT_CLOSE}"
    )


def _call(name, index, *arguments):
    return (
        f'<|open|>call tool="{name}" index="{index}"<|sep|>'
        + "".join(arguments)
        + CALL_CLOSE
    )


def _tools_section(*calls):
    return TOOLS_OPEN + "".join(calls) + TOOLS_CLOSE


def _grammar(tools, tool_choice="auto", thinking_mode=False, parallel_tool_calls=True):
    structural_tag = get_kimik3_structural_tag(
        tools,
        tool_choice=tool_choice,
        thinking_mode=thinking_mode,
        parallel_tool_calls=parallel_tool_calls,
    )
    return xgr.Grammar.from_structural_tag(structural_tag)


def _accepts(grammar, value):
    return _is_grammar_accept_string(grammar, value)


def _encode_with_close_token(value):
    token_ids = []
    start = 0
    while (index := value.find(_CLOSE_TOKEN, start)) != -1:
        token_ids.extend(value[start:index].encode())
        token_ids.append(_CLOSE_TOKEN_ID)
        start = index + len(_CLOSE_TOKEN)
    token_ids.extend(value[start:].encode())
    return token_ids


def _token_accepts(structural_tag, value):
    compiled = _TOKEN_COMPILER.compile_structural_tag(structural_tag)
    matcher = xgr.GrammarMatcher(compiled)
    for token_id in _encode_with_close_token(value):
        if not matcher.accept_token(token_id):
            return False
    return matcher.is_completed()


def _valid_weather_call(index=1):
    return _call(
        "weather",
        index,
        _argument("city", "string", "San Francisco"),
        _argument("days", "number", "3"),
        _argument("unit", "string", "celsius"),
        _argument("metadata", "object", '{"source":"forecast"}'),
        _argument("tags", "array", '["coastal","windy"]'),
    )


def test_strict_schema_accepts_native_xtml_values():
    grammar = _grammar([_tool()], tool_choice="required")

    assert _accepts(grammar, _tools_section(_valid_weather_call()))
    assert _accepts(
        grammar,
        _tools_section(
            _call(
                "weather",
                1,
                _argument("city", "string", "Paris"),
                _argument("days", "number", "1"),
            )
        ),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (_argument("city", "string", "paris"), _argument("days", "number", "3")),
        (
            _argument("city", "string", "Paris"),
            _argument("days", "number", "11"),
        ),
        (
            _argument("city", "number", "3"),
            _argument("days", "number", "3"),
        ),
        (
            _argument("city", "string", "Paris"),
            _argument("days", "number", "3"),
            _argument("unit", "string", "kelvin"),
        ),
        (
            _argument("city", "string", "Paris"),
            _argument("days", "number", "3"),
            _argument("tags", "array", "[coastal]"),
        ),
        (
            _argument("city", "string", "Paris"),
            _argument("days", "number", "3"),
            _argument("unknown", "string", "value"),
        ),
    ],
)
def test_strict_schema_rejects_invalid_parameters(arguments):
    grammar = _grammar([_tool()], tool_choice="required")

    assert not _accepts(grammar, _tools_section(_call("weather", 1, *arguments)))


def test_required_allows_response_prefix_but_requires_tools():
    grammar = _grammar([_tool()], tool_choice="required")
    response = "<|open|>response<|sep|>Checking." "<|close|>response<|sep|>"

    assert _accepts(grammar, response + _tools_section(_valid_weather_call()))
    assert not _accepts(grammar, response)


def test_auto_allows_plain_response_or_multiple_tool_calls():
    grammar = _grammar([_tool(), _tool("forecast")])
    plain = (
        "<|open|>response<|sep|>No tool needed."
        "<|close|>response<|sep|><|close|>message<|sep|>"
    )
    calls = _tools_section(
        _valid_weather_call(),
        _call(
            "forecast",
            2,
            _argument("city", "string", "London"),
            _argument("days", "number", "2"),
        ),
    )

    assert _accepts(grammar, plain)
    assert _accepts(grammar, calls)


def test_named_tool_choice_forces_only_the_selected_tool():
    grammar = _grammar(
        [_tool(), _tool("forecast")],
        tool_choice=ToolChoice(function=ToolChoiceFuncName(name="forecast")),
    )
    forecast_call = _call(
        "forecast",
        1,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )

    assert _accepts(grammar, _tools_section(forecast_call))
    assert not _accepts(grammar, _tools_section(_valid_weather_call()))


def test_function_call_parser_uses_native_tag_for_named_tool_choice():
    tool_choice = ToolChoice(function=ToolChoiceFuncName(name="forecast"))
    constraint = FunctionCallParser(
        [_tool(), _tool("forecast")], "kimi_k3"
    ).get_structure_constraint(tool_choice)
    assert constraint is not None
    grammar = xgr.Grammar.from_structural_tag(constraint[1])
    forecast_call = _call(
        "forecast",
        1,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )

    assert _accepts(grammar, _tools_section(forecast_call))
    assert not _accepts(grammar, _tools_section(_valid_weather_call()))


def test_non_strict_tool_keeps_xtml_structure_and_loose_parameters():
    grammar = _grammar([_tool(strict=False)], tool_choice="required")
    call = _call(
        "weather",
        1,
        _argument("custom", "array", '["x",1]'),
        _argument("other", "string", "raw text"),
    )

    assert _accepts(grammar, _tools_section(call))
    assert not _accepts(grammar, _tools_section("unstructured"))


def test_strict_schema_supports_refs_and_mixed_unions():
    tool = Tool(
        type="function",
        function=Function(
            name="convert",
            strict=True,
            parameters={
                "type": "object",
                "$defs": {
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "safe"],
                    }
                },
                "properties": {
                    "mode": {"$ref": "#/$defs/mode"},
                    "value": {
                        "anyOf": [
                            {"type": "string", "enum": ["auto"]},
                            {
                                "type": "integer",
                                "minimum": 2,
                                "maximum": 3,
                            },
                        ]
                    },
                },
                "required": ["mode", "value"],
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(
            _call(
                "convert",
                1,
                _argument("mode", "string", "fast"),
                _argument("value", "number", "2"),
            )
        ),
    )
    assert not _accepts(
        grammar,
        _tools_section(
            _call(
                "convert",
                1,
                _argument("mode", "string", "unsafe"),
                _argument("value", "number", "1"),
            )
        ),
    )


def test_strict_schema_handles_number_enums_and_all_of_integer_constraints():
    tool = Tool(
        type="function",
        function=Function(
            name="score",
            strict=True,
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "enum": [1, 1.5],
                    },
                    "count": {
                        "allOf": [
                            {"type": "number"},
                            {"type": "integer", "minimum": 1},
                        ]
                    },
                },
                "required": ["value", "count"],
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(
            _call(
                "score",
                1,
                _argument("value", "number", "1"),
                _argument("count", "number", "2"),
            )
        ),
    )
    assert not _accepts(
        grammar,
        _tools_section(
            _call(
                "score",
                1,
                _argument("value", "number", "2"),
                _argument("count", "number", "1.5"),
            )
        ),
    )


def test_strict_schema_preserves_additional_properties_default():
    tool = Tool(
        type="function",
        function=Function(
            name="annotate",
            strict=True,
            parameters={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                },
                "required": ["label"],
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(
            _call(
                "annotate",
                1,
                _argument("label", "string", "sample"),
                _argument("confidence", "number", "0.9"),
            )
        ),
    )


def test_dynamic_argument_key_compiles_without_xgrammar_unicode_warning(capfd):
    tool = Tool(
        type="function",
        function=Function(
            name="annotate",
            strict=True,
            parameters={
                "type": "object",
                "properties": {},
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(_call("annotate", 1, _argument("置信度", "number", "0.9"))),
    )
    assert "Negative Character class" not in capfd.readouterr().err


def test_strict_empty_object_accepts_no_arguments_only():
    tool = Tool(
        type="function",
        function=Function(
            name="ping",
            strict=True,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(grammar, _tools_section(_call("ping", 1)))
    assert not _accepts(
        grammar,
        _tools_section(_call("ping", 1, _argument("unexpected", "string", "value"))),
    )


def test_tool_strict_level_controls_native_tag_parameter_schema():
    invalid_call = _tools_section(
        _call(
            "weather",
            264,
            _argument("city", "string", "paris"),
            _argument("days", "number", "99"),
        )
    )
    empty_call = _tools_section(_call("weather", 264))

    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.OFF.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto")
        assert constraint is not None
        assert _token_accepts(constraint[1], invalid_call)
        assert not _token_accepts(constraint[1], empty_call)

    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.FUNCTION.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto")
        assert constraint is not None
        grammar = xgr.Grammar.from_structural_tag(constraint[1])
        assert _accepts(grammar, invalid_call)
        assert _accepts(grammar, empty_call)

    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto")
        assert constraint is not None
        assert not _accepts(
            xgr.Grammar.from_structural_tag(constraint[1]), invalid_call
        )


def test_auto_hook_constrains_all_calls_and_requires_nonempty_values():
    structural_tag = get_kimik3_auto_tool_call_structural_tag([_tool(strict=False)])
    assert structural_tag is not None
    first = _call(
        "weather",
        3,
        _argument("city", "string", "Paris"),
        _argument("days", "number", "3"),
    )
    second = _call(
        "weather",
        264,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )

    assert _token_accepts(structural_tag, _tools_section(first, second))
    assert not _token_accepts(
        structural_tag,
        _tools_section(first, _call("weather", 264)),
    )
    assert not _token_accepts(
        structural_tag,
        _tools_section(
            _call(
                "weather",
                3,
                _argument("city", "string", ""),
                _argument("days", "number", "3"),
            )
        ),
    )
    assert not _token_accepts(structural_tag, _tools_section(_call("weather", 3)))


def test_auto_hook_rejects_unknown_or_unclosed_calls():
    structural_tag = get_kimik3_auto_tool_call_structural_tag([_tool(strict=False)])
    assert structural_tag is not None
    call = _call(
        "weather",
        49,
        _argument("city", "string", "Paris"),
        _argument("days", "number", "3"),
    )
    unknown = _call(
        "forecast",
        49,
        _argument("city", "string", "Paris"),
        _argument("days", "number", "3"),
    )

    assert not _token_accepts(structural_tag, _tools_section(unknown))
    assert not _token_accepts(
        structural_tag, TOOLS_OPEN + call.removesuffix(CALL_CLOSE)
    )
    assert not _token_accepts(structural_tag, TOOLS_OPEN + call)


def test_auto_hook_does_not_swallow_parser_visible_closes():
    structural_tag = get_kimik3_auto_tool_call_structural_tag([_tool(strict=False)])
    assert structural_tag is not None
    output = (
        TOOLS_OPEN
        + '<|open|>call tool="weather" index="23"<|sep|>'
        + _argument("city", "string", "Paris")
        + '<|open|>argument key="days" type="number"<|sep|>'
        + ARGUMENT_CLOSE
        + CALL_CLOSE
        + "3"
        + ARGUMENT_CLOSE
        + CALL_CLOSE
        + TOOLS_CLOSE
    )

    parsed = KimiK3Detector().detect_and_parse(output, [_tool(strict=False)])

    assert json.loads(parsed.calls[0].parameters) == {"city": "Paris", "days": ""}
    assert not _token_accepts(structural_tag, output)


@pytest.mark.parametrize(
    "tool_strict_level",
    [
        ToolStrictLevel.OFF,
        ToolStrictLevel.FUNCTION,
        ToolStrictLevel.PARAMETER,
    ],
)
def test_parallel_tool_calls_false_rejects_second_call(tool_strict_level):
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(tool_strict_level.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto", parallel_tool_calls=False)
    assert constraint is not None
    first = _call(
        "weather",
        165,
        _argument("city", "string", "Paris"),
        _argument("days", "number", "3"),
    )
    second = _call(
        "weather",
        166,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )

    assert _token_accepts(constraint[1], _tools_section(first))
    assert not _token_accepts(constraint[1], _tools_section(first, second))
    assert not _token_accepts(
        constraint[1], _tools_section(first) + _tools_section(second)
    )


def test_parallel_tool_calls_true_constrains_every_call():
    grammar = _grammar(
        [_tool(strict=False)],
        parallel_tool_calls=True,
    )
    first = _call(
        "weather",
        7,
        _argument("city", "string", "Paris"),
    )
    second = _call(
        "weather",
        19,
        _argument("other", "array", '["loose"]'),
    )

    assert _accepts(grammar, _tools_section(first, second))


def test_parameter_level_constrains_every_parallel_call():
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k3"
        ).get_structure_constraint("auto")
    assert constraint is not None
    grammar = xgr.Grammar.from_structural_tag(constraint[1])
    first = _valid_weather_call(index=3)
    valid_second = _call(
        "weather",
        49,
        _argument("city", "string", "London"),
        _argument("days", "number", "2"),
    )
    invalid_second = _call(
        "weather",
        49,
        _argument("city", "string", "london"),
        _argument("days", "number", "99"),
    )

    assert _accepts(grammar, _tools_section(first, valid_second))
    assert not _accepts(grammar, _tools_section(first, invalid_second))


def test_strict_tool_without_parameters_compiles_to_empty_arguments():
    """SGLANG_TOOL_STRICT_LEVEL=2 marks every tool strict, including tools
    that declare no parameters; the grammar build must not fail for them."""
    tool = Tool(type="function", function=Function(name="ping", strict=True))
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(grammar, _tools_section(_call("ping", 1)))
    assert not _accepts(
        grammar,
        _tools_section(_call("ping", 1, _argument("x", "string", "y"))),
    )


def test_parameter_level_keeps_constraint_with_no_parameter_tool():
    """A single no-parameter tool must not poison the whole request: a build
    error here was swallowed and required fell back to a JSON-only grammar
    the K3 parser cannot read."""
    tools = [
        _tool(strict=False),
        Tool(type="function", function=Function(name="ping")),
    ]
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
        constraint = FunctionCallParser(tools, "kimi_k3").get_structure_constraint(
            "required"
        )

    assert constraint is not None
    assert constraint[0] == "structural_tag"


def test_all_of_number_branches_do_not_narrow_to_integer():
    """allOf with only number branches was intersected down to integer,
    silently dropping non-integer enum values."""
    tool = Tool(
        type="function",
        function=Function(
            name="scale",
            strict=True,
            parameters={
                "type": "object",
                "properties": {
                    "factor": {"allOf": [{"type": "number"}], "enum": [1.5, 2]},
                },
                "required": ["factor"],
                "additionalProperties": False,
            },
        ),
    )
    grammar = _grammar([tool], tool_choice="required")

    assert _accepts(
        grammar,
        _tools_section(_call("scale", 1, _argument("factor", "number", "1.5"))),
    )
    assert not _accepts(
        grammar,
        _tools_section(_call("scale", 1, _argument("factor", "number", "3"))),
    )


def test_auto_hook_serializes_into_sampling_parameters():
    constraint = FunctionCallParser(
        [_tool(strict=False)], "kimi_k3"
    ).get_structure_constraint("auto")
    assert constraint is not None
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Weather?"}],
        max_completion_tokens=16,
    )

    sampling_params = request.to_sampling_params(
        stop=[],
        model_generation_config={},
        tool_call_constraint=constraint,
    )

    serialized = json.loads(sampling_params["structural_tag"])
    assert serialized["type"] == "structural_tag"
    assert serialized["format"]["type"] == "triggered_tags"


def test_auto_hook_forces_one_typed_property_when_none_are_required():
    tool = Tool(
        type="function",
        function=Function(
            name="search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        ),
    )
    structural_tag = get_kimik3_auto_tool_call_structural_tag([tool])
    assert structural_tag is not None

    assert _token_accepts(
        structural_tag,
        _tools_section(_call("search", 1, _argument("limit", "number", "3"))),
    )
    assert not _token_accepts(structural_tag, _tools_section(_call("search", 1)))


def test_auto_hook_keeps_structure_for_ambiguous_required_argument_type():
    tool = Tool(
        type="function",
        function=Function(
            name="lookup",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": ["string", "integer"]},
                },
                "required": ["key"],
            },
        ),
    )

    structural_tag = get_kimik3_auto_tool_call_structural_tag([tool])
    assert structural_tag is not None
    grammar = xgr.Grammar.from_structural_tag(structural_tag)

    assert _accepts(
        grammar,
        _tools_section(_call("lookup", 9254, _argument("key", "number", "3"))),
    )
    assert not _accepts(
        grammar,
        TOOLS_OPEN + _call("lookup", 9254, _argument("key", "number", "3")),
    )


def test_parameter_level_applies_to_other_model_native_tags():
    with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
        constraint = FunctionCallParser(
            [_tool(strict=False)], "kimi_k2"
        ).get_structure_constraint("auto")

    assert constraint is not None
    serialized = constraint[1].model_dump_json()
    assert '"properties"' in serialized
    assert '"city"' in serialized


def test_reasoning_prefix_is_owned_by_exactly_one_layer():
    tool = _tool()
    wrapped_by_xgrammar = _grammar([tool], tool_choice="required", thinking_mode=True)
    post_reasoning_only = _grammar([tool], tool_choice="required", thinking_mode=False)
    output = "reasoning" + THINK_CLOSE + _tools_section(_valid_weather_call())

    assert _accepts(wrapped_by_xgrammar, output)
    assert not _accepts(post_reasoning_only, output)
    assert _accepts(post_reasoning_only, _tools_section(_valid_weather_call()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
