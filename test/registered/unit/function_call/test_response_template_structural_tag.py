"""Tests for tool call structural tags derived from response templates."""

import copy
import json
import unittest

from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.function_call.response_template_structural_tag import (
    supports_tool_call_structural_tag,
    tool_call_structural_tag,
)
from sglang.srt.parser.chat_parsing.response_templates import load_response_template
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TEMPLATE = {
    "start_anchor": "<assistant>",
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "content": {"content": "text"},
        "tool_calls": {
            "open_pattern": r'<call name="(?P<name>[\w.-]+)">',
            "close": "</call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)</arg>'
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}
WEATHER = {
    "type": "function",
    "function": {
        "name": "get.weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}
PING = {"type": "function", "function": {"name": "ping"}}
CALL = '<call name="get.weather"><arg name="city">Paris</arg></call>'


ANSWER_TEMPLATE = copy.deepcopy(TEMPLATE)
ANSWER_TEMPLATE["fields"]["content"] = {
    "open": "<answer>",
    "close": "</answer>",
    "content": "text",
}


def _build(
    template=TEMPLATE,
    tools=(WEATHER, PING),
    *,
    parallel=True,
    thinking=True,
    answer=False,
):
    loaded = load_response_template(template)
    return tool_call_structural_tag(
        loaded.fields["tool_calls"],
        list(tools),
        parallel_tool_calls=parallel,
        thinking=loaded.fields["thinking"] if thinking else None,
        answer=loaded.fields["content"] if answer else None,
    )


def _accepts(tag, text):
    return _is_grammar_accept_string(Grammar.from_structural_tag(json.dumps(tag)), text)


class TestToolCallStructuralTag(unittest.TestCase):
    def test_forced_calls_follow_template_format(self):
        cases = [
            ("<think>plan</think>" + CALL, True),
            (CALL, True),
            (CALL[: -len("</call>")], True),
            (
                '<call name="get.weather"><arg name="city">Paris</arg>\n'
                '<arg name="days">3</arg></call>',
                True,
            ),
            (
                '<call name="get.weather"><arg name="city">Paris</arg>'
                '<arg name="days">soon</arg></call>',
                False,
            ),
            ('<call name="get.weather"><arg name="days">3</arg></call>', False),
            (CALL + '<call name="ping"><arg name="any">x</arg></call>', True),
            ('<call name="other"></call>', False),
            ("<think>plan</think>The answer.", False),
        ]
        tag = _build()
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIs(_accepts(tag, text), expected)

    def test_single_call_when_parallel_calls_are_disabled(self):
        tag = _build(parallel=False)

        self.assertTrue(_accepts(tag, CALL))
        self.assertFalse(_accepts(tag, CALL + CALL))

    def test_thinking_is_rejected_when_not_requested(self):
        self.assertFalse(_accepts(_build(thinking=False), "<think>plan</think>" + CALL))

    def test_answer_field_lets_the_model_reply_instead(self):
        cases = [
            ("<think>plan</think><answer>Sunny.", True),
            ("<answer>Sunny.</answer>", True),
            ("<think>plan</think>" + CALL, True),
            ("<answer>Sunny.</answer>" + CALL, False),
            ('<call name="get.weather"><arg name="days">soon</arg></call>', False),
        ]
        tag = _build(ANSWER_TEMPLATE, answer=True)
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIs(_accepts(tag, text), expected)

    def test_answer_without_literal_opener_returns_none(self):
        self.assertIsNone(_build(answer=True))

    def test_json_arguments_follow_schema(self):
        template = copy.deepcopy(TEMPLATE)
        template["fields"]["tool_calls"].update(content="json", content_args={})
        tag = _build(template, tools=[WEATHER])

        self.assertTrue(
            _accepts(tag, '<call name="get.weather">{"city": "Paris"}</call>')
        )
        self.assertFalse(_accepts(tag, '<call name="get.weather">{"days": 3}</call>'))

    def test_underivable_tool_formats_return_none(self):
        changes = {
            "json_dialect": lambda field: field.update(
                content="json", content_args={"unquoted_keys": True}
            ),
            "fixed_name": lambda field: field["transform"]["function"].update(
                name="fixed"
            ),
            "kv_lines": lambda field: field.update(content="kv-lines", content_args={}),
        }
        self.assertTrue(
            supports_tool_call_structural_tag(
                load_response_template(TEMPLATE).fields["tool_calls"]
            )
        )
        for name, change in changes.items():
            with self.subTest(name):
                template = copy.deepcopy(TEMPLATE)
                change(template["fields"]["tool_calls"])

                self.assertIsNone(_build(template))
                self.assertFalse(
                    supports_tool_call_structural_tag(
                        load_response_template(template).fields["tool_calls"]
                    )
                )

    def test_non_literal_thinking_delimiters_return_none(self):
        template = copy.deepcopy(TEMPLATE)
        del template["fields"]["thinking"]["open"]
        template["fields"]["thinking"]["open_pattern"] = r"<think(?: mode=\w+)?>"

        self.assertIsNone(_build(template))
        self.assertIsNotNone(_build(template, thinking=False))

    def test_argument_values_follow_how_the_parser_reads_them(self):
        level = {"type": "object", "properties": {"level": {"$ref": "#/$defs/level"}}}
        cases = [
            ({"type": "integer", "nullable": True}, "null", True),
            ({"type": "integer", "nullable": True}, "soon", False),
            ({"type": ["integer", "null"]}, "null", True),
            ({"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, "true", True),
            ({"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, "soon", False),
            ({"type": ["string", "integer"]}, "soon", True),
            ({"enum": ["celsius", "fahrenheit"]}, "celsius", True),
            ({"enum": ["celsius", "fahrenheit"]}, "kelvin", False),
            ({"type": "string", "const": "on"}, "off", False),
            (level, '{"level": 2}', True),
            (level, '{"level": "high"}', False),
        ]
        for schema, value, expected in cases:
            with self.subTest(schema=schema, value=value):
                parameters = {
                    "type": "object",
                    "properties": {"v": schema},
                    "required": ["v"],
                    "$defs": {"level": {"type": "integer"}},
                }
                tag = _build(tools=[_tool(parameters)])
                self.assertIs(_accepts(tag, _call(("v", value))), expected)

    def test_extra_arguments_need_additional_properties(self):
        parameters = {"type": "object", "properties": {"city": {"type": "string"}}}
        extra = _call(("city", "Paris"), ("zip", "75001"))

        self.assertFalse(_accepts(_build(tools=[_tool(parameters)]), extra))
        allowed = {**parameters, "additionalProperties": True}
        self.assertTrue(_accepts(_build(tools=[_tool(allowed)]), extra))

    def test_required_argument_without_schema_is_still_required(self):
        parameters = {"type": "object", "properties": {}, "required": ["a"]}
        tag = _build(tools=[_tool(parameters)])

        self.assertTrue(_accepts(tag, _call(("a", "x"))))
        self.assertFalse(_accepts(tag, _call()))

    def test_unencodable_names_are_rejected(self):
        cases = {
            "tool name": _tool({}, name="get weather"),
            "parameter name": _tool({"properties": {'a"b': {"type": "string"}}}),
        }
        for match, tool in cases.items():
            with self.subTest(match), self.assertRaisesRegex(ValueError, match):
                _build(tools=[tool])

    def test_support_does_not_depend_on_encodable_probe_name(self):
        template = copy.deepcopy(TEMPLATE)
        template["fields"]["tool_calls"][
            "open_pattern"
        ] = r'<call name="(?P<name>\w+\.\w+)">'
        field = load_response_template(template).fields["tool_calls"]

        self.assertTrue(supports_tool_call_structural_tag(field))


def _tool(parameters, name="f"):
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def _call(*arguments):
    body = "".join(f'<arg name="{key}">{value}</arg>' for key, value in arguments)
    return f'<call name="f">{body}</call>'


if __name__ == "__main__":
    unittest.main()
