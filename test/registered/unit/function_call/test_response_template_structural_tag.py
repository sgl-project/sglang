"""Tests for tool call structural tags derived from response templates."""

import copy
import json
import unittest

from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.function_call.response_template_structural_tag import (
    json_answer_structural_tag,
    supports_tool_call_structural_tag,
    tool_call_structural_tag,
)
from sglang.srt.parser.chat_parsing.response_templates import load_response_template
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

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
        "strict": True,
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


class TestToolCallStructuralTag(CustomTestCase):
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
        """Among them, formats in which a value or the arguments and the text
        next to them can spell a closer, where the parser would end the call."""
        end_x = {"tag_pattern": r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)X'}
        changes = {
            "json_dialect": lambda field: field.update(
                content="json", content_args={"unquoted_keys": True}
            ),
            "fixed_name": lambda field: field["transform"]["function"].update(
                name="fixed"
            ),
            "kv_lines": lambda field: field.update(content="kv-lines", content_args={}),
            "text value end": lambda field: field.update(
                close=["}X", "</call>"], content_args=end_x
            ),
            "json value end": lambda field: field.update(
                close=["}X", "</call>"],
                content_args={**end_x, "value_parser": {"name": "json"}},
            ),
            "number end": lambda field: field.update(
                close=["7X", "</call>"], content_args=end_x
            ),
            "value start": lambda field: field.update(close=[">X", "</call>"]),
            "parameter end": lambda field: field.update(close=["></call>", "</call>"]),
            "next parameter": lambda field: field.update(close=["><a", "</call>"]),
            "json arguments end": lambda field: field.update(
                close=["}</call>", "</call>"], content="json", content_args={}
            ),
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

    def test_only_call_text_that_can_spell_a_closer_is_not_derived(self):
        """Call text that could spell a closer before its own must not be derived
        natively."""
        spelled = copy.deepcopy(TEMPLATE)
        spelled["fields"]["tool_calls"]["close"] = ['b">x', "</call>"]
        spaced = copy.deepcopy(TEMPLATE)
        spaced["fields"]["tool_calls"]["close"] = ["\n\n", "</call>"]
        colon = copy.deepcopy(TEMPLATE)
        colon["fields"]["tool_calls"]["content_args"]["tag_pattern"] = (
            r"<arg:(?P<key>\w+)>(?P<value>.*?)</arg>"
        )
        separated = copy.deepcopy(TEMPLATE)
        separated["fields"]["tool_calls"].update(
            close=["}X", "</call>"],
            content_args={
                "tag_pattern": r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)QX'
            },
        )

        def declared(key):
            return _tool({"type": "object", "properties": {key: {"type": "string"}}})

        cases = [
            ("declared key", spelled, declared("ab"), False),
            ("whitespace", spaced, declared("v"), False),
            ("other key", spelled, declared("cd"), True),
            ("key pattern", colon, _tool(None, strict=False), True),
            ("separated end", separated, declared("v"), True),
        ]
        for name, template, tool, derivable in cases:
            with self.subTest(name):
                self.assertIs(_build(template, tools=[tool]) is not None, derivable)

        for close, derivable in [(["}</call>", "</call>"], False), ("</call>", True)]:
            with self.subTest(close=close):
                template = copy.deepcopy(ANSWER_TEMPLATE)
                template["fields"]["tool_calls"]["close"] = close
                fields = load_response_template(template).fields
                tag = json_answer_structural_tag(
                    fields["content"], {"type": "object"}, fields["tool_calls"], ["f"]
                )
                self.assertIs(tag is not None, derivable)

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
        # Next to declared keys, an extra one could repeat them unchecked.
        self.assertIsNone(
            _build(tools=[_tool({**parameters, "additionalProperties": True})])
        )
        free_form = {"type": "object", "additionalProperties": True}
        self.assertTrue(_accepts(_build(tools=[_tool(free_form)]), extra))

    def test_schemas_text_values_cannot_hold_are_not_derived(self):
        """Untyped arguments, or text a strict schema cannot hold, must not get a native
        grammar."""
        pattern = {"type": "string", "pattern": "^[a-z]+$"}
        cases = [
            ({"enum": [1, 2]}, True, True),
            ({"enum": [1, "two"]}, True, False),
            ({"$ref": "#/$defs/level"}, False, True),
            ({"$ref": "https://example.com/level.json"}, False, False),
            # Raw text cannot hold its closer, and XGrammar rejects the value.
            ({"enum": ["a</arg>b"]}, True, False),
            ({"type": "object", "properties": {"x": {"$ref": "#/x"}}}, True, False),
            (False, True, False),
            (pattern, True, False),
            (pattern, False, True),
        ]
        for schema, strict, derivable in cases:
            with self.subTest(schema=schema, strict=strict):
                parameters = {
                    "type": "object",
                    "properties": {"v": schema},
                    "$defs": {"level": {"type": "integer"}},
                }
                tag = _build(tools=[_tool(parameters, strict=strict)])
                self.assertIs(tag is not None, derivable)

    def test_json_values_and_names_may_hold_closers(self):
        closers = copy.deepcopy(TEMPLATE)
        closers["fields"]["tool_calls"]["close"] = ["</call>", "STOP"]
        loose = {"type": "integer", "allOf": [{"type": ["integer", "string"]}]}
        text = {"type": "string"}
        cases = [
            (TEMPLATE, loose, "7", True),
            (TEMPLATE, loose, '"a</call>b"', True),
            (
                closers,
                {"type": "object", "properties": {"x": text}},
                '{"x": "STOP"}',
                True,
            ),
            (closers, text, "aSTOPb", False),
        ]
        for template, schema, value, accepted in cases:
            with self.subTest(schema=schema, value=value):
                schema = {"type": "object", "properties": {"v": schema}}
                tag = _build(template, tools=[_tool(schema)])
                self.assertIs(_accepts(tag, _call(("v", value))), accepted)

        tag = _build(closers, tools=[_tool(None, strict=False)])
        self.assertTrue(_accepts(tag, _call(("aSTOPb", "x"))))
        quoted = copy.deepcopy(TEMPLATE)
        quoted["fields"]["tool_calls"]["content_args"]["tag_pattern"] = (
            r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)"X'
        )
        tag = _build(quoted, tools=[_tool({"properties": {"v": {"type": "object"}}})])
        self.assertTrue(
            _accepts(tag, '<call name="f"><arg name="v">{"x": "a\\"Xb"}"X</call>')
        )
        written = copy.deepcopy(TEMPLATE)
        written["fields"]["tool_calls"]["close"] = [',"', "</call>"]
        json_value = {"type": "object", "properties": {"v": {"type": "integer"}}}
        self.assertIsNone(_build(written, tools=[_tool(json_value)]))

    def test_aliases_stand_for_the_local_schema_they_name(self):
        """An aliased schema must not lose its target's restrictions, as a nullable enum
        would."""
        defs = {"s": {"type": "string"}, "c": {"enum": ["red", "green"]}}

        def build(schema):
            parameters = {"type": "object", "properties": {"v": schema}, "$defs": defs}
            return _build(tools=[_tool(parameters)])

        nullable = build({"$ref": "#/$defs/s", "nullable": True})
        color = build({"allOf": [{"$ref": "#/$defs/c"}], "description": "a color"})

        self.assertTrue(_accepts(nullable, _call(("v", "Paris"))))
        self.assertTrue(_accepts(color, _call(("v", "red"))))
        self.assertFalse(_accepts(color, _call(("v", "blue"))))
        self.assertIsNone(build({"$ref": "#/$defs/c", "nullable": True}))

    def test_only_strict_tools_follow_their_schema(self):
        off_schema = _call(("days", "soon"))
        parameters = WEATHER["function"]["parameters"]

        self.assertTrue(
            _accepts(_build(tools=[_tool(parameters, strict=False)]), off_schema)
        )
        self.assertFalse(_accepts(_build(tools=[_tool(parameters)]), off_schema))
        self.assertTrue(_accepts(_build(tools=[_tool(None)]), _call()))
        self.assertFalse(_accepts(_build(tools=[_tool(None)]), _call(("a", "x"))))

    def test_required_argument_without_schema_is_still_required(self):
        """A required entry that is no name leaves the tool to the generic JSON
        schema instead."""
        parameters = {"type": "object", "properties": {}, "required": ["a"]}
        tag = _build(tools=[_tool(parameters)])

        self.assertTrue(_accepts(tag, _call(("a", "x"))))
        self.assertFalse(_accepts(tag, _call()))
        self.assertIsNone(_build(tools=[_tool({**parameters, "required": ["a", 1]})]))

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
        template["fields"]["tool_calls"]["open_pattern"] = (
            r'<call name="(?P<name>\w+\.\w+)">'
        )
        field = load_response_template(template).fields["tool_calls"]

        self.assertTrue(supports_tool_call_structural_tag(field))


def _tool(parameters, name="f", strict=True):
    return {
        "type": "function",
        "function": {"name": name, "parameters": parameters, "strict": strict},
    }


def _call(*arguments):
    body = "".join(f'<arg name="{key}">{value}</arg>' for key, value in arguments)
    return f'<call name="f">{body}</call>'


if __name__ == "__main__":
    unittest.main()
