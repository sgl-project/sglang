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
    thinking=True,
    answer=False,
    sorted_arguments=False,
):
    loaded = load_response_template(template)
    return tool_call_structural_tag(
        loaded.fields["tool_calls"],
        list(tools),
        parallel_tool_calls=True,
        thinking=loaded.fields["thinking"] if thinking else None,
        answer=loaded.fields["content"] if answer else None,
        sorted_arguments=sorted_arguments,
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

    def test_without_tools_only_the_answer_is_derived(self):
        cases = [
            ("<think>plan</think><answer>Sunny.</answer>", True),
            ("<answer>Sunny.", True),
            ("<think>plan</think>" + CALL, False),
            ("Sunny.", False),
        ]
        tag = _build(ANSWER_TEMPLATE, tools=(), answer=True)
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIs(_accepts(tag, text), expected)
        self.assertIsNone(_build(ANSWER_TEMPLATE, tools=()))

    def test_required_reasoning_opens_the_message_with_nonempty_reasoning(self):
        loaded = load_response_template(ANSWER_TEMPLATE)
        tag = tool_call_structural_tag(
            loaded.fields["tool_calls"],
            [WEATHER],
            parallel_tool_calls=True,
            thinking=loaded.fields["thinking"],
            answer=loaded.fields["content"],
            message_start=["<assistant>"],
            grammar_requires_reasoning=True,
        )
        cases = [
            ("<answer>Sunny.", False),
            ("<think></think><answer>Sunny.", False),
            ("<think> \n</think><answer>Sunny.", True),
            ("<think>\nplan</think><answer>Sunny.", True),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIs(_accepts(tag, text), expected)

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
        spaced["fields"]["tool_calls"]["close"] = ["\n", "</call>"]
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
        grammar for a strict tool."""
        # A match could write the start of the value's end.
        pattern = {"type": "string", "pattern": "^[a-z<]+$"}
        remote = {"$ref": "https://example.com/level.json"}
        cases = [
            ({"enum": [1, 2]}, True, True),
            ({"enum": [1, "two"]}, True, False),
            ({"$ref": "#/$defs/level"}, False, True),
            (remote, True, False),
            (remote, False, True),
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

    def test_names_and_escaped_json_values_may_hold_closers(self):
        closers = copy.deepcopy(TEMPLATE)
        closers["fields"]["tool_calls"]["close"] = ["</call>", "STOP"]
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

    def test_only_strict_tools_need_a_schema_shape_the_parser_types(self):
        """Metadata such as a generator's `$schema` must not matter, and a tool
        that takes any arguments must not need its schema understood."""
        parameters = WEATHER["function"]["parameters"]
        dialect = {"$schema": "http://json-schema.org/draft-07/schema#", **parameters}
        conditional = {**parameters, "if": {"required": ["days"]}}

        tag = _build(tools=[_tool(dialect)])
        self.assertTrue(_accepts(tag, _call(("city", "Paris"))))
        self.assertFalse(_accepts(tag, _call(("days", "soon"))))
        self.assertIsNone(_build(tools=[_tool(conditional)]))
        loose = _build(tools=[_tool(conditional, strict=False)])
        self.assertTrue(_accepts(loose, _call(("days", "soon"))))

    def test_strict_tool_without_parameters_takes_none(self):
        self.assertTrue(_accepts(_build(tools=[_tool(None)]), _call()))
        self.assertFalse(_accepts(_build(tools=[_tool(None)]), _call(("a", "x"))))

    def test_strict_arguments_follow_declared_or_sorted_order(self):
        parameters = {
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path"],
        }
        # Calls, and whether they are accepted without and with sorted arguments.
        cases = [
            (_call(("path", "a"), ("content", "b")), True, True),
            (_call(("content", "b"), ("path", "a")), False, True),
            (_call(("path", "a")), True, True),
            (_call(("content", "b")), False, False),
            (_call(("path", "a"), ("path", "c")), False, False),
            (_call(("path", "a"), ("mode", "c")), False, False),
        ]
        tags = [
            _build(tools=[_tool(parameters)], sorted_arguments=sorted_arguments)
            for sorted_arguments in (False, True)
        ]
        for text, *expected in cases:
            with self.subTest(text=text):
                self.assertEqual([_accepts(tag, text) for tag in tags], expected)

    def test_values_end_where_a_lookahead_end_does(self):
        template = copy.deepcopy(TEMPLATE)
        content_args = template["fields"]["tool_calls"]["content_args"]
        content_args["tag_pattern"] = (
            r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)\n(?:</arg>|(?=<arg name="))'
        )
        # Bodies, and whether a loose and a strict tool accept them.
        cases = [
            ('<arg name="city">Paris\n</arg>\n<arg name="days">3\n</arg>', True, True),
            ('<arg name="city">Paris\n<arg name="days">3\n</arg>', True, True),
            ('<arg name="days">3\n<arg name="city">Paris\n</arg>', True, False),
            ('<arg name="city">Paris\n<arg name="city">Lyon\n</arg>', True, False),
            ('<arg name="city">Paris\n<arg name="hours">3\n</arg>', True, False),
            ('<arg name="city">Paris\n<arg name="', False, False),
            ('<arg name="city">Paris\n<arg name="days\n</arg>', False, False),
            ('<arg name="city">a <arg name="b"> c\n</arg>', True, True),
        ]
        parameters = WEATHER["function"]["parameters"]
        tags = [
            _build(template, tools=[_tool(parameters, strict=strict)])
            for strict in (False, True)
        ]
        for body, *expected in cases:
            with self.subTest(body=body):
                call = f'<call name="f">{body}</call>'
                self.assertEqual([_accepts(tag, call) for tag in tags], expected)

    def test_last_parameter_follows_every_required_one(self):
        template = copy.deepcopy(TEMPLATE)
        template["fields"]["tool_calls"]["content_args"]["tag_pattern"] = (
            r'<arg name="(?P<key>[^"]+)">(?P<value>.*?)\n(?:</arg>|(?=<arg name="))'
        )
        parameters = {
            "properties": {key: {"type": "string"} for key in "abcde"},
            "required": ["b"],
        }
        tag = _build(template, tools=[_tool(parameters)])

        def call(keys: str) -> str:
            body = "".join(f'<arg name="{k}">x\n</arg>' for k in keys)
            return f'<call name="f">{body}</call>'

        # Parameters in call order, and whether the call is accepted.
        cases = [
            ("b", True),
            ("ab", True),
            ("bc", True),
            ("bd", True),
            ("be", True),
            ("abde", True),
            ("bcde", True),
            ("", False),
            ("a", False),
            ("acd", False),
            ("bdc", False),
            ("bcc", False),
        ]
        for keys, expected in cases:
            with self.subTest(keys=keys):
                self.assertIs(_accepts(tag, call(keys)), expected)

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
