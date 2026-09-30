"""Tests for Beam response-template parsing."""

import json
import unittest

from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PREFIX = "<|start|>assistant"


def _tool(
    name: str = "get_weather",
    properties: dict | None = None,
) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={
                "type": "object",
                "properties": properties
                or {
                    "city": {"type": "string"},
                },
            },
        ),
    )


def _parameter(name: str, value: str) -> str:
    return f'<rfl:parameter name="{name}">\n{value}\n</rfl:parameter>'


def _continuation_call(
    name: str,
    body: str,
    *,
    terminator: str = "<|call|>",
    constrain: bool = True,
) -> str:
    constraint = " <|constrain|>xml" if constrain else ""
    return f" to={name}<|channel|>analysis{constraint}<|message|>{body}{terminator}"


def _full_call(
    name: str,
    body: str,
    *,
    terminator: str = "<|call|>",
    constrain: bool = True,
) -> str:
    return PREFIX + _continuation_call(
        name,
        body,
        terminator=terminator,
        constrain=constrain,
    )


class TestBeamResponseTemplate(CustomTestCase):
    def test_reasoning_and_final_content(self):
        parser = ReasoningParser(model_type="beam", prefix=PREFIX)
        generation = (
            "<|channel|>analysis<|message|>Think first.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Answer.<|return|>"
        )

        reasoning, content = parser.parse_non_stream(generation)

        self.assertEqual(reasoning, "Think first.")
        self.assertEqual(content, "Answer.")

    def test_prefilled_reasoning_continues_from_prompt(self):
        parser = ReasoningParser(
            model_type="beam",
            prefix=PREFIX + "<|channel|>analysis<|message|>Existing",
        )

        reasoning, content = parser.parse_non_stream(
            " thought.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Done.<|return|>"
        )

        self.assertEqual(reasoning, " thought.")
        self.assertEqual(content, "Done.")

    def test_reasoning_to_parallel_tool_parser_handoff(self):
        tools = [_tool(), _tool("set_alarm", {"hour": {"type": "integer"}})]
        reasoning_parser = ReasoningParser(model_type="beam", prefix=PREFIX)
        tool_parser = FunctionCallParser(
            tools=tools,
            tool_call_parser="beam",
            prefix=PREFIX,
        )
        generation = (
            "<|channel|>analysis<|message|>Use two tools.<|end|>"
            + _full_call(
                "get_weather",
                _parameter("city", "Paris"),
                terminator="<|end|>",
            )
            + _full_call("set_alarm", _parameter("hour", "7"))
        )

        reasoning, tool_wire = reasoning_parser.parse_non_stream(generation)
        content, calls = tool_parser.parse_non_stream(tool_wire)

        self.assertEqual(reasoning, "Use two tools.")
        self.assertEqual(content, "")
        self.assertEqual([call.name for call in calls], ["get_weather", "set_alarm"])
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris"})
        self.assertEqual(json.loads(calls[1].parameters), {"hour": 7})

    def test_tool_call_accepts_constraint_marker_variants(self):
        for constrain in (True, False):
            with self.subTest(constrain=constrain):
                parser = FunctionCallParser(
                    tools=[_tool()],
                    tool_call_parser="beam",
                    prefix=PREFIX,
                )
                generation = _full_call(
                    "get_weather",
                    _parameter("city", "Paris"),
                    constrain=constrain,
                )

                content, calls = parser.parse_non_stream(generation)

                self.assertEqual(content, "")
                self.assertEqual([call.name for call in calls], ["get_weather"])
                self.assertEqual(
                    json.loads(calls[0].parameters),
                    {"city": "Paris"},
                )

    def test_declared_dotted_tool_name_is_kept(self):
        parser = FunctionCallParser(
            tools=[_tool("refl.search", {"query": {"type": "string"}})],
            tool_call_parser="beam",
            prefix=PREFIX,
        )

        content, calls = parser.parse_non_stream(
            _full_call("refl.search", _parameter("query", "weather"))
        )

        self.assertEqual(content, "")
        self.assertEqual([call.name for call in calls], ["refl.search"])
        self.assertEqual(json.loads(calls[0].parameters), {"query": "weather"})

    def test_forced_tool_calls_use_template_grammar(self):
        """Forced calls after the reasoning follow the template grammar, which
        accepts only calls the parser reads back."""
        count = _tool("set_count", {"count": {"type": "integer"}})
        count.function.strict = True
        parser = FunctionCallParser(
            tools=[_tool(), count], tool_call_parser="beam", prefix=PREFIX
        )
        weather = ("get_weather", {"city": "Paris"})
        weather_call = _full_call("get_weather", _parameter("city", "Paris"))
        cases = [
            ("required", weather_call, [weather]),
            (
                "required",
                _full_call("get_weather", _parameter("city", "Paris"), constrain=False),
                [weather],
            ),
            (
                "required",
                _full_call(
                    "get_weather", _parameter("city", "Paris"), terminator="<|end|>"
                )
                + _full_call("set_count", _parameter("count", "7")),
                [weather, ("set_count", {"count": 7})],
            ),
            ("required", _full_call("set_count", _parameter("count", "x")), None),
            ("required", PREFIX + "<|channel|>final<|message|>Paris<|return|>", None),
            (
                ToolChoice(type="function", function={"name": "set_count"}),
                weather_call,
                None,
            ),
            (
                ToolChoice(type="function", function={"name": "get_weather"}),
                weather_call,
                [weather],
            ),
            (
                ToolChoice(type="function", function={"name": "get_weather"}),
                _full_call(
                    "get_weather", _parameter("city", "Paris"), terminator="<|end|>"
                )
                + weather_call,
                None,
            ),
        ]

        self.assertTrue(parser.detector.supports_structural_tag())
        for tool_choice, text, calls in cases:
            with self.subTest(tool_choice=tool_choice, text=text):
                kind, tag = parser.get_structure_constraint(tool_choice)
                grammar = Grammar.from_structural_tag(tag.model_dump_json())

                self.assertEqual(kind, "structural_tag")
                self.assertIs(_is_grammar_accept_string(grammar, text), bool(calls))
                if calls:
                    _, parsed = parser.parse_non_stream(text)
                    self.assertEqual(
                        [(call.name, json.loads(call.parameters)) for call in parsed],
                        calls,
                    )

    def test_tag_from_message_start_covers_skipped_reasoning(self):
        """A call continuing the start anchor must not be admitted unless it is read
        back."""
        count = _tool("set_count", {"count": {"type": "integer"}})
        count.function.strict = True
        parser = FunctionCallParser(
            tools=[_tool(), count], tool_call_parser="beam", prefix=PREFIX
        )
        thought = "<|channel|>analysis<|message|>plan<|end|>"
        city = _parameter("city", "Paris")
        weather = [("get_weather", {"city": "Paris"})]
        cases = [
            (_continuation_call("get_weather", city), weather),
            (thought + _full_call("get_weather", city), weather),
            (_continuation_call("set_count", _parameter("count", "x")), None),
            (thought + _full_call("set_count", _parameter("count", "x")), None),
            (
                _continuation_call("get_weather", city, terminator="<|end|>")
                + _full_call("get_weather", city),
                None,
            ),
            ("<|channel|>final<|message|>Paris<|return|>", None),
        ]

        named = ToolChoice(type="function", function={"name": "get_weather"})
        for tool_choice, parallel_tool_calls in (("required", False), (named, True)):
            tag = parser.detector.get_structural_tag(
                tools=parser.tools,
                tool_choice=tool_choice,
                thinking_mode=True,
                parallel_tool_calls=parallel_tool_calls,
                message_start=True,
            )
            grammar = Grammar.from_structural_tag(tag.model_dump_json())
            for text, calls in cases:
                with self.subTest(tool_choice=tool_choice, text=text):
                    self.assertIs(_is_grammar_accept_string(grammar, text), bool(calls))
                    if calls:
                        _, parsed = parser.parse_non_stream(text)
                        self.assertEqual(
                            [
                                (call.name, json.loads(call.parameters))
                                for call in parsed
                            ],
                            calls,
                        )

    def test_values_typed_by_enum_or_local_ref_are_constrained(self):
        """A value typed only by its enum or local `$ref` must not be read back as text."""
        either = {"anyOf": [{"$ref": "#/$defs/n"}, {"$ref": "#/$defs/loc"}]}
        tool = _tool(
            "set_level",
            {
                "n": {"enum": [1, 2]},
                "loc": {"$ref": "#/$defs/loc"},
                "either": either,
                "name": {"$ref": "#/$defs/name"},
            },
        )
        tool.function.parameters["$defs"] = {
            "loc": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "n": {"type": "integer"},
            "name": {"type": "string"},
        }
        tool.function.strict = True
        parser = FunctionCallParser(
            tools=[tool], tool_call_parser="beam", prefix=PREFIX
        )

        def call(n: str, loc: str, either: str = "1") -> str:
            arguments = {"n": n, "loc": loc, "either": either, "name": "Paris"}
            return _full_call(
                "set_level", "".join(_parameter(k, v) for k, v in arguments.items())
            )

        kind, tag = parser.get_structure_constraint("required")
        grammar = Grammar.from_structural_tag(tag.model_dump_json())

        self.assertEqual(kind, "structural_tag")
        self.assertFalse(_is_grammar_accept_string(grammar, call("9", '{"x": 1}')))
        self.assertFalse(_is_grammar_accept_string(grammar, call("1", "Paris")))
        self.assertFalse(_is_grammar_accept_string(grammar, call("1", "1", "x")))
        for either, value in (("1", 1), ('{"x": 1}', {"x": 1})):
            with self.subTest(either=either):
                text = call("1", '{"x": 1}', either)
                _, parsed = parser.parse_non_stream(text)

                self.assertTrue(_is_grammar_accept_string(grammar, text))
                self.assertEqual(
                    json.loads(parsed[0].parameters),
                    {"n": 1, "loc": {"x": 1}, "either": value, "name": "Paris"},
                )

    def test_closers_in_json_values_and_names_stay_in_the_call(self):
        """Closers in JSON strings or names must not end the call, nor a last literal
        keep it open."""
        note = _tool("set_note", {"text": {"type": "string"}})
        strings = {
            "x": {"type": "string"},
            "short": {"type": "string", "maxLength": 3},
            "any": {"type": "string", "pattern": "^.*$"},
            "mark": {"enum": ["a<|end|>b"]},
        }
        place = _tool(
            "set_place",
            {
                "loc": {"type": "object", "properties": strings},
                "n": {"type": "integer"},
            },
        )
        for tool in (note, place):
            tool.function.strict = True
        parser = FunctionCallParser(
            tools=[_tool(), note, place],
            tool_call_parser="beam",
            prefix=PREFIX,
            constrained=True,
        )
        _, tag = parser.get_structure_constraint("required")
        grammar = Grammar.from_structural_tag(tag.model_dump_json())
        city = _parameter("city", "ab")
        cases = [
            ("get_weather", _parameter("city", "a<|end|>b"), None),
            ("get_weather", city, {"city": "ab"}),
            (
                "get_weather",
                city + _parameter("a<|end|>b", "x"),
                {"city": "ab", "a<|end|>b": "x"},
            ),
            ("set_note", _parameter("text", "a<|end|>b"), None),
            ("set_note", _parameter("text", "ab"), {"text": "ab"}),
            (
                "set_place",
                _parameter("loc", '{"x": "a<|end|>b"}'),
                {"loc": {"x": "a<|end|>b"}},
            ),
            ("set_place", _parameter("n", "7"), {"n": 7}),
            (
                "set_place",
                _parameter("loc", '{"x": "a</rfl:parameter>b"}'),
                {"loc": {"x": "a</rfl:parameter>b"}},
            ),
            (
                "set_place",
                _parameter("loc", '{"x": "a\\u003c|end|>b"}'),
                {"loc": {"x": "a<|end|>b"}},
            ),
            (
                "set_place",
                _parameter("loc", '{"short": "<|e"}'),
                {"loc": {"short": "<|e"}},
            ),
            ("set_place", _parameter("loc", '{"short": "<|en"}'), None),
            (
                "set_place",
                _parameter("loc", '{"any": "a<|end|>"}'),
                {"loc": {"any": "a<|end|>"}},
            ),
            (
                "set_place",
                _parameter("loc", '{"mark": "a<|end|>b"}'),
                {"loc": {"mark": "a<|end|>b"}},
            ),
        ]

        for name, body, arguments in cases:
            with self.subTest(name=name, body=body):
                call = _full_call(name, body)
                accepted = _is_grammar_accept_string(grammar, call)

                self.assertIs(accepted, arguments is not None)
                if arguments:
                    _, parsed = parser.parse_non_stream(call)
                    self.assertEqual(json.loads(parsed[0].parameters), arguments)

    def test_text_choices_and_names_keep_to_what_is_read_back(self):
        """A text choice must not be written in a framing the parser reads back
        differently."""
        mode = _tool("set_mode", {"mode": {"type": "string", "enum": ["\nab"]}})
        mode.function.strict = True
        parser = FunctionCallParser(
            tools=[mode], tool_call_parser="beam", prefix=PREFIX
        )
        _, tag = parser.get_structure_constraint("required")
        grammar = Grammar.from_structural_tag(tag.model_dump_json())
        framed = _full_call("set_mode", _parameter("mode", "\nab"))
        _, parsed = parser.parse_non_stream(framed)

        self.assertTrue(_is_grammar_accept_string(grammar, framed))
        self.assertEqual(json.loads(parsed[0].parameters), {"mode": "\nab"})
        # Read back as "ab".
        bare = framed.replace("\n\nab\n", "\nab")
        self.assertFalse(_is_grammar_accept_string(grammar, bare))
        for choice in ("a</rfl:parameter>b", "a<|end|>b"):
            with self.subTest(choice=choice):
                tool = _tool("set_mode", {"mode": {"type": "string", "enum": [choice]}})
                tool.function.strict = True
                parser = FunctionCallParser(tools=[tool], tool_call_parser="beam")

                kind, _ = parser.get_structure_constraint("required")

                self.assertEqual(kind, "json_schema")
        named = _tool("set_mode", {"a<|end|>b": {"type": "string"}})
        named.function.strict = True
        with self.assertRaisesRegex(ValueError, "parameter name"):
            FunctionCallParser(
                tools=[named], tool_call_parser="beam"
            ).get_structure_constraint("required")

    def test_strict_tools_constrain_automatic_tool_choice(self):
        count = _tool("set_count", {"count": {"type": "integer"}})
        count.function.strict = True
        answer = PREFIX + "<|channel|>final<|message|>Sunny."
        cases = [
            (_full_call("get_weather", _parameter("city", "P")), True),
            (answer + "<|return|>", True),
            (
                answer + "<|end|>" + _full_call("get_weather", _parameter("city", "P")),
                False,
            ),
            (_full_call("set_count", _parameter("count", "x")), False),
        ]

        self.assertIsNone(
            FunctionCallParser(
                tools=[_tool()], tool_call_parser="beam"
            ).get_structure_constraint("auto")
        )
        parser = FunctionCallParser(tools=[_tool(), count], tool_call_parser="beam")
        kind, tag = parser.get_structure_constraint("auto")
        grammar = Grammar.from_structural_tag(tag.model_dump_json())
        self.assertEqual(kind, "structural_tag")
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIs(_is_grammar_accept_string(grammar, text), expected)

    def test_forced_call_with_unencodable_tool_name_is_rejected(self):
        parser = FunctionCallParser(
            tools=[_tool("get weather")], tool_call_parser="beam"
        )

        with self.assertRaisesRegex(ValueError, "tool name"):
            parser.get_structure_constraint("required")

    def test_parameter_strict_level_checks_the_tools_it_constrains(self):
        """The parameter strict level must not let an unencodable parameter name
        through."""
        with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
            for strict in (False, True):
                for tool_choice in ("auto", "required"):
                    with self.subTest(strict=strict, tool_choice=tool_choice):
                        tool = _tool("get_weather", {'a"b': {"type": "string"}})
                        tool.function.strict = strict
                        parser = FunctionCallParser(
                            tools=[tool], tool_call_parser="beam"
                        )

                        with self.assertRaisesRegex(ValueError, "parameter name"):
                            parser.get_structure_constraint(tool_choice)
                        self.assertIs(tool.function.strict, strict)


if __name__ == "__main__":
    unittest.main()
