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
    constrain: bool = True,
) -> str:
    return PREFIX + _continuation_call(
        name,
        body,
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

    def test_output_cut_before_the_next_message_start_ends_its_region(self):
        """Output that stops after `<|end|>`, such as at `max_tokens`, has ended
        its reasoning or call and lacks only the next message's start."""
        for end in ("<|end|>", "<|end|><|start|>"):
            with self.subTest(end=end):
                reasoning_parser = ReasoningParser(model_type="beam", prefix=PREFIX)
                parser = FunctionCallParser(
                    tools=[_tool()], tool_call_parser="beam", prefix=PREFIX
                )
                generation = _continuation_call(
                    "get_weather", _parameter("city", "Paris"), terminator=end
                )

                reasoning, _ = reasoning_parser.parse_non_stream(
                    "<|channel|>analysis<|message|>plan" + end
                )
                _, calls = parser.parse_non_stream(generation)

                self.assertEqual(reasoning, "plan")
                self.assertEqual([call.name for call in calls], ["get_weather"])
                self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris"})

    def test_value_ends_before_an_opener_on_the_next_line(self):
        tools = [
            _tool("write", {"path": {"type": "string"}, "content": {"type": "string"}})
        ]
        generation = _continuation_call(
            "write",
            '<rfl:parameter name="content">\nprint(1)\n\n' + _parameter("path", "a.py"),
        )
        expected = {"content": "print(1)\n", "path": "a.py"}
        parser = FunctionCallParser(tools=tools, tool_call_parser="beam", prefix=PREFIX)
        _, tag = parser.get_structure_constraint("required")
        grammar = Grammar.from_structural_tag(tag.model_dump_json())

        _, calls = parser.parse_non_stream(generation)

        self.assertTrue(_is_grammar_accept_string(grammar, generation))
        self.assertEqual(json.loads(calls[0].parameters), expected)

    def test_declared_tool_names_are_kept(self):
        for name in ("refl.search", "2fa_check"):
            with self.subTest(name=name):
                parser = FunctionCallParser(
                    tools=[_tool(name, {"query": {"type": "string"}})],
                    tool_call_parser="beam",
                    prefix=PREFIX,
                )
                generation = _continuation_call(name, _parameter("query", "weather"))
                _, tag = parser.get_structure_constraint("required")
                grammar = Grammar.from_structural_tag(tag.model_dump_json())

                content, calls = parser.parse_non_stream(generation)

                self.assertTrue(_is_grammar_accept_string(grammar, generation))
                self.assertEqual(content, "")
                self.assertEqual([call.name for call in calls], [name])
                self.assertEqual(json.loads(calls[0].parameters), {"query": "weather"})

    def test_reasoning_cut_off_in_its_closer_is_kept_as_written(self):
        """The rest of a closer that ends the parse is no output."""
        generation = "<|channel|>analysis<|message|>Thinking.<|end|>"
        for stream in (False, True):
            with self.subTest(stream=stream):
                parser = FunctionCallParser(
                    tools=[_tool()],
                    tool_call_parser="beam",
                    prefix=PREFIX,
                    constrained=True,
                )
                if stream:
                    content = "".join(
                        parser.parse_stream_chunk(char)[0] for char in generation
                    )
                    content += parser.parse_stream_end()[0]
                else:
                    content = parser.parse_non_stream(generation)[0]

                self.assertEqual(content, generation)

    def test_undeclared_tool_calls_follow_the_forwarding_setting(self):
        command = _parameter("command", "pwd")
        generation = _continuation_call(
            "bash", command, terminator="<|end|>"
        ) + _full_call("Bash", command)
        for forward, names in ((False, ["bash"]), (True, ["bash", "Bash"])):
            for stream in (False, True):
                with self.subTest(forward=forward, stream=stream):
                    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward):
                        parser = FunctionCallParser(
                            tools=[_tool("bash", {"command": {"type": "string"}})],
                            tool_call_parser="beam",
                            prefix=PREFIX,
                        )
                        if stream:
                            calls = []
                            for char in generation:
                                calls += parser.parse_stream_chunk(char)[1]
                        else:
                            calls = parser.parse_non_stream(generation)[1]

                    self.assertEqual([call.name for call in calls if call.name], names)

    def test_forced_tool_calls_use_template_grammar(self):
        """Forced calls after the reasoning follow the template grammar, which
        accepts only calls the parser reads back."""
        count = _tool("set_count", {"count": {"type": "integer"}})
        count.function.strict = True
        parser = FunctionCallParser(
            tools=[_tool(), count], tool_call_parser="beam", prefix=PREFIX
        )
        weather = ("get_weather", {"city": "Paris"})
        weather_call = _continuation_call("get_weather", _parameter("city", "Paris"))
        cases = [
            ("required", weather_call, [weather]),
            (
                "required",
                _continuation_call(
                    "get_weather", _parameter("city", "Paris"), constrain=False
                ),
                None,
            ),
            (
                "required",
                _continuation_call(
                    "get_weather", _parameter("city", "Paris"), terminator="<|end|>"
                )
                + _full_call("set_count", _parameter("count", "7")),
                [weather, ("set_count", {"count": 7})],
            ),
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
                _continuation_call(
                    "get_weather", _parameter("city", "Paris"), terminator="<|end|>"
                )
                + _full_call("get_weather", _parameter("city", "Paris")),
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
            return _continuation_call(
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
        """Closers in JSON strings must not end the call."""
        strings = {"x": {"type": "string"}}
        spot = _tool("set_spot", {"loc": {"type": "object", "properties": strings}})
        weather = _tool()
        weather.function.parameters["additionalProperties"] = True
        parser = FunctionCallParser(
            tools=[weather, spot],
            tool_call_parser="beam",
            prefix=PREFIX,
            constrained=True,
        )
        _, tag = parser.get_structure_constraint("required")
        grammar = Grammar.from_structural_tag(tag.model_dump_json())
        cases = [
            ("get_weather", _parameter("city", "a<|end|><|start|>assistantb"), None),
            ("get_weather", _parameter("city", "a<|end|>b"), {"city": "a<|end|>b"}),
            (
                "set_spot",
                _parameter("loc", '{"x": "a<|end|><|start|>assistantb"}'),
                {"loc": {"x": "a<|end|><|start|>assistantb"}},
            ),
        ]

        for name, body, arguments in cases:
            with self.subTest(name=name, body=body):
                call = _continuation_call(name, body)
                accepted = _is_grammar_accept_string(grammar, call)

                self.assertIs(accepted, arguments is not None)
                if arguments:
                    _, parsed = parser.parse_non_stream(call)
                    self.assertEqual(json.loads(parsed[0].parameters), arguments)

    def test_text_choices_and_names_keep_to_what_is_read_back(self):
        """A text choice must not be written in a framing the parser reads back
        differently."""
        # Only the closer after a newline ends the value.
        choices = ["\nab", "a</rfl:parameter>b"]
        mode = _tool("set_mode", {"mode": {"type": "string", "enum": choices}})
        mode.function.strict = True
        parser = FunctionCallParser(
            tools=[mode], tool_call_parser="beam", prefix=PREFIX
        )
        _, tag = parser.get_structure_constraint("required")
        grammar = Grammar.from_structural_tag(tag.model_dump_json())
        for choice in choices:
            with self.subTest(choice=choice):
                framed = _continuation_call("set_mode", _parameter("mode", choice))
                _, parsed = parser.parse_non_stream(framed)

                self.assertTrue(_is_grammar_accept_string(grammar, framed))
                self.assertEqual(json.loads(parsed[0].parameters), {"mode": choice})
        # Read back as "ab".
        bare = _continuation_call("set_mode", _parameter("mode", "\nab"))
        bare = bare.replace("\n\nab\n", "\nab")
        self.assertFalse(_is_grammar_accept_string(grammar, bare))
        for choice in ("a\n</rfl:parameter>b", "a<|end|><|start|>assistantb"):
            with self.subTest(choice=choice):
                tool = _tool("set_mode", {"mode": {"type": "string", "enum": [choice]}})
                tool.function.strict = True
                parser = FunctionCallParser(tools=[tool], tool_call_parser="beam")

                kind, _ = parser.get_structure_constraint("required")

                self.assertEqual(kind, "json_schema")
        named = _tool("set_mode", {"a<|end|><|start|>assistantb": {"type": "string"}})
        named.function.strict = True
        with self.assertRaisesRegex(ValueError, "parameter name"):
            FunctionCallParser(
                tools=[named], tool_call_parser="beam"
            ).get_structure_constraint("required")

    def test_string_formats_and_patterns_are_held_in_raw_text(self):
        """Like XGrammar, a strict string keeps its format or pattern, unless a
        match could end the value early."""
        contact = _tool(
            "set_contact",
            {
                "email": {"type": "string", "format": "email"},
                "code": {"type": "string", "pattern": "^[A-Z]{3}$"},
                "n": {"type": "integer"},
            },
        )
        contact.function.parameters["required"] = ["email", "code", "n"]
        contact.function.strict = True
        parser = FunctionCallParser(
            tools=[contact], tool_call_parser="beam", prefix=PREFIX
        )
        kind, tag = parser.get_structure_constraint("required")
        grammar = Grammar.from_structural_tag(tag.model_dump_json())

        def call(email: str, code: str, *rest: str) -> str:
            body = _parameter("email", email) + _parameter("code", code)
            return _continuation_call("set_contact", body + "".join(rest))

        text = call("a.b@x.co", "ABC", _parameter("n", "1"))
        _, parsed = parser.parse_non_stream(text)
        self.assertEqual(kind, "structural_tag")
        self.assertTrue(_is_grammar_accept_string(grammar, text))
        self.assertEqual(
            json.loads(parsed[0].parameters),
            {"email": "a.b@x.co", "code": "ABC", "n": 1},
        )
        for rejected in (
            call("ab", "ABC", _parameter("n", "1")),
            call("a.b@x.co", "ABCD", _parameter("n", "1")),
            call("a.b@x.co", "ABC"),
        ):
            with self.subTest(rejected=rejected):
                self.assertFalse(_is_grammar_accept_string(grammar, rejected))
        # Line breaks could end the value; XGrammar holds length bounds only in JSON.
        for schema in (
            {"type": "string", "pattern": "^a.b$"},
            {"type": "string", "minLength": 3},
        ):
            with self.subTest(schema=schema):
                tool = _tool("set_code", {"code": schema})
                tool.function.strict = True
                parser = FunctionCallParser(tools=[tool], tool_call_parser="beam")

                kind, _ = parser.get_structure_constraint("required")

                self.assertEqual(kind, "json_schema")

    def test_automatic_calls_hold_declared_arguments_and_names(self):
        """At any strict level, declared arguments are held where they can be,
        and calls name declared tools only."""
        count = _tool("set_count", {"count": {"type": "integer"}})
        unheld = _tool("find", {"q": {"type": "string"}})
        unheld.function.parameters["minProperties"] = 1
        any_arguments = _parameter("x", "1") + _parameter("x", "y")

        for level in (ToolStrictLevel.OFF, ToolStrictLevel.PARAMETER):
            with envs.SGLANG_TOOL_STRICT_LEVEL.override(level.value):
                tools = [_tool(), count] + ([unheld] if not level else [])
                parser = FunctionCallParser(tools=tools, tool_call_parser="beam")
                tag = parser.detector.get_structural_tag(
                    tools=parser.structural_tag_tools(), tool_choice="auto"
                )
            grammar = Grammar.from_structural_tag(tag.model_dump_json())
            for text, expected in [
                (_continuation_call("get_weather", _parameter("city", "P")), True),
                (_continuation_call("get_weather", _parameter("town", "P")), False),
                (_continuation_call("set_count", _parameter("count", "1")), True),
                (_continuation_call("set_count", _parameter("count", "x")), False),
                (_continuation_call("find", any_arguments), not level),
                (_continuation_call("search_files", any_arguments), False),
                (_continuation_call("set_counts", any_arguments), False),
            ]:
                with self.subTest(level=level, text=text):
                    self.assertIs(_is_grammar_accept_string(grammar, text), expected)

    def test_parameter_strict_level_checks_the_tools_it_constrains(self):
        """The parameter strict level must not let an unencodable parameter name
        through."""
        with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER.value):
            for tool_choice in ("auto", "required"):
                with self.subTest(tool_choice=tool_choice):
                    tool = _tool("get_weather", {'a"b': {"type": "string"}})
                    parser = FunctionCallParser(tools=[tool], tool_call_parser="beam")

                    with self.assertRaisesRegex(ValueError, "parameter name"):
                        parser.get_structure_constraint(tool_choice)
                    self.assertFalse(tool.function.strict)


if __name__ == "__main__":
    unittest.main()
