"""Tests for Beam response-template parsing."""

import json
import unittest

from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice
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


def _continuation_call(name: str, body: str, *, terminator: str = "<|call|>") -> str:
    return (
        f" to={name}<|channel|>analysis <|constrain|>xml<|message|>{body}{terminator}"
    )


def _full_call(name: str, body: str, *, terminator: str = "<|call|>") -> str:
    return PREFIX + _continuation_call(name, body, terminator=terminator)


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

    def test_call_stop_token_ends_answer(self):
        reasoning_parser = ReasoningParser(model_type="beam", prefix=PREFIX)
        tool_parser = FunctionCallParser(
            tools=[_tool()], tool_call_parser="beam", prefix=PREFIX
        )

        reasoning, tool_wire = reasoning_parser.parse_non_stream(
            "<|channel|>final<|message|>done<|call|>"
        )

        self.assertEqual(reasoning, "")
        self.assertEqual(tool_parser.parse_non_stream(tool_wire), ("done", []))

    def test_forced_tool_calls_use_template_grammar(self):
        parser = FunctionCallParser(
            tools=[_tool(), _tool("set_count", {"count": {"type": "integer"}})],
            tool_call_parser="beam",
        )
        thought = "<|channel|>analysis<|message|>plan<|end|><|start|>assistant"
        city_call = _continuation_call("get_weather", _parameter("city", "Paris"))
        cases = [
            ("required", city_call, True),
            ("required", thought + city_call, True),
            (
                "required",
                thought + _continuation_call("set_count", _parameter("count", "x")),
                False,
            ),
            ("required", thought + "<|channel|>final<|message|>Paris<|return|>", False),
            (
                ToolChoice(type="function", function={"name": "set_count"}),
                thought + city_call,
                False,
            ),
            (
                ToolChoice(type="function", function={"name": "get_weather"}),
                thought + city_call,
                True,
            ),
            (
                ToolChoice(type="function", function={"name": "get_weather"}),
                thought
                + _continuation_call(
                    "get_weather", _parameter("city", "Paris"), terminator="<|end|>"
                )
                + "<|start|>assistant"
                + city_call,
                False,
            ),
        ]

        self.assertTrue(parser.detector.supports_structural_tag())
        for tool_choice, text, expected in cases:
            with self.subTest(tool_choice=tool_choice, text=text):
                kind, tag = parser.get_structure_constraint(tool_choice)
                grammar = Grammar.from_structural_tag(tag.model_dump_json())

                self.assertEqual(kind, "structural_tag")
                self.assertIs(_is_grammar_accept_string(grammar, text), expected)

    def test_strict_tools_constrain_automatic_tool_choice(self):
        count = _tool("set_count", {"count": {"type": "integer"}})
        count.function.strict = True
        thought = "<|channel|>analysis<|message|>plan<|end|><|start|>assistant"
        final = "<|channel|>final<|message|>"
        cases = [
            (
                thought + _continuation_call("get_weather", _parameter("city", "P")),
                True,
            ),
            (thought + final + "Sunny.", True),
            (final + "Sunny.<|return|>", True),
            (
                thought
                + final
                + "Sunny.<|end|><|start|>assistant"
                + _continuation_call("get_weather", _parameter("city", "P")),
                False,
            ),
            (
                thought + _continuation_call("set_count", _parameter("count", "x")),
                False,
            ),
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


if __name__ == "__main__":
    unittest.main()
