"""Tests for Beam response-template parsing."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
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


if __name__ == "__main__":
    unittest.main()
