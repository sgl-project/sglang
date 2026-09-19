"""Unit tests for the constraint returned by named tool_choice — no server, no model loading.

With ``tool_choice`` pinned to a specific function, the legacy structural tag must
restrict the grammar to that function AND enforce its real parameters schema even
when ``strict`` is not set. Before the fix, a non-strict tool got an empty schema,
so the grammar constrained only the call format and greedy decoding could emit
minimal arguments like ``{}`` (observed deterministically on Llama-3.2-1B with
the llama3 parser, e2e and offline).
"""

import json

import pytest

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _make_tool(name: str, params: dict, strict: bool = False) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters=params,
            strict=strict,
        ),
    )


ADD_PARAMS = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["a", "b"],
}

WEATHER_PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


class TestNamedToolChoiceConstraint(CustomTestCase):
    def _named_constraint(self, tools, name):
        parser = FunctionCallParser(tools, "llama3")
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name=name)
        )
        constraint = parser.get_structure_constraint(
            tool_choice, parallel_tool_calls=False
        )
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint[0], "structural_tag")
        return json.loads(constraint[1].model_dump_json(by_alias=True))

    def test_named_non_strict_enforces_real_schema(self):
        """Named tool_choice on a non-strict tool must carry the real parameters schema."""
        tools = [_make_tool("add", ADD_PARAMS)]
        tag = self._named_constraint(tools, "add")

        self.assertEqual(len(tag["structures"]), 1)
        schema = tag["structures"][0]["schema"]
        self.assertIn("a", schema.get("properties", {}))
        self.assertIn("b", schema.get("properties", {}))
        self.assertEqual(set(schema.get("required", [])), {"a", "b"})

    def test_named_restricts_to_named_function(self):
        """Only the named function's structure may be in the grammar."""
        tools = [
            _make_tool("add", ADD_PARAMS, strict=True),
            _make_tool("get_weather", WEATHER_PARAMS, strict=True),
        ]
        tag = self._named_constraint(tools, "get_weather")

        self.assertEqual(len(tag["structures"]), 1)
        self.assertIn("city", tag["structures"][0]["schema"].get("properties", {}))
        self.assertNotIn("a", tag["structures"][0]["schema"].get("properties", {}))

    def test_named_grammar_begin_carries_function_name(self):
        """The constrained format must emit the named function's name."""
        tools = [_make_tool("add", ADD_PARAMS)]
        tag = self._named_constraint(tools, "add")
        self.assertIn("add", tag["structures"][0]["begin"])

    def test_required_non_strict_keeps_legacy_behavior(self):
        """tool_choice='required' on non-strict tools keeps the empty (any-JSON) schema."""
        tools = [_make_tool("add", ADD_PARAMS)]
        parser = FunctionCallParser(tools, "llama3")
        constraint = parser.get_structure_constraint("required")

        self.assertIsNotNone(constraint)
        self.assertEqual(constraint[0], "structural_tag")
        tag = json.loads(constraint[1].model_dump_json(by_alias=True))
        self.assertEqual(tag["at_least_one"], True)
        for structure in tag["structures"]:
            # legacy behavior: non-strict tools stay schema-unconstrained here
            self.assertEqual(structure["schema"], {})

    def test_auto_non_strict_unconstrained(self):
        """tool_choice='auto' on non-strict tools stays grammar-free."""
        tools = [_make_tool("add", ADD_PARAMS)]
        parser = FunctionCallParser(tools, "llama3")
        self.assertIsNone(parser.get_structure_constraint("auto"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
