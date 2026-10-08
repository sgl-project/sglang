"""Unit tests for the MiMo tool-call detector — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_search_tool() -> Tool:
    return Tool(
        type="function",
        function=Function(
            name="search",
            description="Search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "flag": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        ),
    )


def mimo_call(name: str, **params: str) -> str:
    body = "".join(f"<parameter={k}>{v}</parameter>\n" for k, v in params.items())
    return (
        "<tool_call>\n" f"<function={name}>\n" f"{body}" "</function>\n" "</tool_call>"
    )


class TestMiMoDetectorStringCoercion(unittest.TestCase):
    """A string parameter carries literal text.

    Coercing before the declared type is consulted rewrites values that merely
    read like JSON keywords: a search for the word "null" became a JSON null,
    and an invalid boolean such as "yes" became false.
    """

    def setUp(self):
        self.tools = [make_search_tool()]
        self.detector = MiMoDetector()

    def parse_one(self, **params):
        result = self.detector.detect_and_parse(
            mimo_call("search", **params), self.tools
        )
        self.assertEqual(len(result.calls), 1)
        return json.loads(result.calls[0].parameters)

    def test_plain_string_is_unchanged(self):
        self.assertEqual(self.parse_one(query="hello"), {"query": "hello"})

    def test_string_named_null_keeps_literal_text(self):
        args = self.parse_one(query="null")
        self.assertEqual(args, {"query": "null"})
        self.assertIsInstance(args["query"], str)

    def test_string_named_null_is_case_insensitive(self):
        for raw in ("null", "NULL", "Null"):
            with self.subTest(raw=raw):
                args = self.parse_one(query=raw)
                self.assertEqual(args, {"query": raw})
                self.assertIsInstance(args["query"], str)

    def test_string_keeps_json_looking_literals(self):
        for raw in ("123", "true", "false", "{}", "[]", '""'):
            with self.subTest(raw=raw):
                args = self.parse_one(query=raw)
                self.assertEqual(args, {"query": raw})
                self.assertIsInstance(args["query"], str)

    def test_non_string_types_still_coerce(self):
        # The fix must not stop real conversions for typed parameters.
        self.assertEqual(
            self.parse_one(query="a", flag="true"), {"query": "a", "flag": True}
        )
        self.assertEqual(
            self.parse_one(query="a", flag="false"), {"query": "a", "flag": False}
        )
        self.assertEqual(
            self.parse_one(query="a", limit="7"), {"query": "a", "limit": 7}
        )

    def test_null_still_decodes_for_non_string_types(self):
        self.assertEqual(
            self.parse_one(query="a", limit="null"), {"query": "a", "limit": None}
        )


if __name__ == "__main__":
    unittest.main()
