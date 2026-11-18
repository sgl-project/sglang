"""Tool-call payloads that are valid JSON but not objects."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Parsers that reach the shared BaseFormatDetector.parse_base_json without an
# exception guard of their own.
PARSERS = ["qwen25", "llama3", "mistral", "trinity"]

# Of those, the ones whose payload between the markers is itself a JSON array, so
# a malformed sibling can sit next to a valid call. llama3 separates calls with
# ";" and mistral's "[" is part of its bot_token, so neither takes an array here.
ARRAY_PAYLOAD_PARSERS = ["qwen25", "trinity"]

# Valid JSON, but the array elements are not tool-call objects.
NON_OBJECT_PAYLOADS = ["[1, 2]", '["x"]', "[null]", "[[]]"]


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="test tool",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        )
    ]


def _wrap(parser: FunctionCallParser, payload: str) -> str:
    """Wrap a payload in the parser's own tool-call markers."""
    detector = parser.detector
    return (detector.bot_token or "") + payload + (detector.eot_token or "")


class TestMalformedToolCallPayloads(unittest.TestCase):
    def test_non_object_array_entries_are_skipped(self):
        """A tool-call array whose entries are not objects must parse to no calls
        rather than raising, so /parse_function_call cannot be made to return 500.
        """
        for name in PARSERS:
            parser = FunctionCallParser(tools=_tools(), tool_call_parser=name)
            for payload in NON_OBJECT_PAYLOADS:
                with self.subTest(parser=name, payload=payload):
                    _, calls = parser.parse_non_stream(_wrap(parser, payload))
                    self.assertEqual(calls, [])

    def test_valid_call_survives_a_malformed_sibling(self):
        """One non-object entry must not discard the valid calls beside it."""
        payload = '[{"name": "get_weather", "arguments": {"city": "Paris"}}, "junk"]'
        for name in ARRAY_PAYLOAD_PARSERS:
            parser = FunctionCallParser(tools=_tools(), tool_call_parser=name)
            with self.subTest(parser=name):
                _, calls = parser.parse_non_stream(_wrap(parser, payload))
                self.assertEqual([c.name for c in calls], ["get_weather"])


if __name__ == "__main__":
    unittest.main()
