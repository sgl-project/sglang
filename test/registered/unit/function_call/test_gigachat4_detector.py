"""GCML streaming and non-streaming must agree when generation ends."""

import json
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

OPEN = "<｜GCML｜tool_calls>"
CLOSE = "</｜GCML｜tool_calls>"
PROSE = "Checking.\n\n"


def invoke(name, value):
    return (
        f'<｜GCML｜invoke name="{name}">'
        f'<｜GCML｜parameter name="key" string="true">{value}'
        "</｜GCML｜parameter></｜GCML｜invoke>"
    )


class TestGigaChat4Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(type="function", function=Function(name=name, parameters={}))
            for name in ("lookup_alpha", "lookup_beta")
        ]
        self.enterContext(
            patch.object(envs.SGLANG_FORWARD_UNKNOWN_TOOLS, "get", return_value=False)
        )

    def _assert_parses(self, text, expected_text, expected_calls, extra_chunks=()):
        def normalized(calls):
            return [
                (call.tool_index, call.name, json.loads(call.parameters))
                for call in calls
            ]

        parser = FunctionCallParser(self.tools, "gigachat4")
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual((normal, normalized(calls)), (expected_text, expected_calls))

        chunkings = [[text[:i], text[i:]] for i in range(len(text) + 1)]
        chunkings.extend([[text], list(text), *extra_chunks])
        for chunks in chunkings:
            with self.subTest(chunks=chunks):
                self.assertEqual("".join(chunks), text)
                parser = FunctionCallParser(self.tools, "gigachat4")
                normal, calls = "", []
                for chunk in chunks:
                    delta, emitted = parser.parse_stream_chunk(chunk)
                    normal += delta
                    calls.extend(emitted)
                delta, emitted = parser.parse_stream_end()
                normal += delta
                calls.extend(emitted)
                self.assertEqual(
                    (normal, normalized(calls)), (expected_text, expected_calls)
                )
                self.assertEqual(parser.parse_stream_end(), ("", []))

    def test_prose_followed_by_complete_tool_block_in_last_chunk(self):
        block = OPEN + invoke("lookup_alpha", "alpha") + CLOSE
        self._assert_parses(
            PROSE + block,
            PROSE,
            [(0, "lookup_alpha", {"key": "alpha"})],
            extra_chunks=([PROSE, block], ["Checking.", "\n\n" + block]),
        )

    def test_generation_ends_inside_opening_tag(self):
        for end in range(1, len(OPEN)):
            text = PROSE + OPEN[:end]
            self._assert_parses(text, text, [])

    def test_generation_ends_inside_first_call(self):
        call = invoke("lookup_alpha", "alpha")
        for suffix in ('<｜GCML｜invoke name="lookup_', 'string="true">al'):
            end = call.index(suffix) + len(suffix)
            text = PROSE + OPEN + call[:end]
            self._assert_parses(text, text, [])

    def test_generation_ends_inside_second_call(self):
        first = invoke("lookup_alpha", "alpha")
        second = invoke("lookup_beta", "beta")
        for suffix in ('<｜GCML｜invoke name="lookup_', 'string="true">be'):
            end = second.index(suffix) + len(suffix)
            self._assert_parses(
                PROSE + OPEN + first + second[:end],
                PROSE,
                [(0, "lookup_alpha", {"key": "alpha"})],
            )

    def test_unknown_tool_name_is_not_emitted(self):
        self._assert_parses(
            PROSE + OPEN + invoke("unknown_tool", "value") + CLOSE, PROSE, []
        )


if __name__ == "__main__":
    unittest.main()
