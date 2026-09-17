"""`function_call/parser_names.py` is the dependency-free copy of the tool-call
parser registry that `server_args` uses for CLI choices; it must list exactly
the registered parsers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.parser_names import TOOL_CALL_PARSER_NAMES
from sglang.test.test_utils import CustomTestCase


class TestToolCallParserNames(CustomTestCase):
    def test_matches_registry(self):
        self.assertEqual(
            sorted(TOOL_CALL_PARSER_NAMES),
            sorted(FunctionCallParser.ToolCallParserEnum),
        )

    def test_no_duplicates(self):
        self.assertEqual(len(TOOL_CALL_PARSER_NAMES), len(set(TOOL_CALL_PARSER_NAMES)))


if __name__ == "__main__":
    unittest.main()
