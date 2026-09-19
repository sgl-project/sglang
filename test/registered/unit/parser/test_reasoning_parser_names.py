"""`parser/reasoning_parser_names.py` is the dependency-free copy of the
reasoning parser registry that `server_args` uses for CLI choices; it must list
exactly the registered parsers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.reasoning_parser_names import REASONING_PARSER_NAMES
from sglang.test.test_utils import CustomTestCase


class TestReasoningParserNames(CustomTestCase):
    def test_matches_registry(self):
        self.assertEqual(
            sorted(REASONING_PARSER_NAMES), sorted(ReasoningParser.DetectorMap)
        )

    def test_no_duplicates(self):
        self.assertEqual(len(REASONING_PARSER_NAMES), len(set(REASONING_PARSER_NAMES)))


if __name__ == "__main__":
    unittest.main()
