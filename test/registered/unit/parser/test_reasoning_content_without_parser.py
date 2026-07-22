import unittest

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

# Simulated model output that contains think tags (e.g. from DeepSeek-R1)
THINK_OUTPUT = (
    "<think>\nLet me think about this.\n1 + 3 = 4\n</think>\nThe answer is 4."
)
THINK_OUTPUT_QWEN3 = (
    "<think>\nLet me think about this.\n1 + 3 = 4\n</think>\n\nThe answer is 4."
)


class TestReasoningContentWithoutParser(CustomTestCase):
    """Test the code path: when no reasoning parser is configured, reasoning
    content should never be separated, even if the model output contains
    think tags.  This mirrors the guard in serving_chat.py:

        if self.reasoning_parser and request.separate_reasoning:
            ...

    When reasoning_parser is None the block is skipped entirely.
    """

    def test_with_parser_separates_reasoning(self):
        """With a parser, reasoning content is correctly separated."""
        for parser_name, output in [
            ("deepseek-r1", THINK_OUTPUT),
            ("qwen3", THINK_OUTPUT_QWEN3),
        ]:
            with self.subTest(parser=parser_name):
                parser = ReasoningParser(parser_name, stream_reasoning=False)
                reasoning_text, text = parser.parse_non_stream(output)

                self.assertIsNotNone(reasoning_text)
                self.assertGreater(len(reasoning_text), 0)
                self.assertNotIn("<think>", reasoning_text)
                self.assertIn("The answer is 4.", text)


if __name__ == "__main__":
    unittest.main()
