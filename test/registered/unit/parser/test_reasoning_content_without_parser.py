import unittest

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")

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

    def test_no_parser_text_passthrough(self):
        """Without a parser, raw text with <think> tags passes through as-is."""
        reasoning_parser = None

        # Simulate serving_chat.py logic
        reasoning_text = None
        text = THINK_OUTPUT
        if reasoning_parser:
            parser = ReasoningParser(reasoning_parser)
            reasoning_text, text = parser.parse_non_stream(text)

        self.assertIsNone(reasoning_text)
        self.assertIn("<think>", text)
        self.assertIn("The answer is 4.", text)

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

    def test_no_parser_streaming_passthrough(self):
        """Without a parser, streaming chunks pass through without reasoning separation."""
        reasoning_parser = None

        # Simulate serving_chat.py streaming logic
        chunks = ["<think>\nLet me", " think.\n</think>\nThe answer", " is 4."]
        all_text = ""
        reasoning_text_seen = False

        for chunk in chunks:
            delta = chunk
            if reasoning_parser:
                # This block would separate reasoning in streaming
                reasoning_text_seen = True
            all_text += delta

        self.assertFalse(reasoning_text_seen)
        self.assertIn("<think>", all_text)


if __name__ == "__main__":
    unittest.main()
