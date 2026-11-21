import unittest

from sglang.srt.entrypoints.openai import encoding_dsv32
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEncodingDsv32(CustomTestCase):
    def test_thinking_mode_keeps_developer_messages(self):
        """Dropping earlier reasoning in thinking mode must keep developer turns.

        A developer message before the last user-like turn was left out of the
        thinking-mode message list, so encoding indexed past its end and failed
        with DS32EncodingError (HTTP 400) instead of rendering the developer turn.
        """
        conversations = [
            # An OpenAI-style developer instruction ahead of the user turn.
            [
                {"role": "system", "content": ""},
                {"role": "developer", "content": "Answer in French."},
                {"role": "user", "content": "What is the capital of Italy?"},
            ],
            # A second developer turn after an assistant reply.
            [
                {"role": "developer", "content": "Find the population of Rome."},
                {
                    "role": "assistant",
                    "content": "About 2.8M.",
                    "reasoning_content": "r",
                },
                {"role": "developer", "content": "And Milan?"},
            ],
        ]
        for messages in conversations:
            with self.subTest(roles=[m["role"] for m in messages]):
                prompt = encoding_dsv32.encode_messages(
                    messages, thinking_mode="thinking"
                )
                for message in messages:
                    if message["role"] == "developer":
                        self.assertIn(message["content"], prompt)
                self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))


if __name__ == "__main__":
    unittest.main()
