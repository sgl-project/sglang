import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestRequestLengthValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start server with auto truncate disabled
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=("--max-total-tokens", "1000", "--context-length", "1000"),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_input_length_longer_than_context_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello " * 1200  # Will tokenize to more than context length

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_input_length_longer_than_maximum_allowed_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello " * 999  # the maximum allowed length is 994 tokens

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_max_tokens_validation(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello "

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
                max_tokens=1200,
            )

        self.assertIn(
            "max_completion_tokens is too large",
            str(cm.exception),
        )


if __name__ == "__main__":
    unittest.main()
