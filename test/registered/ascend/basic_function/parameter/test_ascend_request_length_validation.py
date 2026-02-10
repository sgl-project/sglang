import unittest

import openai
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestRequestLengthValidation(CustomTestCase):
    """Testcase：Verify set --max-total-tokens and --context-length, can correctly reject inference requests that exceed the limits and throw the specified exceptions.

       [Test Category] Parameter
       [Test Target] --max-total-tokens, --context-length
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start server with auto truncate disabled
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=["--max-total-tokens", "1000", "--context-length", "1000"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_input_length_longer_than_context_length(self):
        # The request is rejected when the number of tokens after the input text is segmented exceeds the model's context-length (1000).
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        long_text = "a" * 1200  # Will tokenize to more than context length
        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_max_tokens_validation(self):
        # The request is rejected if the number of tokens to be generated (max_tokens) specified request exceeds the total token limit configured on the server.
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "a"

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=self.model,
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
