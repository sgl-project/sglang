import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestOpenAIServerIgnoreEOS(CustomTestCase):
    """Testcase: Test 'ignore_eos' is True, the EOS is ignore and continue reasoning

    [Test Category] Interface
    [Test Target] ignore_eos
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=cls.other_args,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_ignore_eos(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        max_tokens = 200

        response_default = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": False},
        )

        response_ignore_eos = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
        )

        default_tokens = len(
            self.tokenizer.encode(response_default.choices[0].message.content)
        )
        ignore_eos_tokens = len(
            self.tokenizer.encode(response_ignore_eos.choices[0].message.content)
        )

        # Check if ignore_eos resulted in more tokens or exactly max_tokens
        # The ignore_eos response should either:
        # 1. Have more tokens than the default response (if default stopped at EOS before max_tokens)
        # 2. Have exactly max_tokens (if it reached the max_tokens limit)
        self.assertTrue(
            ignore_eos_tokens > default_tokens or ignore_eos_tokens >= max_tokens,
            f"ignore_eos did not generate more tokens: {ignore_eos_tokens} vs {default_tokens}",
        )

        self.assertEqual(
            response_ignore_eos.choices[0].finish_reason,
            "length",
            f"Expected finish_reason='length' for ignore_eos=True, got {response_ignore_eos.choices[0].finish_reason}",
        )


if __name__ == "__main__":
    unittest.main()
