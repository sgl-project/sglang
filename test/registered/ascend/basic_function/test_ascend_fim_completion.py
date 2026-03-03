import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestFimCompletion(CustomTestCase):
    """Testcase：Verify set --completion-template, the model's FIM (Fill-in-the-Middle) completion function work correctly.

    [Test Category] Parameter
    [Test Target] --completion-template
    """

    model = "/root/.cache/modelscope/hub/models/deepseek-ai/deepseek-coder-1.3b-base"
    other_args = [
        "--completion-template",
        "deepseek_coder",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        0.8,
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
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

    def run_fim_completion(self, number_of_completion):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "function sum(a: number, b: number): number{\n"
        suffix = "}"

        prompt_input = self.tokenizer.encode(prompt) + self.tokenizer.encode(suffix)
        num_prompt_tokens = len(prompt_input) + 2

        response = client.completions.create(
            model=self.model,
            prompt=prompt,
            suffix=suffix,
            temperature=0.3,
            max_tokens=32,
            stream=False,
            n=number_of_completion,
        )
        assert len(response.choices) == number_of_completion
        assert response.id
        assert response.created
        assert response.object == "text_completion"
        assert (
            response.usage.prompt_tokens == num_prompt_tokens
        ), f"{response.usage.prompt_tokens} vs {num_prompt_tokens}"
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_fim_completion(self):
        for number_of_completion in [1, 3]:
            self.run_fim_completion(number_of_completion)


class TestFimCompletionJson(TestFimCompletion):
    other_args = [
        "--completion-template",
        "./deepseek_coder.json",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        0.8,
    ]


if __name__ == "__main__":
    unittest.main()
