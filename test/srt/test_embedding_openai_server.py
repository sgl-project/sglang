import unittest

import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestOpenAIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "intfloat/e5-mistral-7b-instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid, include_self=True)

    def run_embedding(self, use_list_input, token_input):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
        if token_input:
            prompt_input = self.tokenizer.encode(prompt)
            num_prompt_tokens = len(prompt_input)
        else:
            prompt_input = prompt
            num_prompt_tokens = len(self.tokenizer.encode(prompt))

        if use_list_input:
            prompt_arg = [prompt_input] * 2
            num_prompts = len(prompt_arg)
            num_prompt_tokens *= num_prompts
        else:
            prompt_arg = prompt_input
            num_prompts = 1

        response = client.embeddings.create(
            input=prompt_arg,
            model=self.model,
        )

        assert len(response.data) == num_prompts
        assert isinstance(response.data, list)
        assert response.data[0].embedding
        assert response.data[0].index is not None
        assert response.data[0].object == "embedding"
        assert response.model == self.model
        assert response.object == "list"
        assert (
            response.usage.prompt_tokens == num_prompt_tokens
        ), f"{response.usage.prompt_tokens} vs {num_prompt_tokens}"
        assert (
            response.usage.total_tokens == num_prompt_tokens
        ), f"{response.usage.total_tokens} vs {num_prompt_tokens}"

    def run_batch(self):
        # FIXME: not implemented
        pass

    def test_embedding(self):
        # TODO: the fields of encoding_format, dimensions, user are skipped
        # TODO: support use_list_input
        for use_list_input in [False, True]:
            for token_input in [False, True]:
                self.run_embedding(use_list_input, token_input)

    def test_batch(self):
        self.run_batch()


if __name__ == "__main__":
    unittest.main()
