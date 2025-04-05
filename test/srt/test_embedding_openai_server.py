import unittest

import numpy as np
import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestOpenAIServer(CustomTestCase):
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

    def test_empty_string_embedding(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(
            input="",
            model=self.model,
        )

        assert len(response.data) == 1
        assert isinstance(response.data, list)
        assert response.data[0].embedding is not None
        assert len(response.data[0].embedding) > 0
        assert response.data[0].index == 0
        assert response.data[0].object == "embedding"
        assert response.model == self.model
        assert response.object == "list"

        empty_token_count = len(self.tokenizer.encode(""))
        assert response.usage.prompt_tokens == empty_token_count
        assert response.usage.total_tokens == empty_token_count

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

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
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        inputs = ["Hello world", "This is a test", "SGLang is awesome"]
        response = client.embeddings.create(
            input=inputs,
            model=self.model,
        )

        assert len(response.data) == len(inputs)
        assert isinstance(response.data, list)

        embedding_dim = len(response.data[0].embedding)
        for i, embedding_data in enumerate(response.data):
            assert embedding_data.index == i
            assert embedding_data.object == "embedding"
            assert len(embedding_data.embedding) == embedding_dim

        expected_tokens = sum(len(self.tokenizer.encode(text)) for text in inputs)
        assert response.usage.prompt_tokens == expected_tokens
        assert response.usage.total_tokens == expected_tokens

        mixed_inputs = ["", "Non-empty string", ""]
        mixed_response = client.embeddings.create(
            input=mixed_inputs,
            model=self.model,
        )

        assert len(mixed_response.data) == len(mixed_inputs)
        for embedding_data in mixed_response.data:
            assert embedding_data.embedding is not None
            assert len(embedding_data.embedding) > 0

        text = "The quick brown fox jumps over the lazy dog"
        single_response = client.embeddings.create(
            input=text,
            model=self.model,
        )

        batch_response = client.embeddings.create(
            input=[text],
            model=self.model,
        )

        single_embedding = np.array(single_response.data[0].embedding)
        batch_embedding = np.array(batch_response.data[0].embedding)

        np.testing.assert_allclose(single_embedding, batch_embedding, rtol=1e-5)

        large_inputs = [f"Sample text number {i}" for i in range(10)]
        large_response = client.embeddings.create(
            input=large_inputs,
            model=self.model,
        )

        assert len(large_response.data) == len(large_inputs)
        for i, embedding_data in enumerate(large_response.data):
            assert embedding_data.index == i

    def test_embedding(self):
        for use_list_input in [False, True]:
            for token_input in [False, True]:
                self.run_embedding(use_list_input, token_input)

    def test_batch(self):
        self.run_batch()


if __name__ == "__main__":
    unittest.main()
