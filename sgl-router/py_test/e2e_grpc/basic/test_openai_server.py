"""
gRPC Router E2E Test - OpenAI Server API Compatibility

This test file is REUSED from test/srt/openai_server/basic/test_openai_server.py
with minimal changes:
- Swap popen_launch_server() → popen_launch_grpc_router()
- Update teardown to cleanup router + workers
- All test logic and assertions remain identical

Run with:
    python3 -m pytest e2e_grpc/basic/test_openai_server.py -v
    python3 -m unittest e2e_grpc.basic.test_openai_server.TestOpenAIServer.test_completion
"""

import json
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
)

# CHANGE: Import router launcher instead of server launcher
import sys
from pathlib import Path
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_grpc_router


class TestOpenAIServer(CustomTestCase):
    """
    Test OpenAI API through gRPC router.

    REUSED from test/srt/openai_server/basic/test_openai_server.py
    ONLY CHANGE: Server launch mechanism
      - Uses sglang_router.launch_server --grpc-mode
      - Single command launches router + workers
    """

    @classmethod
    def setUpClass(cls):
        # Use llama-3.1-8b-instruct for e2e testing
        cls.model = "/home/ubuntu/models/llama-3.1-8b-instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # CHANGE: Launch gRPC router with integrated workers (single command)
        cls.cluster = popen_launch_grpc_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            num_workers=2,
            policy="round_robin",
            api_key=cls.api_key,
        )

        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        # CHANGE: Cleanup single process (router + workers integrated)
        kill_process_tree(cls.cluster["process"].pid)

    # ALL TEST METHODS BELOW ARE UNCHANGED FROM ORIGINAL
    # They validate that the router maintains OpenAI API compatibility

    def run_completion(
        self, echo, logprobs, use_list_input, parallel_sample_num, token_input
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
        if token_input:
            prompt_input = self.tokenizer.encode(prompt)
            num_prompt_tokens = len(prompt_input)
        else:
            prompt_input = prompt
            num_prompt_tokens = len(self.tokenizer.encode(prompt))

        if use_list_input:
            prompt_arg = [prompt_input, prompt_input]
            num_choices = len(prompt_arg)
            num_prompt_tokens *= 2
        else:
            prompt_arg = prompt_input
            num_choices = 1

        response = client.completions.create(
            model="unknown",
            prompt=prompt_arg,
            temperature=0,
            max_tokens=32,
            echo=echo,
            logprobs=logprobs,
            n=parallel_sample_num,
        )

        assert len(response.choices) == num_choices * parallel_sample_num

        if echo:
            text = response.choices[0].text
            assert text.startswith(prompt)

        if logprobs:
            assert response.choices[0].logprobs
            assert isinstance(response.choices[0].logprobs.tokens[0], str)
            assert isinstance(response.choices[0].logprobs.top_logprobs[1], dict)
            ret_num_top_logprobs = len(response.choices[0].logprobs.top_logprobs[1])
            assert ret_num_top_logprobs > 0

            if not echo:
                assert response.choices[0].logprobs.token_logprobs[0]

        assert response.id
        assert response.created
        assert (
            response.usage.prompt_tokens == num_prompt_tokens
        ), f"{response.usage.prompt_tokens} vs {num_prompt_tokens}"
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def run_completion_stream(
        self, echo, logprobs, use_list_input, parallel_sample_num, token_input
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
        if token_input:
            prompt_input = self.tokenizer.encode(prompt)
            num_prompt_tokens = len(prompt_input)
        else:
            prompt_input = prompt
            num_prompt_tokens = len(self.tokenizer.encode(prompt))

        if use_list_input:
            prompt_arg = [prompt_input, prompt_input]
            num_choices = len(prompt_arg)
            num_prompt_tokens *= 2
        else:
            prompt_arg = prompt_input
            num_choices = 1

        generator = client.completions.create(
            model="unknown",
            prompt=prompt_arg,
            temperature=0,
            max_tokens=32,
            echo=echo,
            logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
        )

        is_firsts = {}
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0, f"usage.prompt_tokens was zero"
                assert usage.completion_tokens > 0, f"usage.completion_tokens was zero"
                assert usage.total_tokens > 0, f"usage.total_tokens was zero"
                continue

            index = response.choices[0].index
            is_first = is_firsts.get(index, True)

            if logprobs:
                assert response.choices[0].logprobs, f"no logprobs in response"
                assert isinstance(
                    response.choices[0].logprobs.tokens[0], str
                ), f"{response.choices[0].logprobs.tokens[0]} is not a string"
                if not (is_first and echo):
                    assert isinstance(
                        response.choices[0].logprobs.top_logprobs[0], dict
                    ), f"top_logprobs was not a dictionary"
                    ret_num_top_logprobs = len(
                        response.choices[0].logprobs.top_logprobs[0]
                    )
                    assert ret_num_top_logprobs > 0, f"ret_num_top_logprobs was 0"

            if is_first:
                if echo:
                    assert response.choices[0].text.startswith(
                        prompt
                    ), f"{response.choices[0].text} and all args {echo} {logprobs} {token_input} {is_first}"
                is_firsts[index] = False
            assert response.id, f"no id in response"
            assert response.created, f"no created in response"

        for index in [i for i in range(parallel_sample_num * num_choices)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

    def run_chat_completion(self, logprobs, parallel_sample_num):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="unknown",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                },
            ],
            temperature=0,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            n=parallel_sample_num,
        )

        if logprobs:
            assert isinstance(
                response.choices[0].logprobs.content[0].top_logprobs[0].token, str
            )

            ret_num_top_logprobs = len(
                response.choices[0].logprobs.content[0].top_logprobs
            )
            assert (
                ret_num_top_logprobs == logprobs
            ), f"{ret_num_top_logprobs} vs {logprobs}"

        assert len(response.choices) == parallel_sample_num
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def run_chat_completion_stream(self, logprobs, parallel_sample_num=1):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        generator = client.chat.completions.create(
            model="unknown",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
        )

        is_firsts = {}
        is_finished = {}
        finish_reason_counts = {}
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0, f"usage.prompt_tokens was zero"
                assert usage.completion_tokens > 0, f"usage.completion_tokens was zero"
                assert usage.total_tokens > 0, f"usage.total_tokens was zero"
                continue

            index = response.choices[0].index
            finish_reason = response.choices[0].finish_reason
            if finish_reason is not None:
                is_finished[index] = True
                finish_reason_counts[index] = finish_reason_counts.get(index, 0) + 1

            data = response.choices[0].delta

            if is_firsts.get(index, True):
                assert (
                    data.role == "assistant"
                ), f"data.role was not 'assistant' for first chunk"
                is_firsts[index] = False
                continue

            if logprobs and not is_finished.get(index, False):
                assert response.choices[0].logprobs, f"logprobs was not returned"
                assert isinstance(
                    response.choices[0].logprobs.content[0].top_logprobs[0].token, str
                ), f"top_logprobs token was not a string"
                assert isinstance(
                    response.choices[0].logprobs.content[0].top_logprobs, list
                ), f"top_logprobs was not a list"
                ret_num_top_logprobs = len(
                    response.choices[0].logprobs.content[0].top_logprobs
                )
                assert (
                    ret_num_top_logprobs == logprobs
                ), f"{ret_num_top_logprobs} vs {logprobs}"

            assert (
                isinstance(data.content, str)
                or isinstance(data.reasoning_content, str)
                or (isinstance(data.tool_calls, list) and len(data.tool_calls) > 0)
                or response.choices[0].finish_reason
            )
            assert response.id
            assert response.created

        for index in [i for i in range(parallel_sample_num)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

        for index in range(parallel_sample_num):
            assert (
                index in finish_reason_counts
            ), f"No finish_reason found for index {index}"
            assert (
                finish_reason_counts[index] == 1
            ), f"Expected 1 finish_reason chunk for index {index}, got {finish_reason_counts[index]}"

    def test_completion(self):
        for echo in [False, True]:
            for logprobs in [None, 5]:
                for use_list_input in [True, False]:
                    for parallel_sample_num in [1, 2]:
                        for token_input in [False, True]:
                            self.run_completion(
                                echo,
                                logprobs,
                                use_list_input,
                                parallel_sample_num,
                                token_input,
                            )

    def test_completion_stream(self):
        for echo in [False, True]:
            for logprobs in [None, 5]:
                for use_list_input in [True, False]:
                    for parallel_sample_num in [1, 2]:
                        for token_input in [False, True]:
                            self.run_completion_stream(
                                echo,
                                logprobs,
                                use_list_input,
                                parallel_sample_num,
                                token_input,
                            )

    def test_chat_completion(self):
        for logprobs in [None, 5]:
            for parallel_sample_num in [1, 2]:
                self.run_chat_completion(logprobs, parallel_sample_num)

    def test_chat_completion_stream(self):
        for logprobs in [None, 5]:
            for parallel_sample_num in [1, 2]:
                self.run_chat_completion_stream(logprobs, parallel_sample_num)

    def test_regex(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        regex = (
            r"""\{\n"""
            + r"""   "name": "[\w]+",\n"""
            + r"""   "population": [\d]+\n"""
            + r"""\}"""
        )

        response = client.chat.completions.create(
            model="unknown",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=128,
            extra_body={"regex": regex},
        )
        text = response.choices[0].message.content

        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            print("JSONDecodeError", text)
            raise
        assert isinstance(js_obj["name"], str)
        assert isinstance(js_obj["population"], int)

    def test_penalty(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="unknown",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=32,
            frequency_penalty=1.0,
        )
        text = response.choices[0].message.content
        assert isinstance(text, str)

    def test_response_prefill(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="unknown",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": """
Extract the name, size, price, and color from this product description as a JSON object:

<description>
The SmartHome Mini is a compact smart home assistant available in black or white for only $49.99. At just 5 inches wide, it lets you control lights, thermostats, and other connected devices via voice or app—no matter where you place it in your home. This affordable little hub brings convenient hands-free control to your smart devices.
</description>
""",
                },
                {
                    "role": "assistant",
                    "content": "{\n",
                },
            ],
            temperature=0,
            extra_body={"continue_final_message": True},
        )

        assert (
            response.choices[0]
            .message.content.strip()
            .startswith('"name": "SmartHome Mini",')
        )
    def test_model_list(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        models = list(client.models.list())
        assert len(models) == 1
        assert isinstance(getattr(models[0], "max_model_len", None), int)

    def test_retrieve_model(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        retrieved_model = client.models.retrieve(self.model)
        self.assertEqual(retrieved_model.id, self.model)
        self.assertEqual(retrieved_model.root, self.model)

        with self.assertRaises(openai.NotFoundError):
            client.models.retrieve("non-existent-model")


if __name__ == "__main__":
    unittest.main()
