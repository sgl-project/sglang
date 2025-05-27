import json
import re
import time
import unittest
from abc import ABC

import numpy as np
import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

HS_TEST_CASES = [True]
ENABLE_HS_ARGS = ["--enable-return-hidden-states"]  # ["--enable-return-hidden-states"]


class BaseTestOpenAIServerWithHiddenStates(ABC):

    def test_completion(self):
        for return_hidden_states in [False, True]:
            for use_list_input in [True, False]:
                for parallel_sample_num in [1]:
                    self.run_completion(
                        use_list_input,
                        parallel_sample_num,
                        return_hidden_states,
                    )

    def test_completion_stream(self):
        # parallel sampling and list input are not supported in streaming mode
        for return_hidden_states in [False, True]:
            for use_list_input in [True]:
                for parallel_sample_num in [1]:
                    self.run_completion_stream(
                        use_list_input,
                        parallel_sample_num,
                        return_hidden_states,
                    )

    def test_chat_completion(self):
        for return_hidden_states in [False, True]:
            for parallel_sample_num in [
                1
            ]:  # parallel sample num 2 breaks in the adapter with a 400 for EAGLE
                self.run_chat_completion(parallel_sample_num, return_hidden_states)

    def test_chat_completion_stream(self):
        for return_hidden_states in [False, True]:
            for parallel_sample_num in [
                1
            ]:  # parallel sample num > 1 breaks in the adapter with a 400 for EAGLE
                self.run_chat_completion_stream(
                    parallel_sample_num, return_hidden_states
                )

    def run_completion(
        self,
        use_list_input,
        parallel_sample_num,
        return_hidden_states,
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
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
            model=self.model,
            prompt=prompt_arg,
            temperature=0,
            max_tokens=32,
            n=parallel_sample_num,
            extra_body=dict(return_hidden_states=return_hidden_states),
        )

        assert len(response.choices) == num_choices * parallel_sample_num

        assert response.id
        assert response.created
        assert (
            response.usage.prompt_tokens == num_prompt_tokens
        ), f"{response.usage.prompt_tokens} vs {num_prompt_tokens}"
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

        if return_hidden_states:
            hidden_states = response.choices[0].hidden_states
            assert hidden_states is not None, "hidden_states was none"
            hidden_states = np.asarray(hidden_states)
            assert (
                len(hidden_states.shape) == 1
            ), f"hidden_states shape is not correct, was {hidden_states.shape}"
        else:
            assert not hasattr(
                response.choices[0], "hidden_states"
            ), "hidden_states was returned and should not have been"

    def run_completion_stream(
        self,
        use_list_input,
        parallel_sample_num,
        return_hidden_states,
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
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
            model=self.model,
            prompt=prompt_arg,
            temperature=0,
            max_tokens=32,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
            extra_body=dict(return_hidden_states=return_hidden_states),
        )

        is_firsts = {}
        hidden_states = None
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0, f"usage.prompt_tokens was zero"
                assert usage.completion_tokens > 0, f"usage.completion_tokens was zero"
                assert usage.total_tokens > 0, f"usage.total_tokens was zero"
                continue

            if (
                hasattr(response.choices[0], "hidden_states")
                and response.choices[0].hidden_states is not None
            ):
                hidden_states = response.choices[0].hidden_states
                continue

            index = response.choices[0].index
            is_first = is_firsts.get(index, True)

            if is_first:
                is_firsts[index] = False
            assert response.id, f"no id in response"
            assert response.created, f"no created in response"

        for index in [i for i in range(parallel_sample_num * num_choices)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

        if return_hidden_states:
            assert hidden_states is not None, "hidden_states is not returned"
            try:
                hidden_states = np.asarray(hidden_states)
            except Exception as e:
                raise Exception(f"Failed to convert hidden states to numpy array: {e}")
            assert (
                len(hidden_states.shape) == 1
            ), f"hidden_states shape is not correct, was {hidden_states.shape}"
        else:
            assert (
                hidden_states is None
            ), "hidden_states was returned and should not have been"

    def run_chat_completion(self, parallel_sample_num, return_hidden_states):
        print(
            f"run_chat_completion: parallel_sample_num={parallel_sample_num}, return_hidden_states={return_hidden_states}"
        )
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                },
            ],
            temperature=0,
            n=parallel_sample_num,
            extra_body=dict(return_hidden_states=return_hidden_states),
        )

        assert len(response.choices) == parallel_sample_num
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

        if return_hidden_states:
            hidden_states = response.choices[0].hidden_states
            assert hidden_states is not None, "hidden_states is not returned"
            hidden_states = np.asarray(hidden_states)
            assert (
                len(hidden_states.shape) == 1
            ), f"hidden_states shape is not correct, was {hidden_states.shape}"
        else:
            assert not hasattr(
                response.choices[0], "hidden_states"
            ), "hidden_states was returned and should not have been"

    def run_chat_completion_stream(
        self, parallel_sample_num=1, return_hidden_states=False
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        generator = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
            extra_body=dict(return_hidden_states=return_hidden_states),
        )

        is_firsts = {}
        hidden_states = None
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0, f"usage.prompt_tokens was zero"
                assert usage.completion_tokens > 0, f"usage.completion_tokens was zero"
                assert usage.total_tokens > 0, f"usage.total_tokens was zero"
                continue

            if hasattr(response.choices[0].delta, "hidden_states"):
                hidden_states = response.choices[0].delta.hidden_states
                continue

            index = response.choices[0].index
            data = response.choices[0].delta

            if is_firsts.get(index, True):
                assert (
                    data.role == "assistant"
                ), f"data.role was not 'assistant' for first chunk"
                is_firsts[index] = False
                continue

            assert (
                isinstance(data.content, str)
                or isinstance(data.reasoning_content, str)
                or len(data.tool_calls) > 0
                or response.choices[0].finish_reason
            )
            assert response.id
            assert response.created

        for index in [i for i in range(parallel_sample_num)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

        if return_hidden_states:
            assert hidden_states is not None, "hidden_states is not returned"
            try:
                hidden_states = np.asarray(hidden_states)
            except Exception as e:
                raise Exception(f"Failed to convert hidden states to numpy array: {e}")
            assert (
                len(hidden_states.shape) == 1
            ), f"hidden_states shape is not correct, was {hidden_states.shape}"
        else:
            assert (
                hidden_states is None
            ), "hidden_states was returned and should not have been"


class TestOpenAIServerWithHiddenStatesEnabled(
    unittest.TestCase, BaseTestOpenAIServerWithHiddenStates
):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=ENABLE_HS_ARGS,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestOpenAIServerWithEAGLEAndHiddenStatesEnabled(
    unittest.TestCase, BaseTestOpenAIServerWithHiddenStates
):
    @classmethod
    def setUpClass(cls):
        print("STARTING EAGLE WITH HIDDEN STATES TEST SUITE")
        cls.model = DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.speculative_draft_model = DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST
        cls.speculative_algorithm = "EAGLE"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *ENABLE_HS_ARGS,
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestOpenAIServerWithEAGLE3AndHiddenStatesEnabled(
    unittest.TestCase, BaseTestOpenAIServerWithHiddenStates
):
    @classmethod
    def setUpClass(cls):
        print("STARTING EAGLE 3 WITH HIDDEN STATES TEST SUITE")
        cls.model = "meta-llama/Llama-3.1-8B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.speculative_algorithm = "EAGLE3"
        cls.speculative_draft_model = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *ENABLE_HS_ARGS,
                "--speculative-algorithm",
                cls.speculative_algorithm,
                "--speculative-draft-model-path",
                cls.speculative_draft_model,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                16,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--dtype",
                "float16",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
