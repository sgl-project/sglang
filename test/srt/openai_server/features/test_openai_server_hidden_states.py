import json
import re
import time
import unittest
from abc import ABC

import numpy as np
import openai
import torch

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


class BaseTestOpenAIServerWithHiddenStates(ABC):

    @classmethod
    def setUpClass(cls):
        cls.return_hidden_states = [False, True]
        cls.use_list_input = [True, False]
        cls.parallel_sample_nums = [1, 2]

    def test_completion(self):
        for return_hidden_states in self.return_hidden_states:
            for use_list_input in self.use_list_input:
                for parallel_sample_num in self.parallel_sample_nums:
                    self.run_completion(
                        use_list_input,
                        parallel_sample_num,
                        return_hidden_states,
                    )

    def test_completion_stream(self):
        # parallel sampling and list input are not supported in streaming mode
        for return_hidden_states in self.return_hidden_states:
            for use_list_input in self.use_list_input:
                for parallel_sample_num in self.parallel_sample_nums:
                    self.run_completion_stream(
                        use_list_input,
                        parallel_sample_num,
                        return_hidden_states,
                    )

    def test_chat_completion(self):
        for return_hidden_states in self.return_hidden_states:
            for (
                parallel_sample_num
            ) in (
                self.parallel_sample_nums
            ):  # parallel sample num 2 breaks in the adapter with a 400 for EAGLE
                self.run_chat_completion(parallel_sample_num, return_hidden_states)

    def test_chat_completion_stream(self):
        for return_hidden_states in self.return_hidden_states:
            for (
                parallel_sample_num
            ) in (
                self.parallel_sample_nums
            ):  # parallel sample num > 1 breaks in the adapter with a 400 for EAGLE
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

        if use_list_input:
            prompt_arg = [prompt_input, prompt_input]
            num_choices = len(prompt_arg)
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

        for choice in response.choices:
            assert hasattr(choice, "hidden_states") == return_hidden_states
            if return_hidden_states:
                assert choice.hidden_states is not None, "hidden_states was None"

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

        hidden_states_list = []
        for response in generator:
            usage = response.usage
            for choice in response.choices:
                if hasattr(choice, "hidden_states"):
                    assert return_hidden_states
                    assert choice.hidden_states is not None
                    hidden_states_list.append(choice.hidden_states)

        if return_hidden_states:
            assert (
                len(hidden_states_list) == parallel_sample_num * num_choices
            ), f"Expected {parallel_sample_num * num_choices} hidden states, got {len(hidden_states_list)}"
        else:
            assert (
                hidden_states_list == []
            ), "hidden_states were returned and should not have been"

    def run_chat_completion(self, parallel_sample_num, return_hidden_states):
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

        for choice in response.choices:
            assert hasattr(choice, "hidden_states") == return_hidden_states
            if return_hidden_states:
                assert choice.hidden_states is not None, "hidden_states was None"

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
        hidden_states_list = []

        for response in generator:
            for choice in response.choices:
                if hasattr(choice.delta, "hidden_states"):
                    assert return_hidden_states
                    assert choice.delta.hidden_states is not None
                    hidden_states_list.append(choice.delta.hidden_states)

        if return_hidden_states:
            assert (
                len(hidden_states_list) == parallel_sample_num
            ), f"Expected {parallel_sample_num} hidden states, got {len(hidden_states_list)}"
        else:
            assert (
                hidden_states_list == []
            ), "hidden_states were returned and should not have been"


class TestOpenAIServerWithHiddenStatesEnabled(
    CustomTestCase, BaseTestOpenAIServerWithHiddenStates
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
            other_args=["--enable-return-hidden-states"],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
        cls.return_hidden_states = [False, True]
        cls.use_list_input = [True, False]
        cls.parallel_sample_nums = [1, 2]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestOpenAIServerWithHiddenStatesEnabledAndCUDAGraphDisabled(
    CustomTestCase, BaseTestOpenAIServerWithHiddenStates
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
            other_args=["--enable-return-hidden-states", "--disable-cuda-graph"],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
        cls.return_hidden_states = [False, True]
        cls.use_list_input = [True, False]
        cls.parallel_sample_nums = [1]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestOpenAIServerWithEAGLEAndHiddenStatesEnabled(
    CustomTestCase, BaseTestOpenAIServerWithHiddenStates
):
    @classmethod
    def setUpClass(cls):
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
                "--enable-return-hidden-states",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST)
        cls.return_hidden_states = [False, True]
        cls.use_list_input = [True, False]
        cls.parallel_sample_nums = [1]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestOpenAIServerWithEAGLE3AndHiddenStatesEnabled(
    CustomTestCase, BaseTestOpenAIServerWithHiddenStates
):
    @classmethod
    def setUpClass(cls):
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
                "--enable-return-hidden-states",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)
        cls.return_hidden_states = [False, True]
        cls.use_list_input = [True, False]
        cls.parallel_sample_nums = [1]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
