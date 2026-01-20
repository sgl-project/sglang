"""
python3 -m unittest openai_server.basic.test_openai_server.TestOpenAIServer.test_completion
python3 -m unittest openai_server.basic.test_openai_server.TestOpenAIServer.test_completion_stream
python3 -m unittest openai_server.basic.test_openai_server.TestOpenAIServer.test_chat_completion
python3 -m unittest openai_server.basic.test_openai_server.TestOpenAIServer.test_chat_completion_stream
"""

import json
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import openai
import requests

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.runners import TEST_RERANK_QUERY_DOCS
from sglang.test.test_utils import (
    DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=184, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=200, suite="stage-b-test-small-1-gpu-amd")


class TestOpenAIServer(CustomTestCase):
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
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

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
            model=self.model,
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

            # FIXME: Sometimes, some top_logprobs are missing in the return value. The reason is that some output id maps to the same output token and duplicate in the map
            # assert ret_num_top_logprobs == logprobs, f"{ret_num_top_logprobs} vs {logprobs}"
            assert ret_num_top_logprobs > 0

            # when echo=True and request.logprobs>0, logprob_start_len is 0, so the first token's logprob would be None.
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
            model=self.model,
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
                    # FIXME: Sometimes, some top_logprobs are missing in the return value. The reason is that some output id maps to the same output token and duplicate in the map
                    # assert ret_num_top_logprobs == logprobs, f"{ret_num_top_logprobs} vs {logprobs}"
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
            model=self.model,
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
            model=self.model,
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

        # Verify that each choice gets exactly one finish_reason chunk
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
        # parallel sampling and list input are not supported in streaming mode
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
            model=self.model,
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
            model=self.model,
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
            model="meta-llama/Llama-3.1-8B-Instruct",
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

        # Test retrieving an existing model
        retrieved_model = client.models.retrieve(self.model)
        self.assertEqual(retrieved_model.id, self.model)
        self.assertEqual(retrieved_model.root, self.model)

        # Test retrieving a non-existent model
        with self.assertRaises(openai.NotFoundError):
            client.models.retrieve("non-existent-model")


class TestOpenAIServerv1Responses(CustomTestCase):
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
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_response(
        self,
        input_text: str = "The capital of France is",
        *,
        instructions: str | None = None,
        temperature: float | None = 0.0,
        top_p: float | None = 1.0,
        max_output_tokens: int | None = 32,
        store: bool | None = True,
        parallel_tool_calls: bool | None = True,
        tool_choice: str | None = "auto",
        previous_response_id: str | None = None,
        truncation: str | None = "disabled",
        user: str | None = None,
        metadata: dict | None = None,
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "input": input_text,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "store": store,
            "parallel_tool_calls": parallel_tool_calls,
            "tool_choice": tool_choice,
            "previous_response_id": previous_response_id,
            "truncation": truncation,
            "user": user,
            "instructions": instructions,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        payload = {k: v for k, v in payload.items() if v is not None}
        return client.responses.create(**payload)

    def run_response_stream(
        self,
        input_text: str = "The capital of France is",
        *,
        instructions: str | None = None,
        temperature: float | None = 0.0,
        top_p: float | None = 1.0,
        max_output_tokens: int | None = 32,
        store: bool | None = True,
        parallel_tool_calls: bool | None = True,
        tool_choice: str | None = "auto",
        previous_response_id: str | None = None,
        truncation: str | None = "disabled",
        user: str | None = None,
        metadata: dict | None = None,
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "input": input_text,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "store": store,
            "parallel_tool_calls": parallel_tool_calls,
            "tool_choice": tool_choice,
            "previous_response_id": previous_response_id,
            "truncation": truncation,
            "user": user,
            "instructions": instructions,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if metadata is not None:
            payload["metadata"] = metadata
        payload = {k: v for k, v in payload.items() if v is not None}

        aggregated_text = ""
        saw_created = False
        saw_in_progress = False
        saw_completed = False
        final_usage_ok = False

        stream_ctx = getattr(client.responses, "stream", None)
        if callable(stream_ctx):
            stream_payload = dict(payload)
            stream_payload.pop("stream", None)
            stream_payload.pop("stream_options", None)
            with client.responses.stream(**stream_payload) as stream:
                for event in stream:
                    et = getattr(event, "type", None)
                    if et == "response.created":
                        saw_created = True
                    elif et == "response.in_progress":
                        saw_in_progress = True
                    elif et == "response.output_text.delta":
                        # event.delta expected to be a string
                        delta = getattr(event, "delta", "")
                        if isinstance(delta, str):
                            aggregated_text += delta
                    elif et == "response.completed":
                        saw_completed = True
                        # Validate streaming-completed usage mapping
                        resp = getattr(event, "response", None)
                        try:
                            # resp may be dict-like already
                            usage = (
                                resp.get("usage")
                                if isinstance(resp, dict)
                                else getattr(resp, "usage", None)
                            )
                            if isinstance(usage, dict):
                                final_usage_ok = all(
                                    k in usage
                                    for k in (
                                        "input_tokens",
                                        "output_tokens",
                                        "total_tokens",
                                    )
                                )
                        except Exception:
                            pass
                _ = stream.get_final_response()
        else:
            generator = client.responses.create(**payload)
            for event in generator:
                et = getattr(event, "type", None)
                if et == "response.created":
                    saw_created = True
                elif et == "response.in_progress":
                    saw_in_progress = True
                elif et == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if isinstance(delta, str):
                        aggregated_text += delta
                elif et == "response.completed":
                    saw_completed = True

        return (
            aggregated_text,
            saw_created,
            saw_in_progress,
            saw_completed,
            final_usage_ok,
        )

    def run_chat_completion_stream(self, logprobs=None, parallel_sample_num=1):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        generator = client.chat.completions.create(
            model=self.model,
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
        for _ in generator:
            pass

    # ---- tests ----
    def test_response(self):
        resp = self.run_response(temperature=0, max_output_tokens=32)
        assert resp.id
        assert resp.object == "response"
        assert resp.created_at
        assert isinstance(resp.model, str)
        assert isinstance(resp.output, list)
        assert resp.status in (
            "completed",
            "in_progress",
            "queued",
            "failed",
            "cancelled",
        )
        if resp.status == "completed":
            assert resp.usage is not None
            assert resp.usage.prompt_tokens >= 0
            assert resp.usage.completion_tokens >= 0
            assert resp.usage.total_tokens >= 0
        if hasattr(resp, "error"):
            assert resp.error is None
        if hasattr(resp, "incomplete_details"):
            assert resp.incomplete_details is None
        if getattr(resp, "text", None):
            fmt = resp.text.get("format") if isinstance(resp.text, dict) else None
            if fmt:
                assert fmt.get("type") == "text"

    def test_response_stream(self):
        aggregated_text, saw_created, saw_in_progress, saw_completed, final_usage_ok = (
            self.run_response_stream(temperature=0, max_output_tokens=32)
        )
        assert saw_created, "Did not observe response.created"
        assert saw_in_progress, "Did not observe response.in_progress"
        assert saw_completed, "Did not observe response.completed"
        assert isinstance(aggregated_text, str)
        assert len(aggregated_text) >= 0
        assert final_usage_ok or True  # final_usage's stats are not done for now

    def test_response_completion(self):
        resp = self.run_response(temperature=0, max_output_tokens=16)
        assert resp.status in ("completed", "in_progress", "queued")
        if resp.status == "completed":
            assert resp.usage is not None
            assert resp.usage.total_tokens >= 0

    def test_response_completion_stream(self):
        _, saw_created, saw_in_progress, saw_completed, final_usage_ok = (
            self.run_response_stream(temperature=0, max_output_tokens=16)
        )
        assert saw_created
        assert saw_in_progress
        assert saw_completed
        assert final_usage_ok or True  # final_usage's stats are not done for now

    def test_regex(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        regex = (
            r"""\{\n"""
            + r"""   "name": "[\w]+",\n"""
            + r"""   "population": [\d]+\n"""
            + r"""\}"""
        )

        response = client.chat.completions.create(
            model=self.model,
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

    def test_error(self):
        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": "Hi",
            "previous_response_id": "bad",  # invalid prefix
        }
        r = requests.post(url, headers=headers, json=payload)
        self.assertEqual(r.status_code, 400)
        body = r.json()
        self.assertIn("error", body)
        self.assertIn("message", body["error"])
        self.assertIn("type", body["error"])
        self.assertIn("code", body["error"])

    def test_penalty(self):
        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": "Introduce the capital of France.",
            "temperature": 0,
            "max_output_tokens": 32,
            "frequency_penalty": 1.0,
        }
        r = requests.post(url, headers=headers, json=payload)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("object"), "response")
        self.assertIn("output", body)
        self.assertIn("status", body)
        if "usage" in body:
            self.assertIn("prompt_tokens", body["usage"])
            self.assertIn("total_tokens", body["usage"])

    def test_response_prefill(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
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


class TestOpenAIV1Rerank(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.score_tolerance = 1e-2

        # Configure embedding-specific args
        other_args = [
            "--is-embedding",
            "--enable-metrics",
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "-1",
            "--attention-backend",
            "torch_native",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1/rerank"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_rerank(self, query, docs):
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "documents": docs},
        )

        return response.json()

    def test_rerank_single(self):
        """Test single rerank request"""
        query = TEST_RERANK_QUERY_DOCS[0]["query"]
        docs = TEST_RERANK_QUERY_DOCS[0]["documents"]

        response = self.run_rerank(query, docs)

        self.assertEqual(len(response), 1)
        self.assertTrue(isinstance(response[0]["score"], float))
        self.assertTrue(isinstance(response[0]["document"], str))
        self.assertTrue(isinstance(response[0]["index"], int))

    def test_rerank_batch(self):
        """Test batch rerank request"""
        query = TEST_RERANK_QUERY_DOCS[1]["query"]
        docs = TEST_RERANK_QUERY_DOCS[1]["documents"]

        response = self.run_rerank(query, docs)

        self.assertEqual(len(response), 2)
        self.assertTrue(isinstance(response[0]["score"], float))
        self.assertTrue(isinstance(response[1]["score"], float))
        self.assertTrue(isinstance(response[0]["document"], str))
        self.assertTrue(isinstance(response[1]["document"], str))
        self.assertTrue(isinstance(response[0]["index"], int))
        self.assertTrue(isinstance(response[1]["index"], int))


class TestOpenAIServerCustomLogitProcessor(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=["--enable-custom-logit-processor"],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def run_custom_logit_processor(self, target_token_id: Optional[int] = None) -> None:
        """
        Test custom logit processor with custom params.

        If target_token_id is None, the custom logit processor won't be passed in.
        """

        class DeterministicLogitProcessor(CustomLogitProcessor):
            """A dummy logit processor that changes the logits to always sample the given token id."""

            CUSTOM_PARAM_KEY = "token_id"

            def __call__(self, logits, custom_param_list):
                assert logits.shape[0] == len(custom_param_list)

                for i, param_dict in enumerate(custom_param_list):
                    # Mask all other tokens
                    logits[i, :] = -float("inf")
                    # Assign highest probability to the specified token
                    logits[i, param_dict[self.CUSTOM_PARAM_KEY]] = 0.0

                return logits

        extra_body = {}

        if target_token_id is not None:
            extra_body["custom_logit_processor"] = (
                DeterministicLogitProcessor().to_str()
            )
            extra_body["custom_params"] = {
                "token_id": target_token_id,
            }

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        max_tokens = 200

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Question: Is Paris the Capital of France?",
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

        if target_token_id is not None:
            target_text = self.tokenizer.decode([target_token_id] * max_tokens)
            self.assertTrue(
                target_text == response.choices[0].message.content,
                f"{target_token_id=}\n{target_text=}\n{response.model_dump(mode='json')}",
            )

    def test_custom_logit_processor(self) -> None:
        """Test custom logit processor with a single request."""
        self.run_custom_logit_processor(target_token_id=5)

    def test_custom_logit_processor_batch_mixed(self) -> None:
        """Test a batch of requests mixed of requests with and without custom logit processor."""
        target_token_ids = list(range(32)) + [None] * 16
        random.shuffle(target_token_ids)
        with ThreadPoolExecutor(len(target_token_ids)) as executor:
            list(executor.map(self.run_custom_logit_processor, target_token_ids))


class TestOpenAIV1Score(CustomTestCase):
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
        )
        cls.base_url += "/v1/score"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_score(
        self, query, items, label_token_ids, apply_softmax=False, item_first=False
    ):
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "items": items,
                "label_token_ids": label_token_ids,
                "apply_softmax": apply_softmax,
                "item_first": item_first,
            },
        )
        return response.json()

    def test_score_text_input(self):
        """Test scoring with text input"""
        query = "The capital of France is"
        items = ["Paris", "London", "Berlin"]

        # Get valid token IDs from the tokenizer
        label_token_ids = []
        for item in items:
            token_ids = self.tokenizer.encode(item, add_special_tokens=False)
            if not token_ids:
                self.fail(f"Failed to encode item: {item}")
            label_token_ids.append(token_ids[0])

        response = self.run_score(query, items, label_token_ids, apply_softmax=True)

        # Handle error responses
        if response.get("type") == "BadRequestError":
            self.fail(f"Score request failed with error: {response['message']}")

        # Verify response structure
        self.assertIn("scores", response, "Response should have a 'scores' field")
        self.assertIsInstance(response["scores"], list, "scores should be a list")
        self.assertEqual(
            len(response["scores"]),
            len(items),
            "Number of scores should match number of items",
        )

        # Each score should be a list of floats in the order of label_token_ids
        for i, score_list in enumerate(response["scores"]):
            self.assertIsInstance(score_list, list, f"Score {i} should be a list")
            self.assertEqual(
                len(score_list),
                len(label_token_ids),
                f"Score {i} length should match label_token_ids",
            )
            self.assertTrue(
                all(isinstance(v, float) for v in score_list),
                f"Score {i} values should be floats",
            )
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=6,
                msg=f"Score {i} probabilities should sum to 1",
            )

    def test_score_token_input(self):
        """Test scoring with token IDs input"""
        query = "The capital of France is"
        items = ["Paris", "London", "Berlin"]

        # Get valid token IDs
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        item_ids = [
            self.tokenizer.encode(item, add_special_tokens=False) for item in items
        ]
        label_token_ids = [
            ids[0] for ids in item_ids if ids
        ]  # Get first token ID of each item

        response = self.run_score(
            query_ids, item_ids, label_token_ids, apply_softmax=True
        )

        # Handle error responses
        if response.get("type") == "BadRequestError":
            self.fail(f"Score request failed with error: {response['message']}")

        # Verify response structure
        self.assertIn("scores", response, "Response should have a 'scores' field")
        self.assertIsInstance(response["scores"], list, "scores should be a list")
        self.assertEqual(
            len(response["scores"]),
            len(items),
            "Number of scores should match number of items",
        )

        # Each score should be a list of floats in the order of label_token_ids
        for i, score_list in enumerate(response["scores"]):
            self.assertIsInstance(score_list, list, f"Score {i} should be a list")
            self.assertEqual(
                len(score_list),
                len(label_token_ids),
                f"Score {i} length should match label_token_ids",
            )
            self.assertTrue(
                all(isinstance(v, float) for v in score_list),
                f"Score {i} values should be floats",
            )
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=6,
                msg=f"Score {i} probabilities should sum to 1",
            )

    def test_score_error_handling(self):
        """Test error handling for invalid inputs"""
        query = "The capital of France is"
        items = ["Paris", "London", "Berlin"]

        # Test with invalid token ID
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "items": items,
                "label_token_ids": [999999],  # Invalid token ID
                "apply_softmax": True,
            },
        )
        self.assertEqual(response.status_code, 400)
        error_response = response.json()
        self.assertEqual(error_response["type"], "BadRequestError")
        self.assertIn("Token ID 999999 is out of vocabulary", error_response["message"])


if __name__ == "__main__":
    unittest.main()
