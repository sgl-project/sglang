"""The OpenAI frontend backend sends one output-length parameter per request."""

import json
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace

import httpx
import openai

from sglang.lang.backend.openai import OpenAI, TokenUsage
from sglang.lang.ir import SglSamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _make_backend(model_name, is_chat_model, transport):
    # Skip OpenAI.__init__ (it downloads a tiktoken encoding); the request path
    # only reads these.
    backend = OpenAI.__new__(OpenAI)
    backend.client = openai.OpenAI(
        api_key="test-key",
        base_url="http://test.invalid/v1",
        http_client=httpx.Client(transport=transport),
    )
    backend.model_name = model_name
    backend.is_chat_model = is_chat_model
    backend.chat_prefix = ""
    backend.logit_bias_int = {}
    backend.token_usage = TokenUsage(0, 0)
    return backend


class TestOpenAIBackendOutputLimit(CustomTestCase):
    def _run(self, model_name, is_chat, call, stream=False, content="7"):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            if stream:
                chunk = {
                    "id": "x",
                    "object": "chat.completion.chunk" if is_chat else "text_completion",
                    "created": 0,
                    "model": model_name,
                    "choices": [],
                    "usage": USAGE,
                }
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                )
            choice = {"index": 0, "finish_reason": "stop"}
            if is_chat:
                choice["message"] = {"role": "assistant", "content": content}
            else:
                choice["text"] = content
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion" if is_chat else "text_completion",
                    "created": 0,
                    "model": model_name,
                    "choices": [choice],
                    "usage": USAGE,
                },
            )

        backend = _make_backend(model_name, is_chat, httpx.MockTransport(handler))
        s = SimpleNamespace(
            text_="hi",
            messages_=[{"role": "user", "content": "hi"}],
            num_api_spec_tokens=None,
        )
        call(backend, s)
        self.assertEqual(len(bodies), 1)
        return bodies[0]

    def _assert_limit(self, body, key, value=9):
        other = {"max_tokens", "max_completion_tokens"} - {key}
        self.assertEqual(body[key], value)
        for name in other:
            self.assertNotIn(name, body)

    def test_constrained_dtype_on_completions_model(self):
        for dtype in (int, str):
            with self.subTest(dtype=dtype):
                params = SglSamplingParams(max_new_tokens=9, dtype=dtype)
                body = self._run(
                    "gpt-3.5-turbo-instruct",
                    False,
                    lambda b, s, params=params: b.generate(s, params),
                )
                self._assert_limit(body, "max_tokens")

    def test_stream_on_completions_model(self):
        params = SglSamplingParams(max_new_tokens=9)
        body = self._run(
            "gpt-3.5-turbo-instruct",
            False,
            lambda b, s: list(b.generate_stream(s, params)),
            stream=True,
        )
        self._assert_limit(body, "max_tokens")

    def test_stream_on_chat_model(self):
        params = SglSamplingParams(max_new_tokens=9)
        body = self._run(
            "gpt-4o",
            True,
            lambda b, s: list(b.generate_stream(s, params)),
            stream=True,
        )
        self._assert_limit(body, "max_tokens")

    def test_stream_on_reasoning_model(self):
        params = SglSamplingParams(max_new_tokens=9)
        body = self._run(
            "o3-mini",
            True,
            lambda b, s: list(b.generate_stream(s, params)),
            stream=True,
        )
        self._assert_limit(body, "max_completion_tokens")

    def test_speculative_execution_on_chat_models(self):
        for model, key in (
            ("gpt-4o", "max_tokens"),
            ("o3-mini", "max_completion_tokens"),
        ):
            with self.subTest(model=model):

                def call(backend, s):
                    s.num_api_spec_tokens = 64
                    s.variables, s.meta_info = {}, {}
                    s.variable_event = defaultdict(threading.Event)
                    s.text_ = ""
                    backend.spec_kwargs, backend.spec_format = {}, []
                    backend.spec_max_num_tries = 3
                    # Two spec gens with different max_new_tokens must not clash.
                    backend.generate(
                        s,
                        SglSamplingParams(max_new_tokens=8, stop="\n"),
                        spec_var_name="a",
                    )
                    backend.generate(
                        s,
                        SglSamplingParams(max_new_tokens=16, stop="\n"),
                        spec_var_name="b",
                    )
                    backend.role_end_generate(s)

                body = self._run(model, True, call, content="7\n8")
                self._assert_limit(body, key, value=64)

    def test_generate_unchanged(self):
        params = SglSamplingParams(max_new_tokens=9)
        for model, is_chat, key in (
            ("gpt-3.5-turbo-instruct", False, "max_tokens"),
            ("gpt-4o", True, "max_tokens"),
            ("o3-mini", True, "max_completion_tokens"),
        ):
            with self.subTest(model=model):
                body = self._run(model, is_chat, lambda b, s: b.generate(s, params))
                self._assert_limit(body, key)


if __name__ == "__main__":
    unittest.main()
