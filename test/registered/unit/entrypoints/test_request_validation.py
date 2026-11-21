# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU-only validation tests for OpenAI-compatible sampling request fields."""

import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    OPENAI_MAX_N,
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, suite="base-a-test-cpu")


def _completion(**kwargs):
    return CompletionRequest(model="test-model", prompt="Hello", **kwargs)


def _chat(**kwargs):
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        **kwargs,
    )


def _responses(**kwargs):
    return ResponsesRequest(input="Hello", **kwargs)


class TestSamplingFieldBounds(CustomTestCase):
    """Reject sampling values that the engine cannot apply."""

    def test_top_k_rejects_values_between_disable_and_one(self):
        requests = (
            ("completion", _completion),
            ("chat", _chat),
            ("responses", _responses),
        )
        for name, factory in requests:
            for top_k in (0, -2):
                with self.subTest(request=name, top_k=top_k):
                    with self.assertRaises(ValidationError) as ctx:
                        factory(top_k=top_k)
                    self.assertEqual(ctx.exception.errors()[0]["loc"], ("top_k",))

    def test_top_k_accepts_disable_and_positive_values(self):
        for top_k in (-1, 1, 50):
            with self.subTest(top_k=top_k):
                self.assertEqual(_completion(top_k=top_k).top_k, top_k)
                self.assertEqual(_chat(top_k=top_k).top_k, top_k)
                self.assertEqual(_responses(top_k=top_k).top_k, top_k)
        self.assertEqual(_completion().top_k, -1)
        self.assertIsNone(_chat().top_k)
        self.assertIsNone(_responses().top_k)

    def test_completion_logprobs_rejects_negative(self):
        with self.assertRaises(ValidationError) as ctx:
            _completion(logprobs=-1)
        self.assertEqual(ctx.exception.errors()[0]["loc"], ("logprobs",))
        for logprobs in (0, 5):
            with self.subTest(logprobs=logprobs):
                self.assertEqual(_completion(logprobs=logprobs).logprobs, logprobs)

    def test_responses_top_logprobs_rejects_negative(self):
        with self.assertRaises(ValidationError) as ctx:
            _responses(top_logprobs=-1)
        self.assertEqual(ctx.exception.errors()[0]["loc"], ("top_logprobs",))
        for value in (0, 20, 128):
            with self.subTest(value=value):
                self.assertEqual(_responses(top_logprobs=value).top_logprobs, value)

    def test_n_bounds_unchanged(self):
        for n in (1, OPENAI_MAX_N):
            with self.subTest(n=n):
                self.assertEqual(_completion(n=n).n, n)
                self.assertEqual(_chat(n=n).n, n)
        for n in (0, OPENAI_MAX_N + 1):
            with self.subTest(n=n), self.assertRaises(ValidationError):
                _completion(n=n)
            with self.subTest(n=n), self.assertRaises(ValidationError):
                _chat(n=n)


class TestSessionParamsType(CustomTestCase):
    """Wrong session_params types must fail at the request boundary."""

    def test_wrong_session_params_type_is_a_validation_error(self):
        for name, factory, value in (
            ("completion", _completion, "oops"),
            ("chat", _chat, [1, 2]),
        ):
            with self.subTest(request=name):
                with self.assertRaises(ValidationError) as ctx:
                    factory(session_params=value)
                self.assertEqual(ctx.exception.errors()[0]["type"], "dict_type")

    def test_mapping_session_params_is_accepted(self):
        params = {"id": "session-1"}
        self.assertEqual(_completion(session_params=params).session_params, params)
        self.assertEqual(_chat(session_params=params).session_params, params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
