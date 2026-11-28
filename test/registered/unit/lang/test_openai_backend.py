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

import unittest
from types import SimpleNamespace

from sglang.lang.backend.openai import OpenAI, TokenUsage
from sglang.lang.ir import SglSamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-a", runner_config="1-gpu-small")


def _make_fake_backend(is_chat_model: bool = True) -> OpenAI:
    backend = object.__new__(OpenAI)
    backend.is_chat_model = is_chat_model
    backend.model_name = "gpt-4o-mini"
    backend.token_usage = TokenUsage(0, 0)
    backend.spec_kwargs = {}
    backend.spec_format = []
    backend.spec_max_num_tries = 3
    return backend


class TestOpenAIBackend(CustomTestCase):
    def test_spec_pattern_match_with_none_and_list_stop(self):
        backend = _make_fake_backend(is_chat_model=True)
        backend.spec_fill("Name: ")
        backend._prepare_spec_execution(
            SglSamplingParams(stop=[",", "\n"]),
            num_api_spec_tokens=32,
            spec_var_name="first",
        )
        backend.spec_fill(", Age: ")
        backend._prepare_spec_execution(
            SglSamplingParams(stop=None),
            num_api_spec_tokens=32,
            spec_var_name="age",
        )

        self.assertTrue(backend.spec_pattern_match("Name: Alice, Age: 30"))
        self.assertEqual(backend.spec_format[1]["text"], "Alice")
        self.assertEqual(backend.spec_format[3]["text"], "30")

    def test_spec_pattern_match_retry_does_not_freeze_failed_capture(self):
        backend = _make_fake_backend(is_chat_model=True)
        backend._prepare_spec_execution(
            SglSamplingParams(stop="|"),
            num_api_spec_tokens=32,
            spec_var_name="a",
        )
        backend.spec_fill("|!")

        self.assertFalse(backend.spec_pattern_match("first|?"))
        self.assertEqual(backend.spec_format[0]["text"], "")

        self.assertTrue(backend.spec_pattern_match("second|!"))
        self.assertEqual(backend.spec_format[0]["text"], "second")

    def test_select_accumulates_completion_tokens(self):
        backend = _make_fake_backend(is_chat_model=False)
        backend.token_usage = TokenUsage(prompt_tokens=5, completion_tokens=3)
        backend.tokenizer = SimpleNamespace(
            encode=lambda s: [ord(c) for c in s],
        )
        responses = iter(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(text="a")],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1),
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(text="c")],
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=1),
                ),
            ]
        )
        backend.client = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: next(responses))
        )
        s = SimpleNamespace(text_="p")

        decision = backend.select(
            s, choices=["ab", "ac"], temperature=0.0, choices_method=None
        )
        self.assertEqual(decision.decision, "ac")
        self.assertEqual(backend.token_usage.prompt_tokens, 5 + 10 + 11)
        self.assertEqual(backend.token_usage.completion_tokens, 3 + 1 + 1)


if __name__ == "__main__":
    unittest.main()
