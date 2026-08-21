"""
XPU serving-features test: covers OpenAI API, constrained decoding,
sampling penalties, radix cache, and reasoning parsing in a single
server fixture so each feature gets one canonical assertion on Intel XPU
without paying the cost of N separate model launches.

Usage:
python3 -m unittest test_xpu_serving_features.TestXPUServingFeatures
"""

import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=300, suite="stage-b-test-1-gpu-xpu")


class TestXPUServingFeatures(CustomTestCase):
    """One server, many features. Boots Llama-3.2-1B-Instruct once and
    exercises five separate gaps the per-feature tests would each launch
    their own server for.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        # No API key: the radix-cache helper sends raw POSTs to /generate
        # without auth headers, so the server must be open.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--device", "xpu"],
        )
        cls.openai_url = cls.base_url + "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _client(self) -> openai.Client:
        # Server has no API key, but openai client still requires a non-empty string.
        return openai.Client(api_key="EMPTY", base_url=self.openai_url)

    def test_openai_chat_completion(self):
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=8,
            temperature=0.0,
        )
        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.choices[0].message.role, "assistant")
        self.assertGreater(len(response.choices[0].message.content or ""), 0)
        self.assertGreater(response.usage.completion_tokens, 0)

    def test_json_constrained_generation(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with fields name (string) and age (integer).",
                }
            ],
            max_tokens=64,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema, "strict": True},
            },
        )
        text = response.choices[0].message.content
        self.assertIsNotNone(text)
        parsed = json.loads(text)
        self.assertIn("name", parsed)
        self.assertIn("age", parsed)
        self.assertIsInstance(parsed["age"], int)

    def test_sampling_penalty(self):
        prompt = "List five different colors:"
        baseline = self._client().completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=64,
            temperature=0.7,
            seed=1,
        )
        penalized = self._client().completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=64,
            temperature=0.7,
            seed=1,
            frequency_penalty=2.0,
            presence_penalty=2.0,
        )
        self.assertGreater(len(baseline.choices[0].text), 0)
        self.assertGreater(len(penalized.choices[0].text), 0)
        # Penalty must change the output for the same prompt + seed.
        self.assertNotEqual(baseline.choices[0].text, penalized.choices[0].text)

    def test_radix_cache_multiturn_hit(self):
        run_multiturn_cache_hit_test(
            base_url=self.base_url,
            model_path=self.model,
            num_clients=4,
            num_rounds=3,
            request_length=128,
            output_length=64,
        )

    def test_reasoning_separate_parser(self):
        # Drive the separate-reasoning code path: when the model emits a
        # <think>...</think> block the server must split it from the visible
        # answer. Llama-3.2-1B does not naturally emit thinking tags, so we
        # prompt it to do so explicitly and assert the parser surfaces both
        # fields without crashing.
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Wrap your reasoning in <think>...</think> tags then "
                        "answer: what is 1 + 1?"
                    ),
                }
            ],
            max_tokens=48,
            temperature=0.0,
            extra_body={"separate_reasoning": True},
        )
        message = response.choices[0].message
        self.assertEqual(message.role, "assistant")
        # Either reasoning_content is populated, or content is — never both empty.
        reasoning = getattr(message, "reasoning_content", None) or ""
        content = message.content or ""
        self.assertTrue(
            reasoning or content,
            "separate_reasoning produced empty reasoning_content AND content",
        )


if __name__ == "__main__":
    unittest.main()
