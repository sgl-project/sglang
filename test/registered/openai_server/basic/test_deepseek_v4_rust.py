"""Real-checkpoint parity through both HTTP frontends, tokenization and inference.

Run with: python test/registered/openai_server/basic/test_deepseek_v4_rust.py
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=360, stage="base-c", runner_config="8-gpu-h200")


def chat_cases():
    user = {"role": "user", "content": "What is the capital of France?"}
    yield "default", {"messages": [user]}
    yield "system", {"messages": [{"role": "system", "content": "Be concise."}, user]}
    yield "text_parts", {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital"},
                    {"type": "text", "text": "of France?"},
                ],
            }
        ]
    }
    for thinking in (False, True):
        for effort in (None, "low", "high", "max", "medium"):
            yield f"thinking={thinking},effort={effort}", {
                "messages": [user],
                "chat_template_kwargs": {"thinking": thinking},
                "reasoning_effort": effort,
            }
    yield "reasoning_history", {
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
            {
                "role": "assistant",
                "content": "2",
                "reasoning_content": "Add one and one.",
            },
            user,
        ],
        "chat_template_kwargs": {"thinking": True},
    }
    yield "effort_kwarg_overrides_top_level", {
        "messages": [user],
        "reasoning_effort": "low",
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
    }
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    yield "tools", {"messages": [user], "tools": [tool], "tool_choice": "none"}
    yield "tool_history", {
        "tools": [tool],
        "tool_choice": "none",
        "chat_template_kwargs": {"thinking": True},
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Look up the weather.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny."},
            {"role": "user", "content": "Summarize the result."},
        ],
    }


class TestDeepSeekV4RustParity(CustomTestCase):
    def collect(self, model, rust):
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={
                "SGLANG_RUST_SERVER": str(int(rust)),
                "SGLANG_DEFAULT_THINKING": "0",
                "SGLANG_DSV4_REASONING_EFFORT": "",
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
            },
            other_args=[
                "--tp",
                "2",
                "--moe-runner-backend",
                "marlin",
                "--context-length",
                "8192",
                "--chunked-prefill-size",
                "512",
                "--max-running-requests",
                "1",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--disable-radix-cache",
                "--random-seed",
                "42",
            ],
        )
        try:
            results = {}
            for name, body in chat_cases():
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/v1/chat/completions",
                    json={
                        "model": model,
                        "temperature": 0,
                        "max_tokens": 8,
                        "logprobs": True,
                        "top_logprobs": 5,
                        **body,
                    },
                    timeout=180,
                )
                self.assertEqual(response.status_code, 200, (name, response.text))
                results[name] = response.json()
            return results
        finally:
            kill_process_tree(process.pid)

    def test_chat_parity(self):
        for model in (
            "deepseek-ai/DeepSeek-V4-Flash",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ):
            python = self.collect(model, rust=False)
            rust = self.collect(model, rust=True)
            for name in python:
                with self.subTest(model=model, case=name):
                    reference, actual = python[name], rust[name]
                    self.assertEqual(
                        actual["usage"]["prompt_tokens"],
                        reference["usage"]["prompt_tokens"],
                    )
                    reference, actual = reference["choices"][0], actual["choices"][0]
                    self.assertEqual(
                        actual["message"]["content"], reference["message"]["content"]
                    )
                    self.assertEqual(
                        actual["finish_reason"], reference["finish_reason"]
                    )
                    reference = reference["logprobs"]["content"]
                    actual = actual["logprobs"]["content"]
                    self.assertTrue(reference)
                    self.assertEqual(len(actual), len(reference))
                    for expected, observed in zip(reference, actual, strict=True):
                        self.assertEqual(observed["token"], expected["token"])
                        self.assertAlmostEqual(
                            observed["logprob"], expected["logprob"], delta=1e-5
                        )
                        self.assertEqual(
                            {v["token"] for v in observed["top_logprobs"]},
                            {v["token"] for v in expected["top_logprobs"]},
                        )
                        expected_top = {
                            v["token"]: v["logprob"] for v in expected["top_logprobs"]
                        }
                        for entry in observed["top_logprobs"]:
                            self.assertAlmostEqual(
                                entry["logprob"],
                                expected_top[entry["token"]],
                                delta=1e-5,
                            )


if __name__ == "__main__":
    unittest.main()
