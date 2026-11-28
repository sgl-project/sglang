import importlib.util
import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

# Registered on the CPU suite but skipped wherever mlx is absent; runs for real
# only on Apple Silicon. Also registered under stage-b-e2e-mlx, which the
# macOS CI lane (pr-test-mlx.yml) only dispatches via a gated workflow_dispatch.
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-b-e2e-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None

# llama architecture (LlamaForCausalLM), served on the MLX backend through
# mlx_lm's own llama implementation. The other models_e2e cases are all
# mixture-of-experts (qwen2_moe, qwen3_moe, gpt_oss), so the dense decoder-only
# MLX path is otherwise unguarded end to end.
#
# The 3B 4-bit repo is also the only models_e2e default that fits a base-config
# 16 GB Apple Silicon machine; the MoE defaults need roughly 8-17 GB of weights.
# Override with SGLANG_MLX_TEST_MODEL to point at a local copy, e.g.
#   SGLANG_MLX_TEST_MODEL=models/Llama-3.2-3B-Instruct-4bit
MODEL_PATH = os.environ.get(
    "SGLANG_MLX_TEST_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"
)

# mem-fraction is tuned for a 16 GB Apple Silicon machine, which is the point of
# picking a small dense model here.
MEM_FRACTION_STATIC = os.environ.get("SGLANG_MLX_TEST_MEM_FRACTION", "0.7")

# Llama 3 chat control tokens. These are template scaffolding: if one reaches the
# response body, the chat template or the detokenizer is leaking it.
_CHAT_CONTROL_TOKENS = (
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
)

_REQUEST_TIMEOUT = 120


@unittest.skipUnless(_HAS_MLX, "requires mlx (Apple Silicon only)")
class TestLlamaDenseMlxCorrectness(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    CustomTestCase,
):
    served_model_name = MODEL_PATH

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "1",
                "--disable-radix-cache",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                MEM_FRACTION_STATIC,
                "--max-running-requests",
                "1",
                "--context-length",
                "2048",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _chat(self, messages, max_tokens=32, stream=False):
        payload = {
            "model": MODEL_PATH,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if not stream:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={**payload, "stream": True},
            timeout=_REQUEST_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()
        chunks = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload_line = raw[len("data:") :].strip()
            if payload_line == "[DONE]":
                break
            delta = json.loads(payload_line)["choices"][0]["delta"]
            chunks.append(delta.get("content") or "")
        return "".join(chunks)

    def test_streaming_matches_non_streaming(self):
        """Streamed deltas must concatenate to the non-streamed body.

        Guards the incremental detokenizer against dropping, duplicating or
        re-splitting text at a chunk boundary -- a failure mode no non-streaming
        assertion can see, and one no other models_e2e case covers.
        """
        messages = [{"role": "user", "content": "Name two oceans, comma separated."}]
        streamed = self._chat(messages, stream=True)
        non_streamed = self._chat(messages, stream=False)
        self.assertGreater(len(streamed), 0)
        self.assertEqual(streamed, non_streamed)

    def test_chat_control_tokens_absent(self):
        """Template scaffolding must not survive into the response body."""
        text = self._chat(
            [{"role": "user", "content": "What is the capital of France? One word."}],
            max_tokens=8,
        )
        for token in _CHAT_CONTROL_TOKENS:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
