import importlib.util
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
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
register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-b-e2e-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None

# qwen2_moe architecture (Qwen2MoeForCausalLM), served on the MLX backend.
# The model runs through mlx_lm's own qwen2_moe implementation; the SGLang MLX
# backend does not require any srt/models file for it. This test is a black-box
# correctness guard for the served model.
#
# Default is the MLX-community 4-bit repo so the test is portable. Override with
# SGLANG_MLX_TEST_MODEL to point at a local copy, e.g.
#   SGLANG_MLX_TEST_MODEL=models/Qwen1.5-MoE-A2.7B-Chat-4bit
MODEL_PATH = os.environ.get(
    "SGLANG_MLX_TEST_MODEL", "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit"
)

# mem-fraction is tuned conservatively for a 24 GB Apple Silicon machine.
MEM_FRACTION_STATIC = os.environ.get("SGLANG_MLX_TEST_MEM_FRACTION", "0.7")


@unittest.skipUnless(_HAS_MLX, "requires mlx (Apple Silicon only)")
class TestQwen2MoeMlxCorrectness(CustomTestCase):
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
                "--trust-remote-code",
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

    def _chat(self, messages, max_tokens=32, temperature=0):
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": MODEL_PATH,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def test_basic_generation_nonempty(self):
        text = self._chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Say hello briefly."},
            ],
            max_tokens=16,
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_simple_arithmetic(self):
        text = self._chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2+2? Reply with just the number."},
            ],
            max_tokens=8,
        )
        self.assertIn("4", text)

    def test_simple_fact(self):
        text = self._chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is the capital of France? One word."},
            ],
            max_tokens=8,
        )
        self.assertIn("Paris", text)


if __name__ == "__main__":
    unittest.main()
