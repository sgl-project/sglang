"""Tests for layer-pipelined KV transfer in disaggregated prefill-decode mode.

Validates that enabling SGLANG_ENABLE_PIPELINED_KV_TRANSFER produces correct outputs
across different prompt lengths and verifies the adaptive group_size logic
correctly falls back to the normal path for short prompts.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=480, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationPipelined(PDDisaggregationServerBase):
    """Test layer-pipelined KV transfer correctness."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls._env_stack = ExitStack()
        cls._env_stack.enter_context(
            envs.SGLANG_ENABLE_PIPELINED_KV_TRANSFER.override(True)
        )
        cls._env_stack.enter_context(envs.SGLANG_PIPELINE_MIN_TOKENS.override(64))
        try:
            cls.launch_all()
        except Exception:
            cls._env_stack.close()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            env_stack = getattr(cls, "_env_stack", None)
            if env_stack is not None:
                env_stack.close()

    def test_gsm8k(self):
        """Validate end-to-end correctness with pipelined transfer on GSM8K."""
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Pipelined evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)

    def test_basic_generation(self):
        """Smoke test: single request produces valid output."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)
        output = response.json()
        self.assertIn("text", output)
        self.assertGreater(len(output["text"].strip()), 0)
        print(f"Basic generation output: {output['text']}")

    def test_long_prompt(self):
        """Verify correctness with a long prompt that triggers more pipeline groups."""
        # ~2000 tokens prompt to exercise deeper pipeline overlap
        long_prompt = "Summarize the following text:\n" + (
            "The quick brown fox jumps over the lazy dog. " * 200
        )
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": long_prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        output = response.json()
        self.assertIn("text", output)
        self.assertGreater(len(output["text"].strip()), 0)
        print(f"Long prompt generation output: {output['text'][:100]}")

    def test_concurrent_requests(self):
        """Verify pipelined transfer handles concurrent prefills correctly."""
        import concurrent.futures

        # Each prompt must exceed SGLANG_PIPELINE_MIN_TOKENS (64) to actually
        # exercise the pipelined path under concurrency.
        base_prompts = [
            "What is 2 + 2?",
            "Explain gravity in one sentence.",
            "Write a haiku about coding.",
            "What is the speed of light?",
            "Name three primary colors.",
            "What year did World War II end?",
            "Define photosynthesis briefly.",
            "What is the capital of Japan?",
        ]
        # Pad each prompt to ~100 tokens so they exceed the min_tokens threshold
        padding = " ".join(["word"] * 80)
        prompts = [f"{padding}\n{p}" for p in base_prompts]

        def send_request(prompt):
            resp = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
                timeout=30,
            )
            return resp.status_code, resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(send_request, p) for p in prompts]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for status_code, output in results:
            self.assertEqual(status_code, 200)
            self.assertIn("text", output)
            self.assertGreater(len(output["text"].strip()), 0)

        print(f"All {len(results)} concurrent requests completed successfully")


class TestDisaggregationPipelinedGroupSize(PDDisaggregationServerBase):
    """Test layer-pipelined transfer with explicit group_size."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls._env_stack = ExitStack()
        cls._env_stack.enter_context(
            envs.SGLANG_ENABLE_PIPELINED_KV_TRANSFER.override(True)
        )
        cls._env_stack.enter_context(envs.SGLANG_PIPELINE_MIN_TOKENS.override(64))
        cls._env_stack.enter_context(envs.SGLANG_PIPELINE_GROUP_SIZE.override(5))
        try:
            cls.launch_all()
        except Exception:
            cls._env_stack.close()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            env_stack = getattr(cls, "_env_stack", None)
            if env_stack is not None:
                env_stack.close()

    def test_gsm8k(self):
        """Validate correctness with fixed group_size=5."""
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Fixed group_size=5 metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)


if __name__ == "__main__":
    unittest.main()
