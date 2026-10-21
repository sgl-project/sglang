"""End-to-end smoke test for the standard Torch ModelRunner on Apple MPS."""

import os
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_mps_ci(est_time=240, suite="stage-b-e2e-mps")

MODEL_PATH = os.environ.get("SGLANG_MPS_TEST_MODEL", "Qwen/Qwen3-0.6B")


@unittest.skipUnless(
    hasattr(torch, "mps") and torch.mps.is_available(), "requires Apple MPS"
)
class TestQwen3TorchMps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST

    def _generate(self, prompt: str) -> dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "ignore_eos": True,
                    "max_new_tokens": 4,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def test_standard_runner_reuses_radix_cache(self):
        # Ensure the child uses Torch rather than the MLX runner.
        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "0"
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="mps",
            env=env,
            other_args=[
                "--disable-overlap-schedule",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
                "--mem-fraction-static",
                "0.6",
                "--max-total-tokens",
                "4096",
                "--context-length",
                "2048",
            ],
        )
        try:
            prompt = (
                "This is a deterministic prefix about science and geography. " * 24
                + "The capital of France is"
            )
            cold = self._generate(prompt)
            warm = self._generate(prompt)

            self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
            self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
            self.assertEqual(cold["output_ids"], warm["output_ids"])
        finally:
            kill_process_tree(process.pid, wait_timeout=30)
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
