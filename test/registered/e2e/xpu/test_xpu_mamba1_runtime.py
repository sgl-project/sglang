"""
Prefill -> decode runtime test for the Mamba-1 mixer on Intel XPU.

Guards the MambaMixer1 <-> Mamba2AttnBackend contract (3-tuple return) and the
SSM conv/state cache end to end; the CPU weight-remap test cannot catch either.
Uses a real server so the scheduler initializes the mamba selective-scan backend.

Usage:
  python3 -m unittest test_xpu_mamba1_runtime.TestXPUMamba1Runtime
"""

import unittest

import requests

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_xpu_ci(est_time=600, suite="stage-b-test-1-gpu-xpu")

# Small Mamba-1 (state-spaces) checkpoint; exercises MambaMixer1 on XPU.
MODEL = "state-spaces/mamba-130m-hf"


class TestXPUMamba1Runtime(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="xpu",
            other_args=[
                "--device",
                "xpu",
                "--attention-backend",
                "intel_xpu",
                "--disable-radix-cache",  # Mamba-1 has no radix track state
                "--max-total-tokens",
                "65536",
                "--mem-fraction-static",
                "0.9",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, prompt, max_new_tokens=32):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["text"]

    def test_prefill_to_decode(self):
        # Multi-token prompt forces a real prefill scan; max_new_tokens>1 forces the
        # decode-step recurrence. Before the 3-tuple fix this raised
        # "not enough values to unpack" on the first forward.
        out = self._generate("The capital of France is")
        self.assertTrue(out and out.strip(), "empty completion")

    def test_greedy_is_deterministic(self):
        # Identical greedy requests must match; a corrupted conv/ssm state cache
        # across requests would make them diverge.
        prompt = "Count: one two three"
        self.assertEqual(self._generate(prompt), self._generate(prompt))


if __name__ == "__main__":
    unittest.main()
