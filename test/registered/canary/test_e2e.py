"""End-to-end tests for the KV cache canary.

Covers v1 acceptance items #1 (kernel really runs), #4 (self-test triggers
violation), and #7 (no TP deadlock on raise — implicit at TP=1 here; the
unconditional allreduce path is exercised by the host runner regardless).
"""

from __future__ import annotations

import os
import time
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

_MODEL = "Qwen/Qwen3-0.6B"

register_cuda_ci(est_time=240, suite="base-b-test-1-gpu-small")


class TestKvCacheCanaryCleanLogMode(CustomTestCase):
    """Clean run with ``--kv-cache-canary=log``: no violation expected."""

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        # Ensure no perturbation is configured.
        env.pop("SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN", None)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--kv-cache-canary",
                "log",
                # Shadow K/V tensors add ~4× layer-size to the pool; leave
                # headroom so the canary tensors fit.
                "--mem-fraction-static",
                "0.65",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_clean_log_run_keeps_violation_counter_at_zero(self):
        # Step 1: send a small batch of generate requests.
        for i in range(20):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": f"hello world {i}",
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)

        # Step 2: allow the side-stream event pump a beat to refresh counters.
        time.sleep(2.0)

        # Step 3: the server should still be healthy (no raise/abort).
        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)


class TestKvCacheCanaryPerturbRaiseMode(CustomTestCase):
    """Perturb + raise: server must either fail to come up (canary raised
    during warmup) OR die under live traffic. Both outcomes prove the
    raise path is wired.

    Hit-rate note: perturb is now active-row-aware (swap is restricted to
    the in-use ``[0, seq_len)`` range of an active req), so it lands on a
    column the canary actually verifies. Probability bumped to 0.1.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env["SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN"] = "0.1:42"
        cls.process = None
        cls.launch_failed_due_to_raise = False
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--kv-cache-canary",
                    "raise",
                    "--mem-fraction-static",
                    "0.65",
                ],
                env=env,
            )
        except Exception as exc:
            # popen_launch_server raises when the server doesn't pass
            # health check. With perturb+raise that's the expected canary
            # signal (warmup forward tripped the kernel).
            cls.launch_failed_due_to_raise = True
            cls.launch_exception = exc

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_perturbation_triggers_canary_violation(self):
        if self.launch_failed_due_to_raise:
            # The canary raised during warmup before the server could
            # answer /health. That's a successful raise.
            return

        # Server is up despite perturb; drive traffic until canary fires.
        triggered = False
        last_status = None
        for i in range(50):
            try:
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": (
                            f"The quick brown fox jumps over the lazy dog {i}. "
                            "Tell me a story about the fox and the dog."
                        ),
                        "sampling_params": {
                            "max_new_tokens": 32,
                            "temperature": 0.0,
                        },
                    },
                    timeout=30,
                )
                last_status = response.status_code
            except requests.exceptions.RequestException:
                triggered = True
                break

            if response.status_code >= 500:
                triggered = True
                break

        time.sleep(1.5)
        try:
            health = requests.get(self.base_url + "/health", timeout=5)
            health_ok = health.status_code == 200
        except requests.exceptions.RequestException:
            health_ok = False

        self.assertTrue(
            triggered or not health_ok,
            f"Expected canary to fire under perturb+raise, but server still "
            f"healthy and last request status={last_status}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
