"""End-to-end canary tests on Qwen3-0.6B.

Two paired scenarios:

- ``TestKvCacheCanaryCleanRaise``: ``--kv-cache-canary=raise`` with no
  fault injection. Server must come up and stay healthy under a parallel
  request burst — any false positive would abort the server.
- ``TestKvCacheCanaryPerturbRaise``: same config but with
  ``SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB=0.01``. The canary must
  fire (either at warmup or under live traffic).
"""

from __future__ import annotations

import time
import unittest
from typing import List

import requests

from sglang.test.canary_e2e_base import CanaryE2EBase
from sglang.test.ci.ci_register import register_cuda_ci

_MODEL = "Qwen/Qwen3-0.6B"

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-small")


class TestKvCacheCanaryCleanRaise(CanaryE2EBase):
    """Clean run with ``--kv-cache-canary=raise``: no violation expected."""

    model = _MODEL

    def test_clean_raise_run_stays_healthy(self) -> None:
        results = self.send_parallel_requests(n=64)
        bad = [r for r in results if r.get("status_code") != 200]
        self.assertFalse(bad, f"non-200 responses on clean run: {bad[:3]}")

        # Allow the side-stream event pump a beat to refresh counters.
        time.sleep(2.0)

        self.assert_health_ok()


class TestKvCacheCanaryPerturbRaise(CanaryE2EBase):
    """Perturb + raise: server must either fail to come up (canary raised
    during warmup) OR die under live traffic. Both outcomes prove the
    raise path is wired.

    Perturb is active-row-aware (swap is restricted to the in-use
    ``[0, seq_len)`` range of an active req), so it lands on a column
    the canary actually verifies. Probability is 0.01 — sparse enough
    to keep per-forward mismatches rare, but over ~50 parallel requests
    we expect a hit.
    """

    model = _MODEL
    perturb_prob = 0.01
    perturb_seed = 42
    allow_launch_failure = True

    def test_perturbation_triggers_canary_violation(self) -> None:
        if not self.launch_failed:
            results: List[dict] = self.send_parallel_requests(
                n=128, max_new_tokens=32, timeout=30.0
            )
            triggered = any(
                "error" in r or int(r.get("status_code", 0)) >= 500 for r in results
            )

            # Give the server a moment, then probe /health.
            time.sleep(1.5)
            try:
                health_ok = (
                    requests.get(self.base_url + "/health", timeout=5).status_code
                    == 200
                )
            except requests.exceptions.RequestException:
                health_ok = False

            self.assertTrue(
                triggered or not health_ok,
                f"Expected canary to fire under perturb+raise, but server still "
                f"healthy and no failed requests; first 3 responses: {results[:3]}",
            )
        # Hard-assert the per-step path is what caught it (req_to_token perturb
        # corrupts slots the head/tail kernels verify directly).
        self.assert_violation_kind_logged(["head_k", "head_v", "tail_k", "tail_v"])


if __name__ == "__main__":
    unittest.main(verbosity=3)
