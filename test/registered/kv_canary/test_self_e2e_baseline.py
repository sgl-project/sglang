from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class _BaselineBase(CanaryE2EBase):
    """No perturb, kv-canary=log, sweep off. Server should run clean with no canary
    violations and every request must come back 200."""

    __test__ = False

    kv_canary_mode = CanaryMode.LOG
    # V coverage needed for SwaBase's SWA-divergence assertion to see V-side divergences.
    extra_env = {"SGLANG_KV_CANARY_ENABLE_MHA_V": "1"}

    def test_no_violation(self) -> None:
        """Verify the baseline canary run completes without violations."""
        self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)
        self.maybe_assert_swa_divergence_observed()


class TestBaselineMha(_BaselineBase):
    __test__ = True

    model_mode = "mha"


class TestBaselineSwa(_BaselineBase):
    __test__ = True

    model_mode = "swa"
    # Tight KV pool forces eviction under the 8 × ~7K parallel prompts, which slides the SWA
    # window past the full-pool tail and produces non-zero swa_full_idx_divergence.
    extra_server_args = ("--max-total-tokens", "8192")


if __name__ == "__main__":
    unittest.main()
