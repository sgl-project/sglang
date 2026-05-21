"""Regression for PR #25015 EAGLE positions misalign: revert the fix and expect canary fire."""

from __future__ import annotations

import json
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")

_SPEC_EAGLE_TOKEN_ORACLE_ENV = {
    "SGLANG_KV_CANARY_INPUT_CHECK": "0",
    "SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1",
}
_SPEC_EAGLE_SERVER_ARGS = (
    "--json-model-override-args",
    json.dumps({"num_hidden_layers": 1}),
    "--sampling-backend",
    "token_oracle",
    "--speculative-algorithm",
    "EAGLE",
    "--cuda-graph-max-bs",
    "8",
    "--max-running-requests",
    "32",
    "--max-total-tokens",
    "16384",
    "--disable-piecewise-cuda-graph",
)


class TestEaglePositionsMisalignRegression(CanaryE2EBase, unittest.TestCase):
    """Revert PR #25015 fix and expect canary to fire a position-mismatch violation."""

    model_mode = "mha"
    kv_canary_mode = "raise"
    extra_server_args = _SPEC_EAGLE_SERVER_ARGS
    extra_env = {
        **_SPEC_EAGLE_TOKEN_ORACLE_ENV,
        "SGLANG_DEBUG_REVERT_PR": "25015",
    }

    def test_position_mismatch_in_server_stderr(self) -> None:
        self.assert_violation_logged_any(
            launch_tag_patterns=("*",),
            fail_reason="position",
            flush_wait_seconds=0.0,
        )


class TestEaglePositionsMatchWithFix(CanaryE2EBase, unittest.TestCase):
    """With the PR #25015 fix in place, no canary fires."""

    model_mode = "mha"
    kv_canary_mode = "raise"
    extra_server_args = _SPEC_EAGLE_SERVER_ARGS
    extra_env = _SPEC_EAGLE_TOKEN_ORACLE_ENV

    def test_no_canary_fire(self) -> None:
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": list(range(1, 65)),
                "sampling_params": {
                    "max_new_tokens": 4,
                    "temperature": 0.0,
                },
            },
            timeout=60.0,
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        health = requests.get(self.base_url + "/health", timeout=10.0)
        self.assertEqual(health.status_code, 200, health.text)
        self.assert_no_violation(wait_seconds=2.0)


if __name__ == "__main__":
    unittest.main()
