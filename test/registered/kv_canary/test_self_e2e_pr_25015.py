"""Regression for PR #25015 EAGLE positions misalign: revert the fix and expect canary fire."""

from __future__ import annotations

import logging
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

logger = logging.getLogger(__name__)

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")

_SPEC_EAGLE_TOKEN_ORACLE_ENV = {
    "SGLANG_KV_CANARY_INPUT_CHECK": "0",
    "SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1",
}


class TestEaglePositionsMisalignRegression(CanaryE2EBase, unittest.TestCase):
    """Revert PR #25015 fix and expect canary to fire a position-mismatch violation."""

    model_mode = "spec_eagle"
    kv_canary_mode = "raise"
    extra_env = {
        **_SPEC_EAGLE_TOKEN_ORACLE_ENV,
        "SGLANG_DEBUG_REVERT_PR": "25015",
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls._launch_exc = None
        try:
            super().setUpClass()
        except Exception as exc:
            cls._launch_exc = exc
            logger.warning(
                "server launch raised during revert path: %r", exc, exc_info=True
            )

    def test_position_mismatch_in_server_stderr(self) -> None:
        self.assert_violation_logged_any(
            launch_tag_patterns=("*",),
            fail_reason="position",
            flush_wait_seconds=0.0,
        )


class TestEaglePositionsMatchWithFix(CanaryE2EBase, unittest.TestCase):
    """With the PR #25015 fix in place, no canary fires."""

    model_mode = "spec_eagle"
    kv_canary_mode = "raise"
    extra_env = _SPEC_EAGLE_TOKEN_ORACLE_ENV

    def test_no_canary_fire(self) -> None:
        resp = self.send_token_id_request(input_ids=list(range(1, 65)))
        self.assertEqual(resp.status_code, 200, resp.text)

        self.assert_server_healthy()
        self.assert_no_violation(wait_seconds=2.0)


if __name__ == "__main__":
    unittest.main()
