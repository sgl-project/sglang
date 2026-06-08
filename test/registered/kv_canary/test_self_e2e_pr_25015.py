"""Regression for PR #25015 EAGLE positions misalign: revert the fix and expect canary fire."""

from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=98, stage="extra-a", runner_config="1-gpu-small")

_SPEC_EAGLE_TOKEN_ORACLE_ENV = {
    "SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT": "0",
    "SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1",
    "SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT": "0",
}
_SPEC_EAGLE_REVERT_PR_ENV = {
    **_SPEC_EAGLE_TOKEN_ORACLE_ENV,
    "SGLANG_DEBUG_REVERT_PR": "25015",
}
_CUDA_GRAPH_MAX_BS = 1
_EAGER_DRAFT_REQUEST_COUNT = 20
assert _EAGER_DRAFT_REQUEST_COUNT > _CUDA_GRAPH_MAX_BS

_SPEC_EAGLE_SERVER_ARGS = (
    "--sampling-backend",
    "token_oracle",
    "--speculative-algorithm",
    "EAGLE",
    "--cuda-graph-max-bs",
    str(_CUDA_GRAPH_MAX_BS),
    "--max-running-requests",
    "32",
)


class _EaglePositionsBase(CanaryE2EBase):
    model_mode = "mha"
    # LOG mode keeps the server alive after the first violation so server warmup + this test's
    # parallel requests both run; we then read the violation log to assert the position bit fired.
    kv_canary_mode = CanaryMode.LOG
    extra_server_args = _SPEC_EAGLE_SERVER_ARGS
    revert_pr: ClassVar[bool]

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _EaglePositionsBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set revert_pr")
        cls.extra_env = (
            _SPEC_EAGLE_REVERT_PR_ENV if cls.revert_pr else _SPEC_EAGLE_TOKEN_ORACLE_ENV
        )
        super().setUpClass()

    def test_pr_25015_eagle_positions(self) -> None:
        self.send_parallel_requests(
            n=_EAGER_DRAFT_REQUEST_COUNT,
            assert_all_success=not self.revert_pr,
            max_new_tokens=32,
            timeout=60.0,
        )

        if self.revert_pr:
            self.assert_violation_logged_any(
                launch_tag_patterns=("*",),
                fail_reason="verify_position",
                flush_wait_seconds=0.0,
            )
        else:
            self.assert_no_violation(wait_seconds=2.0)


class TestEaglePositionsMisalignRegression(_EaglePositionsBase):
    """Revert PR #25015 fix and expect canary to fire a position-mismatch violation."""

    revert_pr = True


class TestEaglePositionsMatchWithFix(_EaglePositionsBase):
    """With the PR #25015 fix in place, no canary fires."""

    revert_pr = False


if __name__ == "__main__":
    unittest.main()
