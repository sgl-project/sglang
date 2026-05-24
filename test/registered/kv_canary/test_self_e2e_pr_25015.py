"""Regression for PR #25015 EAGLE positions misalign: revert the fix and expect canary fire."""

from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")

_SPEC_EAGLE_TOKEN_ORACLE_ENV = {
    # Enable write-time oracle input checks so the canary write kernel can
    # directly catch a perturbed-position eagle draft write as
    # ``fail_reason=write_position`` on the affected slot. Without this,
    # detection has to wait for a later per-forward verify cycle to include
    # the perturbed slot in its prefix scope — which never happens when the
    # PR-revert garbled outputs make the eagle requests time out before
    # producing enough decode tokens.
    "SGLANG_KV_CANARY_INPUT_CHECK": "1",
    "SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1",
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
    __test__ = False  # pytest must not collect the abstract base (revert_pr is unset)

    model_mode = "mha"
    kv_canary_mode = CanaryMode.RAISE
    extra_server_args = _SPEC_EAGLE_SERVER_ARGS
    revert_pr: ClassVar[bool]

    @classmethod
    def setUpClass(cls) -> None:
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
            # Reverting PR #25015 shifts every eagle draft position by +1. With
            # SGLANG_KV_CANARY_INPUT_CHECK=1 the canary write kernel directly
            # compares each write's actual position against the oracle's
            # expected position and fires ``write_position`` immediately on the
            # first perturbed draft write. Fall back to ``position`` /
            # ``chain_hash`` (the per-forward-verify signals) if a different
            # cycle catches it first.
            for reason in ("write_position", "position", "chain_hash"):
                try:
                    self.assert_violation_logged_any(
                        launch_tag_patterns=("*",),
                        fail_reason=reason,
                        flush_wait_seconds=0.0,
                    )
                    return
                except AssertionError:
                    continue
            raise AssertionError(
                "Expected canary to fire one of fail_reason ∈ "
                "{write_position, position, chain_hash} after reverting PR #25015, "
                "but none was logged."
            )
        else:
            self.assert_no_violation(wait_seconds=2.0)


class TestEaglePositionsMisalignRegression(_EaglePositionsBase):
    """Revert PR #25015 fix and expect canary to fire a position-mismatch violation."""

    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    revert_pr = True


class TestEaglePositionsMatchWithFix(_EaglePositionsBase):
    """With the PR #25015 fix in place, no canary fires."""

    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    revert_pr = False


if __name__ == "__main__":
    unittest.main()
