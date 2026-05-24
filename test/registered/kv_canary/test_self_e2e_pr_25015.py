"""Regression for PR #25015 EAGLE positions misalign: revert the fix and expect canary fire."""

from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")

_SPEC_EAGLE_TOKEN_ORACLE_ENV = {
    "SGLANG_KV_CANARY_INPUT_CHECK": "0",
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
    # Skip server-stage warmup: under revert_pr=True the eagle-draft position mismatch fires the
    # canary on the first warmup decode request, which kills the server in CanaryMode.RAISE before
    # setUpClass even returns. The test sends its own parallel requests after the server is up; the
    # geometric assert still triggers there and surfaces the regression.
    "--skip-server-warmup",
)


class _EaglePositionsBase(CanaryE2EBase):
    __test__ = False  # pytest must not collect the abstract base (revert_pr is unset)

    model_mode = "mha"
    kv_canary_mode = CanaryMode.RAISE
    extra_server_args = _SPEC_EAGLE_SERVER_ARGS
    # Use unique-prefix prompts so each request keeps its own KV-cache slots
    # without radix folding. This isolates the eagle position-mismatch
    # detection path from the prefix-share verify path, which has subtle
    # chain-hash recomputation timing that's outside the scope of this test.
    use_unique_prompts: ClassVar[bool] = True
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
            # Reverting PR #25015 shifts every eagle draft position by +1. The
            # mismatch manifests as either a direct ``position`` bit (if the
            # write-side stored_position differs from verify-side
            # expected_position on a slot that gets per-forward-verified) or a
            # propagated ``chain_hash`` bit (if the affected slot is reached
            # transitively through the per-forward chain on a sibling slot
            # — happens when canary mode is RAISE so the first detected
            # violation kills the scheduler before the draft slot itself
            # appears in a later prefix). Either signal proves canary caught
            # the regression; accept whichever fires first.
            for reason in ("position", "chain_hash"):
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
                "Expected canary to fire one of fail_reason ∈ {position, chain_hash} "
                "after reverting PR #25015, but neither was logged."
            )
        else:
            self.assert_no_violation(wait_seconds=2.0)


class TestEaglePositionsMisalignRegression(_EaglePositionsBase):
    """Revert PR #25015 fix and expect canary to fire a position-mismatch violation.

    Caught at write-time by the kernel's init-gated geometric write_position
    assert (canary_write.cuh): single-entry decode writes must satisfy
    ``position == seed.position + 1 + entry_offset``. Eagle DRAFT under the
    reverted PR perturbs position by +1, breaking that arithmetic and firing
    ``fail_reason = position``. The assert is gated on
    ``CanaryDeviceState.runtime_assert_enable`` (flipped from 0 to 1 in
    ``CanaryManager.mark_init_finished()``) so warmup / cuda-graph capture
    paths — whose seed slots may hold synthetic positions — don't misfire.
    """

    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    revert_pr = True


class TestEaglePositionsMatchWithFix(_EaglePositionsBase):
    """With the PR #25015 fix in place, no canary fires."""

    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    revert_pr = False


if __name__ == "__main__":
    unittest.main()
