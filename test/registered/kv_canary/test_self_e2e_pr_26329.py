"""Regression for PR #26329 EAGLE chunked-prefill rotation."""

from __future__ import annotations

import random
import string
import unittest
from typing import ClassVar

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=81, stage="extra-a", runner_config="1-gpu-small")


_CHUNKED_PREFILL_SIZE = 2048
_EAGLE_CHUNKED_SERVER_ARGS = (
    "--speculative-algorithm",
    "EAGLE",
    "--chunked-prefill-size",
    str(_CHUNKED_PREFILL_SIZE),
    "--cuda-graph-max-bs",
    "1",
    "--max-running-requests",
    "4",
)


class _EagleChunkedRotationBase(CanaryE2EBase):
    model_mode = "mha"
    kv_canary_mode = CanaryMode.LOG
    extra_server_args = _EAGLE_CHUNKED_SERVER_ARGS

    revert_pr: ClassVar[bool]

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _EagleChunkedRotationBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set revert_pr")
        cls.extra_env = {"SGLANG_DEBUG_REVERT_PR": "26329"} if cls.revert_pr else {}
        super().setUpClass()

    def make_prompts(self, n: int) -> list[str]:
        # Seeded random ASCII so the model can't predict the next prompt token
        # — otherwise target's bonus token can accidentally match prompt[K1]
        # at the chunk boundary and the validator never fires.
        rng = random.Random(0)
        # ~3K tokens after BPE — spans 2+ chunks at chunked_prefill_size=2048.
        body = "".join(rng.choices(string.ascii_letters + string.digits + " ", k=8000))
        return [body] * n

    def test_chunked_rotation_token_id_mismatch(self) -> None:
        self.send_parallel_requests(
            n=1,
            assert_all_success=not self.revert_pr,
            max_new_tokens=8,
            timeout=60.0,
        )

        if self.revert_pr:
            self.assert_violation_logged_any(
                launch_tag_patterns=("*",),
                fail_reason="verify_token",
                flush_wait_seconds=3.0,
            )
        else:
            self.assert_no_violation(wait_seconds=2.0)


class TestEagleChunkedRotationRegression(_EagleChunkedRotationBase):
    """Revert PR #26329 fix; expect canary to fire a verify_token violation."""

    revert_pr = True


class TestEagleChunkedRotationClean(_EagleChunkedRotationBase):
    """With the PR #26329 fix in place, the same request runs clean."""

    revert_pr = False


if __name__ == "__main__":
    unittest.main()
