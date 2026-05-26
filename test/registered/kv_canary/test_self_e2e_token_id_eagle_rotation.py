"""Regression for PR #26329 EAGLE chunked-prefill rotation: revert the fix and
expect the kv_canary real-model token-id validator to fire.

The bug: when a req is mid-chunked-prefill, the EAGLE draft prefill rotation
overrides the chunk-1 seg-end tail with the target model's predicted next
token instead of the next prompt token. PR #26329 fixes this by reading
``req.origin_input_ids`` at the chunk boundary. The kv_canary token-id
validator (``SGLANG_KV_CANARY_ENABLE_REQ_TOKEN_IDS_CHECK=1``) reads the same
``origin_input_ids + output_ids`` per req and asserts the write kernel's
``input_ids`` matches the source-of-truth at the chunk seg-end's logical
position, so the buggy revert reliably fires a ``write_token`` violation
without depending on any inter-request race.
"""

from __future__ import annotations

import random
import string
import unittest
from typing import ClassVar

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")


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
    __test__ = False  # pytest must not collect the abstract base (revert_pr is unset)

    model_mode = "mha"
    # LOG mode keeps the server alive so we can read the violation log
    # after the request completes (vs RAISE which would crash the worker
    # on the first violation).
    kv_canary_mode = CanaryMode.LOG
    extra_server_args = _EAGLE_CHUNKED_SERVER_ARGS

    revert_pr: ClassVar[bool]

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_env = {"SGLANG_DEBUG_REVERT_PR": "26329"} if cls.revert_pr else {}
        super().setUpClass()

    def make_prompts(self, n: int) -> list[str]:
        # Override the default repetitive English body. The chunked-prefill bug
        # only manifests as a TOKEN mismatch when target's predicted bonus token
        # differs from the next prompt token; with predictable text the model
        # often guesses right and the validator never fires. Seeded random ASCII
        # is unpredictable enough to make ``next_token_ids[0] != prompt[K1]``
        # at every chunk boundary.
        rng = random.Random(0)
        # ~3K tokens after BPE — long enough to span 2+ chunked-prefill chunks
        # at chunked_prefill_size=2048, short enough to stay under context_length.
        body = "".join(
            rng.choices(string.ascii_letters + string.digits + " ", k=8000)
        )
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
                fail_reason="write_token",
                flush_wait_seconds=3.0,
            )
        else:
            self.assert_no_violation(wait_seconds=2.0)


class TestEagleChunkedRotationRegression(_EagleChunkedRotationBase):
    """Revert PR #26329 fix; expect canary to fire a write_token violation."""

    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    revert_pr = True


class TestEagleChunkedRotationClean(_EagleChunkedRotationBase):
    """With the PR #26329 fix in place, the same request runs clean."""

    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    revert_pr = False


if __name__ == "__main__":
    unittest.main()
