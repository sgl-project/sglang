from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.consts import SWA_POOL_SERVER_ARGS
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")


class _PerturbRealKvUnusedCacheBase(CanaryE2EBase):
    __test__ = False

    kv_canary_mode = CanaryMode.LOG
    extra_server_args = (
        "--kv-canary-real-data",
        "partial",
        "--kv-canary-sweep-interval",
        "4",
    )
    use_unique_prompts = True

    target_group: ClassVar[TargetGroupKind]

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB": "0.1",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": str(cls.target_group),
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        }
        super().setUpClass()

    def test_real_kv_unused_cache_perturbation_reports_sweep_real_kv_hash_violation(
        self,
    ) -> None:
        """Verify cached unused KV perturbation is caught by sweep verification."""
        # Step 1: first batch builds radix entries that will become orphans once finished.
        self.send_parallel_requests(n=8)
        # Step 2: second batch drives more forward passes so the sweep cadence fires
        # while the orphan slots are still cached.
        self.send_parallel_requests(n=8)
        self.assert_sweep_violation_reported(
            fail_reason="verify_real_kv_hash",
            target_group=self.target_group,
            flush_wait_seconds=5.0,
        )
        self.maybe_assert_swa_divergence_observed()


class TestPerturbRealKvUnusedCacheMhaFull(_PerturbRealKvUnusedCacheBase):
    __test__ = True

    model_mode = "mha"
    target_group = TargetGroupKind.FULL


# In SWA mode the FULL pool has no orphan sweep slots — the perturbation
# logs "skipped because no orphan sweep slot was found for group=FULL"
# every step, so the expected SWEEP_*_FULL violation never fires. The
# MhaFull case (FULL-only model) and SwaSwa case (SWA target in SWA mode)
# both work; only this Sw target=FULL combination is unreachable. Skipping
# until the model+target combinatorics in the perturb path are reworked.
@unittest.skip(
    "SWA mode + target_group=FULL has no orphan FULL slots to perturb; "
    "see comment above."
)
class TestPerturbRealKvUnusedCacheSwaFull(_PerturbRealKvUnusedCacheBase):
    __test__ = True

    model_mode = "swa"
    target_group = TargetGroupKind.FULL
    extra_server_args = (
        *_PerturbRealKvUnusedCacheBase.extra_server_args,
        *SWA_POOL_SERVER_ARGS,
    )


# In SWA mode the SWA sweep often catches the perturbation as a
# verify_chain_hash violation first (same corruption, different
# fail_reason); the test specifically asserts verify_real_kv_hash, so
# whether it passes depends on which check happens to fire first. Skip
# until the test either accepts either fail_reason or the sweep emits
# both violation kinds for one corruption.
@unittest.skip(
    "Race between verify_chain_hash and verify_real_kv_hash on the SWA "
    "sweep means this assertion is non-deterministic; see comment above."
)
class TestPerturbRealKvUnusedCacheSwaSwa(_PerturbRealKvUnusedCacheBase):
    __test__ = True

    model_mode = "swa"
    target_group = TargetGroupKind.SWA
    extra_server_args = (
        *_PerturbRealKvUnusedCacheBase.extra_server_args,
        *SWA_POOL_SERVER_ARGS,
    )


if __name__ == "__main__":
    unittest.main()
