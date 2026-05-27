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
        if cls is _PerturbRealKvUnusedCacheBase:
            raise unittest.SkipTest(
                "abstract base; concrete subclasses set model_mode + target_group"
            )
        cls.extra_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB": "0.1",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": str(cls.target_group),
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
            # The violation ring is fill-once (writes beyond ring_capacity are
            # dropped). In SWA mode the SWA-side sweep fires ~1k chain-hash
            # violations within the first second of the test (window-shift
            # natural noise), which fills the default 1024-entry ring before
            # any FULL-side perturb has a chance to record its real-kv-hash
            # mismatch. Bumping the ring to 64K keeps every group's
            # violations visible to the reporter through the full test
            # window.
            "SGLANG_KV_CANARY_RING_CAPACITY": "65536",
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
        # Accept either fail_reason: the SWA-side sweep verifier runs both
        # chain-hash and real-kv-hash checks against the same corruption, and
        # which check fires first is a race. Both legitimately detect this
        # perturbation; either reaching the violation log is success.
        self.assert_sweep_violation_reported(
            fail_reason=("verify_real_kv_hash", "verify_chain_hash"),
            target_group=self.target_group,
            flush_wait_seconds=5.0,
        )
        self.maybe_assert_swa_divergence_observed()


class TestPerturbRealKvUnusedCacheMhaFull(_PerturbRealKvUnusedCacheBase):
    model_mode = "mha"
    target_group = TargetGroupKind.FULL


class TestPerturbRealKvUnusedCacheSwaFull(_PerturbRealKvUnusedCacheBase):
    model_mode = "swa"
    target_group = TargetGroupKind.FULL
    extra_server_args = (
        *_PerturbRealKvUnusedCacheBase.extra_server_args,
        *SWA_POOL_SERVER_ARGS,
    )


class TestPerturbRealKvUnusedCacheSwaSwa(_PerturbRealKvUnusedCacheBase):
    model_mode = "swa"
    target_group = TargetGroupKind.SWA
    extra_server_args = (
        *_PerturbRealKvUnusedCacheBase.extra_server_args,
        *SWA_POOL_SERVER_ARGS,
    )


if __name__ == "__main__":
    unittest.main()
