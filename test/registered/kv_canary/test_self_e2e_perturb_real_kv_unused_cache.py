from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class _PerturbRealKvUnusedCacheBase(CanaryE2EBase):
    """Perturb point (c): flip the first byte of a radix-cached but currently-unused
    (orphan) KV slot.

    Because the slot has no active reader, the per-forward HEAD/TAIL paths never touch
    it; only the periodic sweep does. The test asserts a real_kv_hash violation under
    a ``SWEEP_*_<GROUP>`` launch_tag. ``use_unique_prompts`` guarantees no prefix reuse
    so orphan slots stay orphan throughout the run.
    """

    kv_canary_mode = "log"
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
            fail_reason="real_kv_hash",
            target_group=self.target_group,
            flush_wait_seconds=5.0,
        )


class TestPerturbRealKvUnusedCacheMhaFull(
    _PerturbRealKvUnusedCacheBase, unittest.TestCase
):
    model_mode = "mha"
    target_group = TargetGroupKind.FULL


class TestPerturbRealKvUnusedCacheSwaFull(
    _PerturbRealKvUnusedCacheBase, unittest.TestCase
):
    model_mode = "swa"
    target_group = TargetGroupKind.FULL


class TestPerturbRealKvUnusedCacheSwaSwa(
    _PerturbRealKvUnusedCacheBase, unittest.TestCase
):
    model_mode = "swa"
    target_group = TargetGroupKind.SWA


if __name__ == "__main__":
    unittest.main()
