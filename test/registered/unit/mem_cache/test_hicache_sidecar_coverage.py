import unittest

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiCacheSidecarCoverage(unittest.TestCase):
    def test_full_sidecar_coverage_keeps_completed_tokens(self):
        result = PoolTransferResult.empty()
        result.update_extra_pool_hit_pages({PoolName.INDEXER: [True, True, True]})

        completed_tokens = result.clamp_to_all_pages_coverage(
            completed_tokens=12,
            page_size=4,
            pool_transfers=[PoolTransfer(name=PoolName.INDEXER)],
        )

        self.assertEqual(completed_tokens, 12)

    def test_short_sidecar_result_clamps_completed_tokens(self):
        result = PoolTransferResult.empty()
        result.update_extra_pool_hit_pages({PoolName.INDEXER: [True, False, False]})

        completed_tokens = result.clamp_to_all_pages_coverage(
            completed_tokens=12,
            page_size=4,
            pool_transfers=[PoolTransfer(name=PoolName.INDEXER)],
        )

        self.assertEqual(completed_tokens, 4)

    def test_sidecar_hole_clamps_at_leading_prefix(self):
        result = PoolTransferResult.empty()
        result.update_extra_pool_hit_pages({PoolName.INDEXER: [True, False, True]})

        completed_tokens = result.clamp_to_all_pages_coverage(
            completed_tokens=12,
            page_size=4,
            pool_transfers=[PoolTransfer(name=PoolName.INDEXER)],
        )

        self.assertEqual(result.extra_pool_hit_pages[PoolName.INDEXER], 2)
        self.assertEqual(completed_tokens, 4)

    def test_missing_required_sidecar_result_clamps_to_zero(self):
        result = PoolTransferResult.empty()

        completed_tokens = result.clamp_to_all_pages_coverage(
            completed_tokens=12,
            page_size=4,
            pool_transfers=[PoolTransfer(name=PoolName.INDEXER)],
        )

        self.assertEqual(completed_tokens, 0)

    def test_no_all_pages_sidecars_keeps_completed_tokens(self):
        result = PoolTransferResult.empty()

        completed_tokens = result.clamp_to_all_pages_coverage(
            completed_tokens=12,
            page_size=4,
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.MAMBA,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ],
        )

        self.assertEqual(completed_tokens, 12)


if __name__ == "__main__":
    unittest.main()
