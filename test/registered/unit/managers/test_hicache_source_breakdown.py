"""CPU coverage for fail-closed HiCache source breakdown aggregation."""

import unittest

from sglang.srt.managers.schedule_batch import compute_hicache_source_breakdown
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestComputeHiCacheSourceBreakdown(CustomTestCase):
    def test_complete_explicit_breakdown_preserves_totals(self):
        # Host total is split into storage and non-storage host portions; the
        # two explicit source counters account for the latter exactly.
        result = compute_hicache_source_breakdown(
            host_hit_length=10,
            host_hit_write_back=4,
            host_hit_prefetch=3,
            storage_hit_length=3,
            prefix_indices_len=20,
            host_source_ranges=((10, 14, "device_backup"), (14, 20, "l3_prefetch")),
        )
        self.assertEqual(result, (10, 7, 3, 4, 3))
        device, host, storage, write_back, prefetch = result
        self.assertEqual(device + host + storage, 20)
        self.assertEqual(write_back + prefetch, host)

    def test_ranges_clip_materialized_host_and_exclude_current_l3_reads(self):
        ranges = ((0, 4, "device_backup"), (4, 10, "l3_prefetch"))
        cases = [
            # H2D took only the suffix of this chain.
            (6, 0, None, 10, (4, 6, 0, 0, 6)),
            # Current-request L3 replaced a middle range of the old backup.
            (10, 2, 1, 10, (0, 8, 2, 2, 6)),
            # The cached prefix itself was shortened before admission.
            (6, 0, None, 6, (0, 6, 0, 4, 2)),
            # L3 span outside the admitted prefix contributes nothing.
            (10, 2, 20, 10, (0, 10, 0, 4, 6)),
        ]
        for host, storage, start, prefix, expected in cases:
            with self.subTest(host=host, storage=storage, start=start, prefix=prefix):
                self.assertEqual(
                    compute_hicache_source_breakdown(
                        host,
                        4,
                        6,
                        storage,
                        prefix,
                        start,
                        host_source_ranges=ranges,
                    ),
                    expected,
                )

    def test_unknown_or_overlapping_ranges_do_not_claim_an_origin(self):
        for ranges in (
            ((0, 4, None), (4, 10, "l3_prefetch")),
            ((0, 6, "device_backup"), (4, 10, "l3_prefetch")),
            ((0, 4, "device_backup"),),
        ):
            with self.subTest(ranges=ranges):
                self.assertEqual(
                    compute_hicache_source_breakdown(
                        10, 4, 6, 0, 10, host_source_ranges=ranges
                    ),
                    (0, 10, 0, 0, 0),
                )

    def test_egress_omits_unknown_and_keeps_known_source_fields(self):
        from types import SimpleNamespace

        from sglang.srt.entrypoints.openai.utils import cached_tokens_details_from_dict
        from sglang.srt.managers.scheduler_components.output_streamer import (
            SchedulerOutputStreamer,
        )

        owner = SchedulerOutputStreamer.__new__(SchedulerOutputStreamer)
        owner.enable_hicache_storage = lambda: False
        req = SimpleNamespace(
            cached_tokens_device=2, cached_tokens_host=8, cached_tokens_storage=0
        )
        details = owner.get_cached_tokens_details(req)
        self.assertEqual(details, {"device": 2, "host": 8})
        req.cached_tokens_host_write_back, req.cached_tokens_host_prefetch = 3, 5
        details = owner.get_cached_tokens_details(req)
        encoded = cached_tokens_details_from_dict(details).model_dump()
        self.assertEqual(
            encoded, {"device": 2, "host": 8, "host_write_back": 3, "host_prefetch": 5}
        )
        req.cached_tokens_host_prefetch = 4
        self.assertNotIn("host_prefetch", owner.get_cached_tokens_details(req))

    def test_unknown_attribution_is_published_as_zero_zero(self):
        self.assertEqual(
            compute_hicache_source_breakdown(10, 0, 0, 3, 20),
            (10, 7, 3, 0, 0),
        )
        self.assertEqual(
            compute_hicache_source_breakdown(10, 4, 0, 3, 20),
            (10, 7, 3, 0, 0),
        )

    def test_negative_external_counts_are_clamped(self):
        self.assertEqual(
            compute_hicache_source_breakdown(-10, -4, -3, -2, -20),
            (0, 0, 0, 0, 0),
        )
        self.assertEqual(
            compute_hicache_source_breakdown(6, -4, 2, -1, 10),
            (4, 6, 0, 0, 0),
        )

    def test_inconsistent_source_counts_remain_unknown(self):
        result = compute_hicache_source_breakdown(10, 99, 0, 0, 20)
        self.assertEqual(result, (10, 10, 0, 0, 0))
        self.assertEqual(result[3] + result[4], 0)

        result = compute_hicache_source_breakdown(10, 8, 99, 0, 20)
        self.assertEqual(result, (10, 10, 0, 0, 0))
        self.assertEqual(result[3] + result[4], 0)

    def test_storage_is_clamped_to_host_total(self):
        result = compute_hicache_source_breakdown(10, 0, 0, 99, 20)
        self.assertEqual(result, (10, 0, 10, 0, 0))
        self.assertEqual(result[0] + result[1] + result[2], 20)

    def test_zero_host_portion_has_zero_source_counters(self):
        self.assertEqual(
            compute_hicache_source_breakdown(0, 5, 3, 0, 4),
            (4, 0, 0, 0, 0),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
