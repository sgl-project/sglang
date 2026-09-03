"""Unit tests for the DSV4 (NPU) cache layer-split ownership plan.

The shard plan maps a contiguous per-rank layer range onto the DSV4 pool's
compression-ratio buckets (swa / c4 / c128). Each rank's owned set inside a
bucket must be a contiguous sub-range of that bucket's buffer list so the PD
transfer can slice the decode-side buffer lists positionally.
"""

import unittest

from sglang.srt.hardware_backend.npu.dsv4.dsv4_layer_split_plan import (
    DSV4LayerShardPlan,
    owned_bucket_range,
)
from sglang.srt.layers.cp.utils import get_layer_owner, get_layer_shard_range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4LayerSplitShardPlan(CustomTestCase):
    # DSV4-Flash-like pattern: two c4 layers and one c128 layer per 5 layers.
    RATIOS = [4, 0, 4, 0, 128] * 12

    def _plan(self, rank, shard_size, ratios=None):
        ratios = self.RATIOS if ratios is None else ratios
        return DSV4LayerShardPlan(
            rank=rank,
            shard_size=shard_size,
            num_layers=len(ratios),
            stage_start=0,
            ratios=ratios,
        )

    def test_shard_ranges_cover_all_layers_once(self):
        ranges = [self._plan(r, 2).owned_stage_local_range() for r in range(2)]
        covered = [i for start, end in ranges for i in range(start, end)]
        self.assertEqual(covered, list(range(len(self.RATIOS))))

    def test_owner_rank_round_trip(self):
        for shard_size in (2, 3, 4):
            for layer in range(len(self.RATIOS)):
                owners = [
                    r
                    for r in range(shard_size)
                    if self._plan(r, shard_size).is_layer_owned(layer)
                ]
                self.assertEqual(len(owners), 1)
                self.assertEqual(
                    owners[0],
                    self._plan(owners[0], shard_size).owner_rank(layer),
                )

    def test_owned_bucket_slices_partition_each_bucket(self):
        for shard_size in (2, 3, 4):
            for bucket in ("swa", "c4", "c128"):
                bucket_ids = self._plan(0, shard_size).bucket_layer_ids(bucket)
                covered, next_start = [], 0
                for rank in range(shard_size):
                    plan = self._plan(rank, shard_size)
                    start, end = plan.owned_bucket_range(bucket)
                    # Contiguous in bucket index, no gap to the previous rank.
                    self.assertEqual(start, next_start)
                    self.assertLessEqual(end, len(bucket_ids))
                    for idx in range(start, end):
                        self.assertTrue(plan.is_stage_local_owned(bucket_ids[idx]))
                    covered.extend(bucket_ids[start:end])
                    next_start = end
                self.assertEqual(next_start, len(bucket_ids))
                self.assertEqual(sorted(covered), sorted(bucket_ids))

    def test_dense_and_degenerate_ratio_patterns(self):
        for ratios in ([0] * 16, [0, 4, 128] * 8, [4] * 7):
            num_layers = len(ratios)
            for shard_size in (2, 3):
                plans = [
                    DSV4LayerShardPlan(
                        rank=r,
                        shard_size=shard_size,
                        num_layers=num_layers,
                        stage_start=0,
                        ratios=ratios,
                    )
                    for r in range(shard_size)
                ]
                for bucket in ("swa", "c4", "c128"):
                    ids = plans[0].bucket_layer_ids(bucket)
                    covered, next_start = [], 0
                    for plan in plans:
                        start, end = plan.owned_bucket_range(bucket)
                        self.assertEqual(start, next_start)
                        covered.extend(ids[start:end])
                        next_start = end
                    self.assertEqual(next_start, len(ids))
                    self.assertEqual(sorted(covered), sorted(ids))

    def test_stage_offset_shifts_global_shard_window(self):
        plan = DSV4LayerShardPlan(
            rank=1,
            shard_size=2,
            num_layers=60,
            stage_start=8,
            ratios=self.RATIOS,
        )
        self.assertEqual(plan.shard_start, 8 + plan.owned_start)
        self.assertEqual(plan.shard_end, 8 + plan.owned_end)
        self.assertTrue(plan.is_layer_owned(8 + plan.owned_start))
        self.assertFalse(plan.is_layer_owned(8 + plan.owned_start - 1))

    def test_owned_bucket_range_helper_monotonic_ids(self):
        self.assertEqual(owned_bucket_range([2, 4, 6, 8], 4, 8), (1, 3))
        self.assertEqual(owned_bucket_range([2, 4, 6, 8], 0, 100), (0, 4))
        self.assertEqual(owned_bucket_range([2, 4, 6, 8], 5, 5), (2, 2))
        self.assertEqual(owned_bucket_range([], 0, 4), (0, 0))

    def test_matches_cp_utils_shard_math(self):
        self.assertEqual(
            [get_layer_shard_range(r, 4, 10) for r in range(4)],
            [(0, 3), (3, 6), (6, 8), (8, 10)],
        )
        self.assertEqual(
            [get_layer_owner(i, 4, 10) for i in range(10)],
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        )


if __name__ == "__main__":
    unittest.main()
