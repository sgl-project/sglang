"""Unit tests for CP Cache LayerSplit ownership helpers."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.cp.utils import is_cp_cache_layer_split_enabled
from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
    CpCacheLayerSplitPoolBase,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _OwnershipOnlyPool(CpCacheLayerSplitPoolBase):
    def __init__(self, cp_rank, cp_size, start_layer, layer_num):
        super().__init__(
            cp_rank=cp_rank,
            cp_size=cp_size,
            layer_shard_start_layer=start_layer,
            layer_shard_layer_num=layer_num,
        )


class TestCpCacheLayerSplitOwnership(CustomTestCase):
    def test_owner_boundaries_for_balanced_contiguous_blocks(self):
        pool = _OwnershipOnlyPool(0, 4, 0, 10)

        self.assertEqual(
            [pool._get_layer_owner_rank(i) for i in range(10)],
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        )

    def test_owned_ranges_cover_pipeline_stage_slice_once(self):
        pools = [_OwnershipOnlyPool(rank, 4, 20, 20) for rank in range(4)]
        ranges = [pool._owned_global_layer_range() for pool in pools]
        self.assertEqual(ranges, [(20, 25), (25, 30), (30, 35), (35, 40)])

        covered = [layer_id for start, end in ranges for layer_id in range(start, end)]
        self.assertEqual(covered, list(range(20, 40)))
        self.assertEqual(len(covered), len(set(covered)))

    def test_uneven_model_layers_are_accounted_once(self):
        counts = [
            len(_OwnershipOnlyPool(rank, 4, 0, 61)._build_owned_layer_local_index_map())
            for rank in range(4)
        ]
        self.assertEqual(counts, [16, 15, 15, 15])
        self.assertEqual(sum(counts), 61)

        counts = [
            len(_OwnershipOnlyPool(rank, 4, 0, 10)._build_owned_layer_local_index_map())
            for rank in range(4)
        ]
        self.assertEqual(counts, [3, 3, 2, 2])
        self.assertEqual(sum(counts), 10)

    def test_more_cp_ranks_than_layers_leave_tail_ranks_empty(self):
        pools = [_OwnershipOnlyPool(rank, 4, 7, 2) for rank in range(4)]
        counts = [len(pool._build_owned_layer_local_index_map()) for pool in pools]
        self.assertEqual(counts, [1, 1, 0, 0])
        self.assertEqual(pools[2]._owned_global_layer_range(), (9, 9))
        self.assertEqual(pools[2]._build_owned_layer_local_index_map(), {})

    def test_owned_layer_local_index_map_respects_stage_slice(self):
        mapping = _OwnershipOnlyPool(
            cp_rank=2, cp_size=4, start_layer=20, layer_num=20
        )._build_owned_layer_local_index_map()

        self.assertEqual(set(mapping), set(range(30, 35)))
        self.assertEqual(mapping[30], 0)
        self.assertEqual(mapping[34], 4)

    def test_layer_outside_stage_is_rejected(self):
        pool = _OwnershipOnlyPool(0, 2, 10, 4)

        with self.assertRaisesRegex(ValueError, "outside Cache LayerSplit stage"):
            pool._get_layer_owner_rank(9)


class TestCpCacheLayerSplitPredicates(CustomTestCase):
    def test_enabled_uses_canonical_server_arg(self):
        args = SimpleNamespace(enable_cp_cache_layer_split=True)
        self.assertTrue(is_cp_cache_layer_split_enabled(args))

        args.enable_cp_cache_layer_split = False
        self.assertFalse(is_cp_cache_layer_split_enabled(args))


if __name__ == "__main__":
    unittest.main()
