"""Unit tests for CP Cache LayerSplit ownership helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
    CpCacheLayerSplitPoolBase,
)
from sglang.srt.mem_cache.cp_cache_layer_split.utils import (
    get_cp_cache_layer_shard_info,
    get_layer_owner,
    get_layer_shard_range,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _OwnershipOnlyPool(CpCacheLayerSplitPoolBase):
    def __init__(self, cp_rank, cp_size, start_layer, layer_num):
        self._init_cp_cache_layer_split(
            cp_rank=cp_rank,
            cp_size=cp_size,
            layer_shard_start_layer=start_layer,
            layer_shard_layer_num=layer_num,
        )


class TestCpCacheLayerSplitOwnership(CustomTestCase):
    def test_balanced_layer_ranges_and_owners(self):
        ranges = [get_layer_shard_range(rank, 4, 10) for rank in range(4)]
        self.assertEqual(ranges, [(0, 3), (3, 6), (6, 8), (8, 10)])
        self.assertEqual(
            [get_layer_owner(i, 4, 10) for i in range(10)],
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        )

        covered = [layer_id for start, end in ranges for layer_id in range(start, end)]
        self.assertEqual(covered, list(range(10)))
        self.assertEqual(
            [get_layer_shard_range(rank, 4, 2) for rank in range(4)],
            [(0, 1), (1, 2), (2, 2), (2, 2)],
        )

    def test_shard_info_respects_pool_selection(self):
        runner = SimpleNamespace(
            is_draft_worker=False,
            server_args=SimpleNamespace(enable_cp_cache_layer_split=True),
        )
        parallel = SimpleNamespace(attn_cp_rank=2, attn_cp_size=4)

        with patch(
            "sglang.srt.mem_cache.cp_cache_layer_split.utils.get_parallel",
            return_value=parallel,
        ):
            self.assertEqual(get_cp_cache_layer_shard_info(runner), (2, 4))
            runner.is_draft_worker = True
            self.assertEqual(get_cp_cache_layer_shard_info(runner), (None, 1))
            runner.is_draft_worker = False
            runner.server_args.enable_cp_cache_layer_split = False
            self.assertEqual(get_cp_cache_layer_shard_info(runner), (None, 1))

    def test_invalid_cp_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid cp_rank"):
            _OwnershipOnlyPool(2, 2, 0, 4)

    def test_stage_local_ranges_and_indices(self):
        pools = [_OwnershipOnlyPool(rank, 4, 20, 20) for rank in range(4)]
        ranges = [pool._owned_global_layer_range() for pool in pools]
        self.assertEqual(ranges, [(20, 25), (25, 30), (30, 35), (35, 40)])

        covered = [layer_id for start, end in ranges for layer_id in range(start, end)]
        self.assertEqual(covered, list(range(20, 40)))
        mapping = pools[2]._build_owned_layer_local_index_map()

        self.assertEqual(set(mapping), set(range(30, 35)))
        self.assertEqual(mapping[30], 0)
        self.assertEqual(mapping[34], 4)

    def test_layer_outside_stage_is_rejected(self):
        pool = _OwnershipOnlyPool(0, 2, 10, 4)

        with self.assertRaisesRegex(ValueError, "outside Cache LayerSplit stage"):
            pool._get_layer_owner_rank(9)


if __name__ == "__main__":
    unittest.main()
