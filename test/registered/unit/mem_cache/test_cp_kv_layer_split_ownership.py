"""Unit tests for CP KV LayerSplit ownership helpers."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.cp_kv_layer_split import (
    build_owned_layer_local_index_map,
    kv_layer_owner,
    layers_per_cp_rank,
    num_owned_kv_layers,
    owned_kv_layer_range,
    owns_kv_layer,
    should_use_cp_kv_layer_split_pool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCpKvLayerSplitOwnership(CustomTestCase):
    def test_owner_boundaries_for_balanced_contiguous_blocks(self):
        self.assertEqual(layers_per_cp_rank(60, 4), 15)
        self.assertEqual(layers_per_cp_rank(61, 4), 16)

        self.assertEqual(kv_layer_owner(0, 4, 60), 0)
        self.assertEqual(kv_layer_owner(14, 4, 60), 0)
        self.assertEqual(kv_layer_owner(15, 4, 60), 1)
        self.assertEqual(kv_layer_owner(59, 4, 60), 3)

        self.assertEqual(kv_layer_owner(0, 4, 10), 0)
        self.assertEqual(kv_layer_owner(2, 4, 10), 0)
        self.assertEqual(kv_layer_owner(3, 4, 10), 1)
        self.assertEqual(kv_layer_owner(5, 4, 10), 1)
        self.assertEqual(kv_layer_owner(6, 4, 10), 2)
        self.assertEqual(kv_layer_owner(7, 4, 10), 2)
        self.assertEqual(kv_layer_owner(8, 4, 10), 3)
        self.assertEqual(kv_layer_owner(9, 4, 10), 3)

    def test_owned_range_respects_pipeline_stage_slice(self):
        # GPU runs layers [20, 40); CP rank 1 owns global KV layers [15, 30).
        self.assertFalse(owns_kv_layer(20, 0, 4, 60))
        self.assertEqual(owned_kv_layer_range(1, 4, 60, 20, 40), (20, 30))
        self.assertEqual(num_owned_kv_layers(1, 4, 60, 20, 40), 10)

    def test_uneven_model_layers_are_accounted_once(self):
        counts = [num_owned_kv_layers(r, 4, 61, 0, 61) for r in range(4)]
        self.assertEqual(counts, [16, 15, 15, 15])
        self.assertEqual(sum(counts), 61)

        counts = [num_owned_kv_layers(r, 4, 10, 0, 10) for r in range(4)]
        self.assertEqual(counts, [3, 3, 2, 2])
        self.assertEqual(sum(counts), 10)

    def test_more_cp_ranks_than_layers_leave_tail_ranks_empty(self):
        counts = [num_owned_kv_layers(r, 4, 2, 0, 2) for r in range(4)]
        self.assertEqual(counts, [1, 1, 0, 0])

        self.assertEqual(owned_kv_layer_range(2, 4, 2, 0, 2), (0, 0))
        self.assertEqual(build_owned_layer_local_index_map(2, 4, 2, 0, 2), {})

    def test_owned_layer_local_index_map_respects_stage_slice(self):
        full = build_owned_layer_local_index_map(1, 4, 60, 0, 60)
        self.assertEqual(len(full), 15)
        self.assertEqual(full[15], 0)
        self.assertEqual(full[29], 14)
        self.assertNotIn(0, full)

        # GPU runs [20, 40); CP rank 2 owns global KV layers [30, 45).
        sliced = build_owned_layer_local_index_map(2, 4, 60, 20, 40)
        self.assertEqual(set(sliced), set(range(30, 40)))
        self.assertEqual(sliced[30], 0)
        self.assertEqual(sliced[39], 9)


class TestCpKvLayerSplitPredicates(CustomTestCase):
    def test_should_use_requires_flag_dsa_cp_and_multi_cp(self):
        args = SimpleNamespace(
            enable_cp_kv_layer_split=True,
            enable_dsa_prefill_context_parallel=True,
            attn_cp_size=4,
        )
        self.assertTrue(should_use_cp_kv_layer_split_pool(args))

        args.enable_cp_kv_layer_split = False
        self.assertFalse(should_use_cp_kv_layer_split_pool(args))

        args.enable_cp_kv_layer_split = True
        args.enable_dsa_prefill_context_parallel = False
        self.assertFalse(should_use_cp_kv_layer_split_pool(args))

        args.enable_dsa_prefill_context_parallel = True
        args.attn_cp_size = 1
        self.assertFalse(should_use_cp_kv_layer_split_pool(args))


if __name__ == "__main__":
    unittest.main()
