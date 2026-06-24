"""Unit tests for CP KV LayerSplit ownership helpers."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.cp_kv_layer_split import (
    build_owned_layer_local_index_map,
    kv_layer_owner,
    layers_per_cp_rank,
    num_owned_compress_layers,
    num_owned_kv_layers,
    owned_kv_layer_range,
    owns_kv_layer,
    should_use_cp_kv_layer_split_pool,
    validate_cp_kv_layer_split_model_arch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCpKvLayerSplitOwnership(CustomTestCase):
    def test_owner_boundaries_for_contiguous_blocks(self):
        self.assertEqual(layers_per_cp_rank(60, 4), 15)
        self.assertEqual(layers_per_cp_rank(61, 4), 16)

        self.assertEqual(kv_layer_owner(0, 4, 60), 0)
        self.assertEqual(kv_layer_owner(14, 4, 60), 0)
        self.assertEqual(kv_layer_owner(15, 4, 60), 1)
        self.assertEqual(kv_layer_owner(59, 4, 60), 3)

    def test_owned_range_respects_pipeline_stage_slice(self):
        # GPU runs layers [20, 40); CP rank 1 owns global KV layers [15, 30).
        self.assertFalse(owns_kv_layer(20, 0, 4, 60))
        self.assertEqual(owned_kv_layer_range(1, 4, 60, 20, 40), (20, 30))
        self.assertEqual(num_owned_kv_layers(1, 4, 60, 20, 40), 10)

    def test_uneven_model_layers_are_accounted_once(self):
        counts = [num_owned_kv_layers(r, 4, 61, 0, 61) for r in range(4)]
        self.assertEqual(counts, [16, 16, 16, 13])
        self.assertEqual(sum(counts), 61)

    def test_owned_compress_layers_count_c4_and_c128(self):
        ratios = [0, 4, 128, 4] * 15
        self.assertEqual(
            num_owned_compress_layers(0, 4, 60, 0, 60, ratios, 4),
            sum(1 for layer_id in range(0, 15) if ratios[layer_id] == 4),
        )
        self.assertEqual(
            num_owned_compress_layers(0, 4, 60, 0, 60, ratios, 128),
            sum(1 for layer_id in range(0, 15) if ratios[layer_id] == 128),
        )
        self.assertEqual(
            sum(
                num_owned_compress_layers(r, 4, 60, 0, 60, ratios, 4) for r in range(4)
            ),
            sum(1 for r in ratios if r == 4),
        )
        self.assertEqual(
            sum(
                num_owned_compress_layers(r, 4, 60, 0, 60, ratios, 128)
                for r in range(4)
            ),
            sum(1 for r in ratios if r == 128),
        )

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

    def test_model_arch_guard_is_noop_when_flag_off(self):
        args = SimpleNamespace(enable_cp_kv_layer_split=False)
        validate_cp_kv_layer_split_model_arch(args, "DeepseekV32ForCausalLM")

    def test_model_arch_guard_rejects_unsupported_arch(self):
        args = SimpleNamespace(enable_cp_kv_layer_split=True)
        with self.assertRaisesRegex(ValueError, "not supported for model arch"):
            validate_cp_kv_layer_split_model_arch(args, "DeepseekV32ForCausalLM")

    def test_model_arch_guard_accepts_deepseek_v4_arches(self):
        args = SimpleNamespace(enable_cp_kv_layer_split=True)
        validate_cp_kv_layer_split_model_arch(args, "DeepseekV4ForCausalLM")
        validate_cp_kv_layer_split_model_arch(args, "DeepseekV4ForCausalLMNextN")


if __name__ == "__main__":
    unittest.main()
