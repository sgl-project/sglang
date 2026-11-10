#!/usr/bin/env python3

import unittest

from nightly_skip_policy import allowed_skips, deselected_nodes


class TestNightlySkipPolicy(unittest.TestCase):
    def test_exact_policy_is_bounded_and_unique(self):
        skips = allowed_skips()
        self.assertEqual(len(skips), 1211)
        self.assertEqual(len(set(skips)), len(skips))
        self.assertEqual(
            skips[0],
            "registered.kernels.ops.attention.test_rope.test_partial_rope[64-80-False-1]",
        )
        self.assertEqual(
            skips[-1],
            "registered.kernels.ops.diffusion.test_timestep_embedding.test_timestep_embedding_perf",
        )

    def test_baseline_conditional_skips(self):
        skips = allowed_skips()
        prefixes = (
            "registered.layers.mamba.test_mamba_ssm.",
            "registered.profiling.test_start_profile.",
            "registered.sampling.test_penalty.",
            "registered.unit.managers.test_hisparse_unit.",
            "registered.lora.test_virtual_experts_kernels.",
            "registered.unit.model_loader.test_modelopt_export.",
            "registered.openai_server.basic.test_openai_server.",
        )
        self.assertEqual(
            sum(node.startswith(prefixes) for node in skips),
            35,
        )
        self.assertIn(
            "registered.layers.mamba.test_mamba_ssm.test_selective_state_update_stochastic_rounding[2048-16-False-0]",
            skips,
        )
        self.assertIn(
            "registered.layers.mamba.test_mamba_ssm.test_selective_state_update_stochastic_rounding[4096-64-True-4]",
            skips,
        )
        self.assertIn(
            "registered.unit.model_loader.test_modelopt_export.TestModelOptExportIntegration.test_full_workflow_with_export",
            skips,
        )
        self.assertIn(
            "registered.openai_server.basic.test_openai_server.TestOpenAICompletionWithRust.test_completion",
            skips,
        )

    def test_large_baseline_inventory_and_legacy_deselect(self):
        skips = allowed_skips()
        prefixes = (
            "registered.radix_cache.test_radix_attention.",
            "registered.unit.mem_cache.test_unified_radix_cache_unittest.",
        )
        self.assertEqual(sum(node.startswith(prefixes) for node in skips), 1031)
        self.assertIn(
            "registered.radix_cache.test_radix_attention.TestRadixCacheLPM.test_radix_attention",
            skips,
        )
        self.assertEqual(
            deselected_nodes(),
            [
                "registered/rl/test_update_weights_from_tensor.py::test_update_weights_from_tensor"
            ],
        )

    def test_rope_policy_only_covers_invalid_head_dimensions(self):
        skips = allowed_skips()
        self.assertEqual(
            sum(
                node.startswith(
                    "registered.kernels.ops.attention.test_rope.test_partial_rope["
                )
                for node in skips
            ),
            144,
        )
        self.assertIn(
            "registered.kernels.ops.attention.test_rope.test_partial_rope[64-128-True-2049]",
            skips,
        )
        self.assertIn(
            "registered.kernels.ops.attention.test_rope.test_partial_rope[64-80-False-20]",
            skips,
        )
        self.assertIn(
            "registered.kernels.ops.attention.test_rope.test_partial_rope[64-80-False-21]",
            skips,
        )
        self.assertNotIn(
            "registered.kernels.ops.attention.test_rope.test_partial_rope[64-80-False-2_0]",
            skips,
        )
        self.assertNotIn(
            "registered.kernels.ops.attention.test_rope.test_partial_rope[128-128-False-1]",
            skips,
        )


if __name__ == "__main__":
    unittest.main()
