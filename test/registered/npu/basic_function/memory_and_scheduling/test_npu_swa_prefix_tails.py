"""Cover existing DeepSeek-V4.1 SWA prefix-tail sizing on the NPU path.

These configuration checks run in the CPU-hosted NPU suite without loading
weights or NPU operators.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.arg_groups.model_overrides.deepseek_v4 import _deepseek_v4_overrides
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.model_executor.pool_configurator import (
    create_memory_pool_configurator,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=5, suite="base-a-test-npu")


class TestNpuDsv41SwaPrefixTails(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)

    def _resolved_args(self, prefix_tails=None, ratio=None, **kwargs):
        args = ServerArgs(
            model_path="dummy",
            device="npu",
            moe_runner_backend="triton",
            page_size=128,
            max_running_requests=4,
            chunked_prefill_size=512,
            swa_prefix_tails=prefix_tails,
            swa_full_tokens_ratio=ratio,
            **kwargs,
        )
        args.resolve_once()
        hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v41"
        )
        declare_resolution(
            args,
            "_handle_model_specific_adjustments",
            **_deepseek_v4_overrides(args, hf_config),
        )
        handle_cache_compatibility(args)
        return args

    def _pool_sizes(self, prefix_tails=None, ratio=None, **kwargs):
        args = self._resolved_args(prefix_tails, ratio, **kwargs)
        publish(args, role="test")
        ratios = [0, 2, 1]
        config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                model_type="deepseek_v41",
                kv_source_layer_ids=[],
            ),
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            index_head_dim=128,
            context_len=4096,
            window_size=128,
            compress_ratios=ratios,
            mla_layer_ids=[],
        )
        kvc = SimpleNamespace(
            model_config=config,
            kv_cache_dtype_str="bfloat16",
            is_hybrid_swa=True,
            sliding_window_size=128,
            page_size=128,
            attn_dp_size=1,
            pp_size=1,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            layer_info=SimpleNamespace(start_layer=0, end_layer=len(ratios)),
        )
        with patch("sglang.srt.model_executor.pool_configurator._is_npu", True):
            planner = create_memory_pool_configurator(kvc)
            sizes = planner.calculate_pool_sizes_from_max_tokens(65536, 128)
        return planner, sizes

    def test_default_uses_cap_sizing(self):
        planner, sizes = self._pool_sizes()
        self.assertIsNone(planner.operator_swa_ratio)
        self.assertIsNotNone(planner.swa_cap_tokens)
        self.assertEqual(planner.swa_prefix_tails, 4 * 4)
        self.assertEqual(sizes.swa_max_total_num_tokens, planner.swa_cap_tokens)

    def test_prefix_tails_reserve_swa_capacity(self):
        _, base_sizes = self._pool_sizes(prefix_tails=0)
        for tails in (8, 128):
            with self.subTest(prefix_tails=tails):
                planner, sizes = self._pool_sizes(prefix_tails=tails)
                self.assertIsNone(planner.operator_swa_ratio)
                self.assertIsNotNone(planner.swa_cap_tokens)
                self.assertEqual(planner.swa_prefix_tails, tails)
                self.assertEqual(
                    sizes.swa_max_total_num_tokens
                    - base_sizes.swa_max_total_num_tokens,
                    tails * (128 + 128),
                )
                self.assertEqual(sizes.full_max_total_num_tokens, 65536)

    def test_explicit_ratio_takes_priority(self):
        for ratio, expected_tokens in ((0.1, 6528), (0.25, 16384), (1.0, 65536)):
            for tails in (0, 8):
                with self.subTest(ratio=ratio, prefix_tails=tails):
                    planner, sizes = self._pool_sizes(prefix_tails=tails, ratio=ratio)
                    self.assertEqual(planner.operator_swa_ratio, ratio)
                    self.assertIsNone(planner.swa_cap_tokens)
                    self.assertEqual(sizes.swa_max_total_num_tokens, expected_tokens)

    def test_radix_disabled_default_has_no_prefix_headroom(self):
        _, base_sizes = self._pool_sizes(prefix_tails=0)
        planner, sizes = self._pool_sizes(disable_radix_cache=True)
        self.assertEqual(planner.swa_prefix_tails, 0)
        self.assertIsNotNone(planner.swa_cap_tokens)
        self.assertEqual(
            sizes.swa_max_total_num_tokens, base_sizes.swa_max_total_num_tokens
        )

    def test_explicit_tails_override_radix_disabled_default(self):
        _, base_sizes = self._pool_sizes(disable_radix_cache=True)
        planner, sizes = self._pool_sizes(prefix_tails=8, disable_radix_cache=True)
        self.assertEqual(planner.swa_prefix_tails, 8)
        self.assertEqual(
            sizes.swa_max_total_num_tokens - base_sizes.swa_max_total_num_tokens,
            8 * (128 + 128),
        )

    def test_invalid_explicit_ratios_are_rejected(self):
        for ratio in (0.0, -0.1, 1.01):
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(ValueError, "swa-full-tokens-ratio"):
                    self._resolved_args(ratio=ratio)

    def test_negative_prefix_tails_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "swa-prefix-tails"):
            self._resolved_args(prefix_tails=-1)


if __name__ == "__main__":
    unittest.main()
