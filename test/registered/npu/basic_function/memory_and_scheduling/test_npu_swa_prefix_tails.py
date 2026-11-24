"""NPU SWA prefix-tail sizing preserves model defaults and explicit ratios.

V4.1 already supports cap sizing when the ratio is omitted. V4's injected
default must not disable the same sizing path. These configuration checks
run in the CPU-hosted NPU suite without loading weights or NPU operators.
"""

import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.arg_groups.model_overrides.deepseek_v4 import _deepseek_v4_overrides
from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.model_executor.pool_configurator import (
    create_memory_pool_configurator,
)
from sglang.srt.runtime_context import get_schedule, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=5, suite="base-a-test-npu")


class TestNpuSwaPrefixTails(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)

    def _record(self, ratio=None, device="npu", **kwargs):
        args = ServerArgs(
            model_path="dummy",
            device=device,
            moe_runner_backend="triton",
            swa_full_tokens_ratio=ratio,
            **kwargs,
        )
        args.resolve_once()
        return args

    def _dsv4_defaults(self, args, model_type="deepseek_v4"):
        hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"], model_type=model_type
        )
        declare_resolution(
            args,
            "_handle_model_specific_adjustments",
            **_deepseek_v4_overrides(args, hf_config),
        )

    def _check_published(self, args, explicit, ratio):
        self.assertIs(
            resolution_result(args, "_swa_full_tokens_ratio_explicitly_set"),
            explicit,
        )
        publish(args, role="test")
        self.assertIs(get_schedule()._swa_full_tokens_ratio_explicitly_set, explicit)
        self.assertEqual(get_schedule().swa_full_tokens_ratio, ratio)

    def _pool_sizes(self, model_type, prefix_tails, ratio=None):
        args = self._record(
            ratio=ratio,
            page_size=128,
            max_running_requests=4,
            chunked_prefill_size=512,
            swa_prefix_tails=prefix_tails,
        )
        self._dsv4_defaults(args, model_type=model_type)
        handle_cache_compatibility(args)
        publish(args, role="test")
        ratios = [0, 2, 1] if model_type == "deepseek_v41" else [4, 128]
        config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                model_type=model_type,
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

    def test_dsv4_automatic_ratio_is_not_explicit(self):
        for device in ("cuda", "npu"):
            with self.subTest(device=device):
                args = self._record(device=device)
                self._dsv4_defaults(args)
                handle_cache_compatibility(args)
                self.assertIsNone(args._raw_input["swa_full_tokens_ratio"])
                self._check_published(args, explicit=False, ratio=0.1)

    def test_dsv41_default_ratio_remains_unset(self):
        args = self._record()
        self._dsv4_defaults(args, model_type="deepseek_v41")
        handle_cache_compatibility(args)
        self.assertIsNone(args._raw_input["swa_full_tokens_ratio"])
        self._check_published(args, explicit=False, ratio=0.8)

    def test_prefix_tails_increase_swa_capacity_for_v4_and_v41(self):
        for model_type in ("deepseek_v4", "deepseek_v41"):
            with self.subTest(model_type=model_type):
                base, base_sizes = self._pool_sizes(model_type, prefix_tails=0)
                self.assertIsNone(base.operator_swa_ratio)
                self.assertIsNotNone(base.swa_cap_tokens)
                for tails in (8, 128):
                    with self.subTest(prefix_tails=tails):
                        planner, sizes = self._pool_sizes(model_type, tails)
                        self.assertIsNone(planner.operator_swa_ratio)
                        self.assertEqual(planner.swa_prefix_tails, tails)
                        self.assertEqual(
                            sizes.swa_max_total_num_tokens
                            - base_sizes.swa_max_total_num_tokens,
                            tails * (128 + 128),
                        )
                        self.assertEqual(sizes.full_max_total_num_tokens, 65536)

    def test_explicit_ratio_selects_ratio_sizing_for_v4_and_v41(self):
        for model_type in ("deepseek_v4", "deepseek_v41"):
            for tails in (0, 8):
                with self.subTest(model_type=model_type, prefix_tails=tails):
                    planner, sizes = self._pool_sizes(model_type, tails, ratio=0.1)
                    self.assertEqual(planner.operator_swa_ratio, 0.1)
                    self.assertIsNone(planner.swa_cap_tokens)
                    self.assertEqual(sizes.swa_max_total_num_tokens, 6528)

    def test_dsv4_explicit_ratio_is_preserved_including_default_value(self):
        for device in ("cuda", "npu"):
            for ratio in (0.1, 0.25, 1.0):
                with self.subTest(device=device, ratio=ratio):
                    args = self._record(ratio=ratio, device=device)
                    self._dsv4_defaults(args)
                    handle_cache_compatibility(args)
                    self._check_published(args, explicit=True, ratio=ratio)

    def test_generic_fallback_is_not_explicit(self):
        args = self._record()
        handle_cache_compatibility(args)
        self._check_published(args, explicit=False, ratio=0.8)

    def test_model_reset_keeps_the_original_user_intent(self):
        args = self._record(ratio=0.25)
        declare_resolution(args, "model_reset", swa_full_tokens_ratio=1.0)
        handle_cache_compatibility(args)
        self.assertEqual(args._raw_input["swa_full_tokens_ratio"], 0.25)
        self._check_published(args, explicit=True, ratio=1.0)

    def test_effective_ratio_validation_is_preserved(self):
        for ratio in (0.0, -0.1, 1.01):
            with self.subTest(ratio=ratio):
                args = self._record()
                declare_resolution(
                    args, "invalid_model_default", swa_full_tokens_ratio=ratio
                )
                with self.assertRaisesRegex(ValueError, "swa-full-tokens-ratio"):
                    handle_cache_compatibility(args)

    def test_explicit_invalid_ratio_is_rejected(self):
        for ratio in (0.0, -0.1, 1.01):
            with self.subTest(ratio=ratio):
                args = self._record(ratio=ratio)
                with self.assertRaisesRegex(ValueError, "swa-full-tokens-ratio"):
                    handle_cache_compatibility(args)

    def test_negative_prefix_tails_is_rejected(self):
        args = self._record(swa_prefix_tails=-1)
        with self.assertRaisesRegex(ValueError, "swa-prefix-tails"):
            handle_cache_compatibility(args)

    def test_origin_survives_serialization_and_publish(self):
        for ratio in (None, 0.1):
            with self.subTest(ratio=ratio):
                args = self._record(ratio=ratio)
                self._dsv4_defaults(args)
                handle_cache_compatibility(args)
                restored = pickle.loads(pickle.dumps(args))
                self.assertEqual(restored._raw_input["swa_full_tokens_ratio"], ratio)
                self._check_published(restored, explicit=ratio is not None, ratio=0.1)


if __name__ == "__main__":
    unittest.main()
