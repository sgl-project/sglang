"""PD Host capacity and slim checkpoint validation regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.speculative_hook import validate_typed_dflash_target_storage
from sglang.srt.disaggregation.draft_lane_runtime import validate_typed_target_pool
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.dflash_context import validate_context_encoder_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class PDCapacityTests(CustomTestCase):
    def test_raw_target_format_is_rejected_before_registration(self):
        cfg = SimpleNamespace(
            dsa_prefill_backend="flashmla_kv",
            dsa_decode_backend="flashmla_kv",
            kv_cache_dtype="fp8_e4m3",
        )
        validate_typed_dflash_target_storage(cfg)
        cfg.dsa_prefill_backend = "trtllm"
        with self.assertRaisesRegex(ValueError, "raw576"):
            validate_typed_dflash_target_storage(cfg)
        pool = SimpleNamespace(
            use_dsa=True,
            dtype=torch.float8_e4m3fn,
            dsa_kv_cache_store_fp8=True,
            kv_cache_dim=656,
            kv_scale_ue8m0=False,
        )
        validate_typed_target_pool(pool)
        for field, value in [("kv_cache_dim", 576), ("kv_scale_ue8m0", True)]:
            bad = SimpleNamespace(**vars(pool))
            setattr(bad, field, value)
            with self.assertRaisesRegex(ValueError, "scaled656"):
                validate_typed_target_pool(bad)

    def test_only_target_hisparse_decode_advertises_host_capacity(self):
        for role, draft, sparse, expected in [
            ("decode", False, True, 45000),
            ("decode", True, True, 24640),
            ("decode", False, False, 24640),
            ("prefill", False, True, 24640),
            ("null", False, True, 24640),
        ]:
            runner = SimpleNamespace(
                is_draft_worker=draft,
                enable_hisparse=sparse,
                max_total_num_tokens=24640,
                kv_cache_configurator=SimpleNamespace(
                    logical_token_capacity=lambda **k: 24640
                ),
                token_to_kv_pool_allocator=SimpleNamespace(size_full=49280),
                req_to_token_pool=SimpleNamespace(
                    schedulable_token_capacity=lambda n: min(n, 45000)
                ),
            )
            with patch(
                "sglang.srt.model_executor.model_runner.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=role),
            ):
                self.assertEqual(
                    ModelRunner.logical_max_total_num_tokens.fget(runner), expected
                )

    def test_reserved_spec_decode_does_not_require_more_physical_rows(self):
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.logical_attn_allocator = SimpleNamespace(available_size=lambda: 128)
        allocator.hisparse_attn_allocator = SimpleNamespace(available_size=lambda: 0)
        for role, scratch, expected in [
            ("decode", 20480, True),
            ("decode", 0, False),
            ("null", 20480, False),
            ("prefill", 20480, False),
        ]:
            allocator._spec_scratch_capacity = scratch
            with patch(
                "sglang.srt.runtime_context.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=role),
            ):
                self.assertEqual(
                    allocator.check_decode_capacity(num_tokens=64, tree_cache=None),
                    expected,
                )
                self.assertFalse(
                    allocator.check_decode_capacity(num_tokens=256, tree_cache=None)
                )

    def test_unquantized_draft_checkpoint_is_distinct_from_target_policy(self):
        config = SimpleNamespace(
            model_type="qwen3", architectures=["DFlash2DraftModel"], num_hidden_layers=5
        )
        validate_context_encoder_config(config)
        config.quantization_config = {"quant_method": "modelopt"}
        with self.assertRaisesRegex(ValueError, "unquantized"):
            validate_context_encoder_config(config)


if __name__ == "__main__":
    unittest.main()
