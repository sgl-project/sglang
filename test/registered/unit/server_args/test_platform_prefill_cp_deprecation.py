"""Reject deprecated platform CP before model loading or topology setup."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.arg_groups.parallel_hook as parallel_hook
from sglang.srt.arg_groups.parallel_hook import (
    handle_context_parallelism,
    validate_prefill_cp_platform,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPlatformPrefillCPDeprecation(CustomTestCase):
    def test_platform_cp_rejected_before_model_lookup(self):
        for platform in ("is_hip", "is_musa"):
            facts = dict(is_hip=False, is_npu=False, is_musa=False)
            facts[platform] = True
            for strategy in (None, "zigzag", "interleave"):
                with self.subTest(platform=platform, strategy=strategy):
                    with override_platform(**facts):
                        args = ServerArgs(
                            model_path="missing-model-must-not-be-loaded",
                            enable_prefill_cp=True,
                            cp_strategy=strategy,
                        )
                        with self.assertRaisesRegex(ValueError, "deprecated.*refactor"):
                            validate_prefill_cp_platform(args)

    def test_context_parallel_handler_rejects_before_model_lookup(self):
        for platform in ("is_hip", "is_musa"):
            facts = dict(is_hip=False, is_npu=False, is_musa=False)
            facts[platform] = True
            with self.subTest(platform=platform), override_platform(**facts):
                args = ServerArgs(
                    model_path="missing-model-must-not-be-loaded",
                    enable_prefill_cp=True,
                    cp_strategy="interleave",
                )
                with self.assertRaisesRegex(ValueError, "deprecated.*refactor"):
                    handle_context_parallelism(args)

    def test_resolution_rejects_even_dummy_models(self):
        for platform in ("is_hip", "is_musa"):
            facts = dict(is_hip=False, is_npu=False, is_musa=False)
            facts[platform] = True
            for model_path in ("dummy", "none", "missing-model-must-not-be-loaded"):
                with self.subTest(platform=platform, model_path=model_path):
                    with override_platform(**facts):
                        args = ServerArgs(
                            model_path=model_path,
                            enable_prefill_cp=True,
                            cp_strategy="interleave",
                        )
                        with self.assertRaisesRegex(ValueError, "deprecated.*refactor"):
                            args.resolve_once()

    def test_non_cp_and_decode_cp_are_not_rejected(self):
        for platform in ("is_hip", "is_musa"):
            facts = dict(is_hip=False, is_npu=False, is_musa=False)
            facts[platform] = True
            for dcp_size in (1, 2):
                with self.subTest(platform=platform, dcp_size=dcp_size):
                    with override_platform(**facts):
                        args = ServerArgs(model_path="dummy", dcp_size=dcp_size)
                        validate_prefill_cp_platform(args)

    @override_platform(is_hip=False, is_npu=False, is_musa=False)
    def test_generic_cp_is_not_rejected_or_modified(self):
        for strategy in ("zigzag", "interleave"):
            with self.subTest(strategy=strategy):
                args = ServerArgs(
                    model_path="dummy", enable_prefill_cp=True, cp_strategy=strategy
                )
                validate_prefill_cp_platform(args)
                self.assertTrue(args.enable_prefill_cp)
                self.assertEqual(args.cp_strategy, strategy)

    def test_dsv4_cp_platform_gates(self):
        # Guards for the NPU prefill-CP gates: a future edit that re-allows
        # CUDA zigzag, drops the single-machine tp cap on NPU (inter-node CP
        # all-gathers bf16 KV and silently degrades), or re-admits an a2a=none
        # combo that only crashes (dp>1) or silently mis-routes (zigzag) at
        # runtime, must turn this red.
        from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v4_cp

        cuda = dict(is_hip=False, is_npu=False, is_musa=False)
        npu = dict(is_hip=False, is_npu=True, is_musa=False)
        with override_platform(**cuda):
            args = ServerArgs(
                model_path="dummy",
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                moe_a2a_backend="none",
            )
            with self.assertRaisesRegex(ValueError, "zigzag CP requires the NPU"):
                validate_deepseek_v4_cp(args)
        with override_platform(**npu):
            args = ServerArgs(
                model_path="dummy",
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                moe_a2a_backend="none",
            )
            with self.assertRaisesRegex(
                ValueError, "zigzag CP requires an MoE a2a backend"
            ):
                validate_deepseek_v4_cp(args)
            args = ServerArgs(
                model_path="dummy",
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                tp_size=8,
                moe_a2a_backend="deepep",
            )
            validate_deepseek_v4_cp(args)  # NPU zigzag with a2a is legal
            args = ServerArgs(
                model_path="dummy",
                enable_prefill_cp=True,
                cp_strategy="interleave",
                tp_size=16,
                moe_a2a_backend="none",
            )
            with self.assertRaises(AssertionError):
                validate_deepseek_v4_cp(args)
            args = ServerArgs(
                model_path="dummy",
                enable_prefill_cp=True,
                cp_strategy="interleave",
                tp_size=8,
                dp_size=2,
                moe_a2a_backend="none",
            )
            with self.assertRaisesRegex(
                ValueError, "dp_size > 1 requires an MoE a2a backend"
            ):
                validate_deepseek_v4_cp(args)
            args = ServerArgs(
                model_path="dummy",
                enable_prefill_cp=True,
                cp_strategy="interleave",
                tp_size=8,
                dp_size=2,
                moe_a2a_backend="deepep",
            )
            validate_deepseek_v4_cp(args)  # dp>1 with a2a is legal

    def test_npu_prefill_cp_rejects_non_dsv4_architectures(self):
        # The NPU CP path is adapted for DeepSeek V4 only: before this gate,
        # V3.2/other archs passed validation and died mid-forward on the
        # deprecated CP shims (rebuild_cp_kv_cache has no NPU definition).
        npu = dict(is_hip=False, is_npu=True, is_musa=False)
        for arch in ("DeepseekV32ForCausalLM", "Glm4MoeForCausalLM"):
            with self.subTest(arch=arch), override_platform(**npu):
                args = ServerArgs(
                    model_path="dummy",
                    enable_prefill_cp=True,
                    cp_strategy="interleave",
                    moe_a2a_backend="deepep",
                )
                fake_model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=[arch])
                )
                with patch.object(
                    parallel_hook,
                    "model_config_of",
                    return_value=fake_model_config,
                ):
                    with self.assertRaisesRegex(ValueError, "only DeepSeek V4"):
                        handle_context_parallelism(args)


if __name__ == "__main__":
    unittest.main()
