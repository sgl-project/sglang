"""Validate platform-specific prefill CP policy."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v4_cp
from sglang.srt.arg_groups.parallel_hook import (
    handle_context_parallelism,
    validate_prefill_cp_platform,
)
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPlatformPrefillCPPolicy(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="interleave")

    def test_platform_cp_rejected_before_model_lookup(self):
        for platform in ("is_musa",):
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
        for platform in ("is_musa",):
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
        for platform in ("is_musa",):
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

    @override_platform(is_hip=True, is_npu=False, is_musa=False)
    def test_hip_gate_allows_migrated_path(self):
        args = ServerArgs(
            model_path="missing-model-is-not-loaded-by-this-gate",
            enable_prefill_cp=True,
            cp_strategy="interleave",
        )
        validate_prefill_cp_platform(args)

    @override_platform(is_hip=True, is_npu=False, is_musa=False)
    def test_hip_context_parallel_rejects_other_models(self):
        args = ServerArgs(
            model_path="local-unsupported-model",
            enable_prefill_cp=True,
            cp_strategy="interleave",
        )
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
            is_multimodal=False,
        )

        with self.assertRaisesRegex(ValueError, "only supported.*DeepseekV4"):
            handle_context_parallelism(args)

    @override_platform(is_hip=True, is_npu=False, is_musa=False)
    def test_hip_context_parallel_allows_deepseek_v4(self):
        args = ServerArgs(
            model_path="local-deepseek-v4",
            enable_prefill_cp=True,
            cp_strategy="interleave",
            tp_size=2,
            attn_cp_size=2,
        )
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
            is_multimodal=False,
        )

        handle_context_parallelism(args)

    @override_platform(is_hip=True, is_npu=False, is_musa=False)
    def test_hip_deepseek_v4_requires_dsv4_for_both_phases(self):
        args = ServerArgs(
            model_path="local-deepseek-v4",
            enable_prefill_cp=True,
            cp_strategy="interleave",
            tp_size=2,
            attention_backend="dsv4",
            prefill_attention_backend="flashinfer",
        )

        with self.assertRaisesRegex(ValueError, "dsv4.*both phases"):
            validate_deepseek_v4_cp(args)

    @override_platform(is_hip=True, is_npu=False, is_musa=False)
    def test_hip_deepseek_v4_rejects_multinode_cp(self):
        args = ServerArgs(
            model_path="local-deepseek-v4",
            enable_prefill_cp=True,
            cp_strategy="interleave",
            tp_size=2,
            nnodes=2,
            attention_backend="dsv4",
        )

        with self.assertRaisesRegex(AssertionError, "only supports one node"):
            validate_deepseek_v4_cp(args)

    def test_non_cp_and_decode_cp_are_not_rejected(self):
        for platform in ("is_hip", "is_npu", "is_musa"):
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


if __name__ == "__main__":
    unittest.main()
