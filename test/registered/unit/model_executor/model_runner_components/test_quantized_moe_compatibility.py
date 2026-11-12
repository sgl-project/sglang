"""Tests for quantized MoE compatibility checks before model construction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner_components import (  # noqa: E402
    moe_ep_setup,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestQuantizedMoeCompatibility(CustomTestCase):
    @staticmethod
    def _model_config(
        *,
        architecture="BailingMoeV3ForCausalLM",
        quant_method="fp8",
        weight_block_size=(128, 128),
        serialized=True,
        moe_intermediate_size=1280,
    ):
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[architecture],
                model_type="bailing_hybrid",
                use_kda=True,
                quantization_config={
                    "quant_method": quant_method,
                    "weight_block_size": list(weight_block_size),
                    "is_checkpoint_fp8_serialized": serialized,
                },
            ),
            hf_text_config=SimpleNamespace(moe_intermediate_size=moe_intermediate_size),
            is_fp4_experts=False,
        )

    def _check(
        self,
        *,
        model_overrides=None,
        moe_runner_backend="triton",
        moe_a2a_backend="none",
        enable_jit_deepgemm=False,
        enable_eplb=False,
        cpu_offload_gb=0,
        offload_group_size=-1,
        tp_size=8,
        moe_ep_size=1,
        moe_dp_size=1,
    ):
        model_config = self._model_config(**(model_overrides or {}))
        exec_config = SimpleNamespace(
            moe=SimpleNamespace(
                moe_runner_backend=moe_runner_backend,
                moe_a2a_backend=moe_a2a_backend,
                enable_eplb=enable_eplb,
                ep_num_redundant_experts=0,
                init_expert_location="trivial",
            ),
            offload=SimpleNamespace(
                cpu_offload_gb=cpu_offload_gb,
                offload_group_size=offload_group_size,
            ),
        )
        with (
            patch.object(moe_ep_setup, "get_exec", return_value=exec_config),
            patch.object(moe_ep_setup, "_use_aiter", False),
            patch.object(
                moe_ep_setup.deep_gemm_wrapper,
                "ENABLE_JIT_DEEPGEMM",
                enable_jit_deepgemm,
            ),
            patch.object(
                moe_ep_setup.envs.SGLANG_SHARED_EXPERT_TP1,
                "get",
                return_value=False,
            ),
        ):
            moe_ep_setup.check_quantized_moe_compatibility(
                model_config=model_config,
                tp_size=tp_size,
                moe_ep_size=moe_ep_size,
                moe_dp_size=moe_dp_size,
            )

    def test_allows_bailing_native_fp8_tail_for_serialized_checkpoint(self):
        self._check()

    def test_allows_bailing_nextn_native_fp8_tail(self):
        self._check(model_overrides={"architecture": "BailingMoeForCausalLMNextN"})

    def test_allows_auto_only_when_it_resolves_to_triton(self):
        self._check(moe_runner_backend="auto")

        for enable_jit_deepgemm in (False, True):
            with self.subTest(enable_jit_deepgemm=enable_jit_deepgemm):
                with self.assertRaisesRegex(ValueError, "For quantized MoE models"):
                    self._check(
                        moe_runner_backend="auto",
                        moe_a2a_backend="deepep",
                        enable_jit_deepgemm=enable_jit_deepgemm,
                    )

    def test_rejects_unsupported_partial_block_combinations(self):
        cases = (
            (
                "other architecture",
                {"model_overrides": {"architecture": "DeepseekV3ForCausalLM"}},
            ),
            ("other quantization", {"model_overrides": {"quant_method": "mxfp8"}}),
            ("not serialized", {"model_overrides": {"serialized": False}}),
            (
                "other block size",
                {"model_overrides": {"weight_block_size": (64, 64)}},
            ),
            ("cutlass", {"moe_runner_backend": "cutlass"}),
            ("deepgemm", {"moe_runner_backend": "deep_gemm"}),
            (
                "a2a",
                {
                    "moe_runner_backend": "triton",
                    "moe_a2a_backend": "deepep",
                },
            ),
            ("eplb", {"enable_eplb": True}),
            ("offload", {"cpu_offload_gb": 1}),
            ("group offload", {"offload_group_size": 1}),
            ("expert parallel", {"moe_ep_size": 2}),
            ("moe data parallel", {"moe_dp_size": 2}),
        )
        for name, kwargs in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "For quantized MoE models"):
                    self._check(**kwargs)

    def test_aligned_shape_remains_supported_for_other_models(self):
        self._check(
            model_overrides={
                "architecture": "DeepseekV3ForCausalLM",
                "moe_intermediate_size": 1024,
            },
            moe_runner_backend="cutlass",
        )


if __name__ == "__main__":
    unittest.main()
