"""Regression tests for FP8 MoE runner configuration ownership."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFp8MoERunnerConfig(CustomTestCase):
    def test_delegate_skips_trtllm_activation_params(self):
        method = object.__new__(Fp8MoEMethod)
        method.block_quant = True
        method._owns_moe_runner = False
        layer = SimpleNamespace(
            num_local_experts=1,
            w13_weight=torch.empty(1),
        )
        backend = SimpleNamespace(
            is_flashinfer_trtllm=lambda: True,
            is_flashinfer_trtllm_routed=lambda: False,
            is_hpc_ops=lambda: False,
        )

        with patch.object(
            method, "process_weights_after_loading_block_quant"
        ) as process, patch(
            "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
            return_value=backend,
        ):
            method.process_weights_after_loading(layer)

        process.assert_called_once_with(layer)
        self.assertFalse(hasattr(layer, "_flashinfer_trtllm_gemm1_alpha"))

    def test_owned_runner_prepares_trtllm_activation_params(self):
        method = object.__new__(Fp8MoEMethod)
        method._owns_moe_runner = True
        method.moe_runner_config = SimpleNamespace(
            gemm1_alpha=1.5,
            gemm1_beta=0.25,
            gemm1_clamp_limit=None,
        )
        layer = SimpleNamespace(
            num_local_experts=2,
            w13_weight=torch.empty(2, 4),
        )

        method._prepare_flashinfer_trtllm_activation_params(layer)

        self.assertTrue(
            torch.equal(
                layer._flashinfer_trtllm_gemm1_alpha,
                torch.full((2,), 1.5),
            )
        )
        self.assertTrue(
            torch.equal(
                layer._flashinfer_trtllm_gemm1_beta,
                torch.full((2,), 0.25),
            )
        )
        self.assertIsNone(layer._flashinfer_trtllm_gemm1_clamp_limit)


if __name__ == "__main__":
    unittest.main()