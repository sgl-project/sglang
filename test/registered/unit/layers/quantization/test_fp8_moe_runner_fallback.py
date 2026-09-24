"""Fp8MoEMethod builds a triton runner when the global MoE runner backend is
flashinfer_cutlass or flashinfer_cutedsl, which have no fp8 MoE path."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization.fp8 as fp8
from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)
from sglang.test.test_utils import CustomTestCase


class TestFp8MoeRunnerFallback(CustomTestCase):
    def test_routed_trtllm_selects_the_checkpoint_expert_dtype(self):
        """Packed MXFP4 must not enter the routed runner's FP8 GEMM branch."""
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        with (
            patch.object(
                fp8,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
            ),
            patch.object(
                fp8,
                "get_platform",
                return_value=SimpleNamespace(
                    is_sm100=True, is_sm90=False, is_sm120=False
                ),
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
                ),
            ),
        ):
            for is_fp4 in (True, False):
                with self.subTest(is_fp4=is_fp4):
                    config = fp8.Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        weight_block_size=[128, 128],
                        is_fp4_experts=is_fp4,
                    )
                    method = config.get_quant_method(
                        layer, "model.layers.0.mlp.experts"
                    )
                    expected = (
                        Mxfp4FlashinferTrtllmMoEMethod if is_fp4 else fp8.Fp8MoEMethod
                    )
                    self.assertIsInstance(method, expected)

    def _runner_backend_for(self, global_backend):
        method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
        with patch.object(fp8, "get_moe_runner_backend", return_value=global_backend):
            method.create_moe_runner(layer=None, moe_runner_config=MoeRunnerConfig())
        return method.runner.runner_backend

    def test_flashinfer_cutlass_falls_back_to_triton(self):
        self.assertTrue(
            self._runner_backend_for(MoeRunnerBackend.FLASHINFER_CUTLASS).is_triton()
        )

    def test_flashinfer_cutedsl_falls_back_to_triton(self):
        self.assertTrue(
            self._runner_backend_for(MoeRunnerBackend.FLASHINFER_CUTEDSL).is_triton()
        )

    def test_triton_is_kept(self):
        self.assertTrue(self._runner_backend_for(MoeRunnerBackend.TRITON).is_triton())


if __name__ == "__main__":
    unittest.main()
