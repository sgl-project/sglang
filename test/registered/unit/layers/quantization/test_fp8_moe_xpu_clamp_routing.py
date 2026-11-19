"""On XPU, an FP8 MoE layer whose SwiGLU clamp has no GPT-OSS alpha runs on the
triton runner instead of sgl-kernel-xpu's fused_experts."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.layers.quantization.fp8 as fp8
from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig

# apply() imports this lazily; importing it while sgl_kernel is faked would fail.
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # noqa: F401
from sglang.test.test_utils import CustomTestCase


def _route_on_xpu(moe_runner_config):
    """Run Fp8MoEMethod.apply as XPU does; return (native kernel called, triton runner called)."""
    method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
    method.is_fp4_expert = False
    method.block_quant = True
    method.moe_runner_config = moe_runner_config
    method.runner = MagicMock(runner_backend=MoeRunnerBackend.TRITON)
    method.get_triton_quant_info = MagicMock()
    layer = SimpleNamespace(
        w13_weight=torch.zeros(1, 2, 2, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(1, 2, 2, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(1, 1, 1),
        w2_weight_scale_inv=torch.ones(1, 1, 1),
    )
    dispatch_output = SimpleNamespace(
        hidden_states=torch.zeros(1, 2, dtype=torch.bfloat16),
        topk_output=(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32), None),
    )
    sgl_kernel = SimpleNamespace(
        fused_experts=MagicMock(return_value=torch.zeros(1, 2))
    )
    with (
        patch.object(fp8, "_is_xpu", True),
        patch.object(fp8, "get_moe_runner_backend", return_value=MoeRunnerBackend.AUTO),
        patch.dict(sys.modules, {"sgl_kernel": sgl_kernel}),
    ):
        method.apply(layer, dispatch_output)
    return sgl_kernel.fused_experts.called, method.runner.run.called


class TestFp8MoeXpuClampRouting(CustomTestCase):
    def test_clamp_without_alpha_runs_on_triton(self):
        """Ling-3.0-flash-fp8 (routed clamp 4 on its last 7 layers, no alpha) died at the
        first forward: sgl-kernel-xpu asserts that a clamp limit comes with the GPT-OSS
        alpha, and its alpha-free clamp op clamps before SiLU rather than after."""
        native, triton = _route_on_xpu(MoeRunnerConfig(gemm1_clamp_limit=4.0))
        self.assertFalse(native)
        self.assertTrue(triton)

    def test_unclamped_and_gpt_oss_layers_stay_native(self):
        for config in (
            MoeRunnerConfig(),
            MoeRunnerConfig(gemm1_alpha=1.702, gemm1_clamp_limit=7.0),
        ):
            with self.subTest(config=config):
                native, triton = _route_on_xpu(config)
                self.assertTrue(native)
                self.assertFalse(triton)


if __name__ == "__main__":
    unittest.main()
