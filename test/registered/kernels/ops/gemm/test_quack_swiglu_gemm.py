"""Tests for the SM120 quack gate_up GEMM with SwiGLU fused into its epilogue."""

import unittest
from unittest import mock

import torch
from torch import nn

from sglang.kernels.arch import current_arch
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HIDDEN, INTER = 1024, 3072


def _swiglu_ref(x, w_gate_up):
    gate_up = x.float() @ w_gate_up.float().t()
    gate, up = gate_up.chunk(2, dim=-1)
    return nn.functional.silu(gate) * up


@unittest.skipIf(
    not torch.cuda.is_available() or not current_arch().is_sm(12, 0),
    "quack's SM120 warp-MMA GEMM requires SM120",
)
class TestQuackSwigluGemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang.test.layer_ut_utils import init_single_process_dist

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        init_single_process_dist()

    def test_mlp_runs_the_fused_gemm(self):
        """The merged weight is [gate; up]. M covers both tile heights, all tile
        widths and partial M tiles."""
        import sglang.kernels.ops.gemm as gemm_ops
        from sglang.srt.models.qwen2 import Qwen2MLP

        torch.manual_seed(0)
        with envs.SGLANG_USE_QUACK_SWIGLU_GEMM.override(True):
            mlp = Qwen2MLP(HIDDEN, INTER, "silu").to("cuda", torch.bfloat16)
        for p in mlp.parameters():
            p.data.normal_(std=0.05)
        fused = gemm_ops.dual_gemm_swiglu_quack
        activations = []

        def record(x, weight):
            activations.append(fused(x, weight))
            return activations[-1]

        for m in (1, 256, 300, 512, 600, 768, 2049):
            with self.subTest(m=m):
                activations.clear()
                x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
                with mock.patch.object(gemm_ops, "dual_gemm_swiglu_quack", record):
                    mlp(x)
                self.assertEqual(len(activations), 1)
                torch.testing.assert_close(
                    activations[0].float(),
                    _swiglu_ref(x, mlp.gate_up_proj.weight),
                    rtol=2e-2,
                    atol=2e-2,
                )


if __name__ == "__main__":
    unittest.main()
