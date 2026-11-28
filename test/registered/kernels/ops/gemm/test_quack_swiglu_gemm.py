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

    def test_kernel_reads_gate_up_layout_in_place(self):
        """The merged weight is [gate; up]. M covers both tile heights, all tile
        widths and partial M tiles."""
        from sglang.kernels.ops.gemm.quack_swiglu_gemm import quack_swiglu_gemm

        for m in (1, 256, 300, 512, 600, 768, 2049):
            with self.subTest(m=m):
                torch.manual_seed(m)
                x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(2 * INTER, HIDDEN, device="cuda", dtype=torch.bfloat16)
                w = w * 0.05
                torch.testing.assert_close(
                    quack_swiglu_gemm(x, w).float(),
                    _swiglu_ref(x, w),
                    rtol=2e-2,
                    atol=2e-2,
                )

    def _mlp(self):
        from sglang.srt.models.qwen2 import Qwen2MLP

        torch.manual_seed(0)
        with envs.SGLANG_USE_QUACK_SWIGLU_GEMM.override(True):
            mlp = Qwen2MLP(HIDDEN, INTER, "silu").to("cuda", torch.bfloat16)
        for p in mlp.parameters():
            p.data.normal_(std=0.05)
        return mlp

    def _run(self, mlp, m):
        import sglang.kernels.ops.gemm as gemm_ops

        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(
            gemm_ops, "dual_gemm_swiglu_quack", wraps=gemm_ops.dual_gemm_swiglu_quack
        ) as fused:
            out = mlp(x)
        return x, out, fused.called

    def test_mlp_takes_fused_path(self):
        """With the flag on, the MLP takes the fused GEMM and matches fp32."""
        mlp = self._mlp()
        for m in (1, 2048):
            with self.subTest(m=m):
                x, out, fused = self._run(mlp, m)
                self.assertTrue(fused)
                ref = _swiglu_ref(x, mlp.gate_up_proj.weight)
                ref = ref @ mlp.down_proj.weight.float().t()
                err = (out.float() - ref).norm() / ref.norm()
                self.assertLess(err.item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
