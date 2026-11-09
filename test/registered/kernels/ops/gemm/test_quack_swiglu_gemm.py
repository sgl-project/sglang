"""Tests for the SM120 quack gate_up GEMM with SwiGLU fused into its epilogue."""

import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

HIDDEN, INTER = 1024, 3072


def _swiglu_ref(x, w_gate_up):
    gate_up = x.float() @ w_gate_up.float().t()
    gate, up = gate_up.chunk(2, dim=-1)
    return nn.functional.silu(gate) * up


class TestTileShape(CustomTestCase):
    def test_tile_shape_avoids_partial_wave(self):
        """On 188 SMs, 128x128 tiles at M=512 make 192 tiles (a wave plus 4)
        and 128x192 tiles at M=768 make 192 too; both must pick a shape that
        fills the waves instead."""
        from sglang.kernels.ops.gemm.quack_swiglu_gemm import _tile_shape

        self.assertEqual(_tile_shape(512, 2 * INTER, 188), (128, 160))
        self.assertEqual(_tile_shape(768, 2 * INTER, 188), (128, 128))

    def test_64_row_tiles_only_in_one_wave(self):
        """At M=256 128-row tiles leave most SMs idle and 64x160 tiles fill one
        wave. At M=384 and M=448 every 64-row shape needs a second wave, where
        128-row tiles are faster, even though 64x128 has no more work per SM."""
        from sglang.kernels.ops.gemm.quack_swiglu_gemm import _tile_shape

        self.assertEqual(_tile_shape(256, 2 * INTER, 188), (64, 160))
        self.assertEqual(_tile_shape(384, 2 * INTER, 188), (128, 128))
        self.assertEqual(_tile_shape(448, 2 * INTER, 188), (128, 160))


@unittest.skipIf(
    not torch.cuda.is_available() or get_device_sm() // 10 != 12,
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
        """SGLang stacks the merged weight as [gate; up]; reading it as
        interleaved or swapped halves changes every output. M covers both
        tile heights, all three tile widths and partial M tiles."""
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
        with envs.SGLANG_ENABLE_QUACK_SWIGLU_GEMM.override(True):
            mlp = Qwen2MLP(HIDDEN, INTER, "silu").to("cuda", torch.bfloat16)
        for p in mlp.parameters():
            p.data.normal_(std=0.05)
        return mlp

    def _run(self, mlp, m):
        from sglang.srt.models import qwen2

        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(
            qwen2, "quack_swiglu_gemm", wraps=qwen2.quack_swiglu_gemm
        ) as fused:
            out = mlp(x)
        return x, out, fused.called

    def test_lora_wrapped_gate_up_keeps_unfused_path(self):
        """LoRA replaces gate_up_proj with a wrapper after construction; the
        fused GEMM reads only the base weight, so taking it would silently
        drop the adapter delta."""
        mlp = self._mlp()
        base = mlp.gate_up_proj

        class _Wrapper(nn.Module):
            def forward(self, x):
                out, bias = base(x)
                return out + 1.0, bias

        mlp.gate_up_proj = _Wrapper()
        x, out, fused = self._run(mlp, 1024)
        self.assertFalse(fused)
        expected, _ = mlp.down_proj(mlp.act_fn(base(x)[0] + 1.0))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_empty_batch_keeps_unfused_path(self):
        """An empty batch must not reach the fused GEMM: quack's launcher kills
        the process (SIGFPE) on 0 rows."""
        mlp = self._mlp()
        _, out, fused = self._run(mlp, 0)
        self.assertFalse(fused)
        self.assertEqual(out.shape, (0, HIDDEN))

    def test_mlp_takes_fused_path(self):
        """With the flag on, decode-sized and prefill-sized batches both take
        the fused GEMM and match the fp32 MLP."""
        mlp = self._mlp()
        for m in (1, 256, 2048):
            with self.subTest(m=m):
                x, out, fused = self._run(mlp, m)
                self.assertTrue(fused)
                ref = _swiglu_ref(x, mlp.gate_up_proj.weight)
                ref = ref @ mlp.down_proj.weight.float().t()
                err = (out.float() - ref).norm() / ref.norm()
                self.assertLess(err.item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
