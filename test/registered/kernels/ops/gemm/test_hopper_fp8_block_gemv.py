"""
Tests the Hopper small-M blockwise-FP8 GEMV JIT kernel against a dequantized
fp32 reference and its bf16 side projection.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _block_quant(w: torch.Tensor):
    """128x128 blockwise e4m3 weights with fp32 [ceil(N/128), K/128] scales."""
    n, k = w.shape
    nb = (n + 127) // 128
    padded = torch.zeros((nb * 128, k), dtype=torch.float32, device=w.device)
    padded[:n] = w.float()
    blocks = padded.view(nb, 128, k // 128, 128)
    scales = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-10) / 448.0
    wq = (blocks / scales[:, None, :, None]).clamp(-448, 448)
    wq = wq.view(nb * 128, k)[:n].to(torch.float8_e4m3fn).contiguous()
    return wq, scales.contiguous()


@unittest.skipIf(not _is_sm90(), "Hopper fp8 block GEMV requires an SM90 GPU")
class TestHopperFp8BlockGemv(CustomTestCase):
    def _inputs(self, m, n, k, seed=0):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        torch.manual_seed(seed)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
        wq, ws = _block_quant(w)
        xq, xs = sglang_per_token_group_quant_fp8(x, 128)
        return x, xq, xs, wq, ws

    def _reference(self, xq, xs, wq, ws):
        m, k = xq.shape
        n = wq.shape[0]
        xdq = xq.float().view(m, k // 128, 128) * xs.view(m, k // 128, 1)
        nb = ws.shape[0]
        wpad = torch.zeros((nb * 128, k), dtype=torch.float32, device=wq.device)
        wpad[:n] = wq.float()
        wdq = wpad.view(nb, 128, k // 128, 128) * ws.view(nb, 1, k // 128, 1)
        return xdq.view(m, k) @ wdq.view(nb * 128, k)[:n].t()

    def test_prequantized_matches_reference(self):
        from sglang.kernels.ops.gemm.hopper_fp8_block_gemv import hopper_fp8_block_gemv

        # KDA qkvfg (row tail past 16), KDA o_proj (K=1024), MLA q_b, dense up.
        for n, k in [(5128, 4096), (4096, 1024), (1536, 1536), (2560, 4096)]:
            for m in (1, 6, 8):
                x, xq, xs, wq, ws = self._inputs(m, n, k)
                out = hopper_fp8_block_gemv(xq, xs, wq, ws)
                ref = self._reference(xq, xs, wq, ws)
                # The e4m3 tensor-core accumulator carries ~13 significant
                # bits per 128-wide block before the fp32 promotion, so a
                # 4096-deep dot product lands within ~0.5% of the fp32 sum.
                tol = 1e-2 * ref.abs().max().item() + 1e-3
                torch.testing.assert_close(out.float(), ref, atol=tol, rtol=0)

    def test_two_column_groups(self):
        from sglang.kernels.ops.gemm.hopper_fp8_block_gemv import hopper_fp8_block_gemv

        x, xq, xs, wq, ws = self._inputs(12, 1536, 1536)
        out = hopper_fp8_block_gemv(xq, xs, wq, ws)
        ref = self._reference(xq, xs, wq, ws)
        torch.testing.assert_close(
            out.float(), ref, atol=1e-2 * ref.abs().max().item() + 1e-3, rtol=0
        )

    def test_side_projection(self):
        from sglang.kernels.ops.gemm.hopper_fp8_block_gemv import hopper_fp8_block_gemv

        for m in (1, 6, 8):
            x, xq, xs, wq, ws = self._inputs(m, 5128, 4096, seed=100 + m)
            side_w = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda") * 0.05
            out, side = hopper_fp8_block_gemv(xq, xs, wq, ws, x_bf16=x, side_w=side_w)
            self.assertTrue(torch.equal(out, hopper_fp8_block_gemv(xq, xs, wq, ws)))
            ref = x.float() @ side_w.float().t()
            cub = (x @ side_w.t()).float()
            err = (side.float() - ref).abs().max().item()
            err_cub = (cub - ref).abs().max().item()
            # fp32 accumulation with a warp-tree reduction: at least as tight
            # as cuBLAS's split-K against the fp32 reference.
            self.assertLessEqual(err, max(err_cub * 2.0, 1e-2), (m, err, err_cub))


if __name__ == "__main__":
    unittest.main()
