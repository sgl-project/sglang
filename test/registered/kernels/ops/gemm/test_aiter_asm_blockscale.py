"""The asm split-K block-scale GEMM into a norm-zeroed output equals aiter's memset-then-launch."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

N, K = 2560, 6144


def _kernel(tile):
    return f"_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_{tile}E"


@unittest.skipUnless(is_gfx95_supported(), "ROCm gfx95 only")
class TestAsmBlockscalePrezeroed(CustomTestCase):
    def setUp(self):
        from aiter.ops.shuffle import shuffle_weight

        from sglang.srt.layers.quantization.mxfp8_block_convert import (
            bf16_to_block_fp8_128,
        )

        self.gen = torch.Generator(device="cuda").manual_seed(0)
        wq, self.ws = bf16_to_block_fp8_128(
            torch.randn(N, K, device="cuda", generator=self.gen) * 0.02, block=128
        )
        self.wq = shuffle_weight(wq, (16, 16))
        self.norm_w = torch.randn(K, device="cuda", generator=self.gen) * 0.1

    def _rows(self, m):
        return torch.randn(m, K, device="cuda", generator=self.gen).to(torch.bfloat16)

    def _aiter_ref(self, x, res, m, kernel):
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_asm

        from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
            gemma_add_rmsnorm_fp8,
        )

        (q, scale), _ = gemma_add_rmsnorm_fp8(
            x, res, self.norm_w, 1e-6, group_size=128, transpose_scale=True
        )
        ref = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
        gemm_a8w8_blockscale_bpreshuffle_asm(
            q, self.wq, ref, scale, self.ws, splitK=2, kernelName=kernel
        )
        return ref

    def test_matches_aiter_launch(self):
        from aiter.ops.gemm_op_a8w8 import get_zero_bias_buf

        from sglang.kernels.ops.gemm.aiter_asm_blockscale import (
            asm_split_k_supported,
            gemm_a8w8_blockscale_bpreshuffle_asm_prezeroed,
        )
        from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
            gemma_add_rmsnorm_fp8,
        )

        wq, ws, norm_w = self.wq, self.ws, self.norm_w
        for m, tile in ((144, "32x128"), (256, "48x128"), (300, "64x128")):
            kernel = _kernel(tile)
            self.assertTrue(asm_split_k_supported(kernel, n=N, k=K, split_k=2))
            x, res = self._rows(m), self._rows(m)
            out = torch.full((m, N), 7.0, device="cuda", dtype=torch.bfloat16)
            (q, scale), _ = gemma_add_rmsnorm_fp8(
                x, res, norm_w, 1e-6, group_size=128, transpose_scale=True, zeros=out
            )
            (q_ref, scale_ref), _ = gemma_add_rmsnorm_fp8(
                x, res, norm_w, 1e-6, group_size=128, transpose_scale=True
            )
            self.assertEqual(int(out.count_nonzero()), 0)
            self.assertTrue(torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)))
            self.assertTrue(torch.equal(scale, scale_ref))
            ref = self._aiter_ref(x, res, m, kernel)
            gemm_a8w8_blockscale_bpreshuffle_asm_prezeroed(
                xq=q,
                wq=wq,
                x_scale=scale,
                w_scale=ws,
                out=out,
                kernel_name=kernel,
                split_k=2,
                bias=get_zero_bias_buf(wq),
            )
            self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
