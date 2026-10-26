"""The asm split-K block-scale GEMM into a norm-zeroed output equals aiter's memset-then-launch."""

import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

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

    def test_norm_hands_zeroed_output_to_linear(self):
        """A norm bound to the GEMM zero-fills its output; the linear launches into it
        once, in eager and in graph replay."""
        from sglang.srt.layers.layer_boundary.residual import add_norm_hip
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        from sglang.srt.layers.quantization import fp8_utils, fp8_utils_hip

        m, kernel = 256, _kernel("48x128")
        norm = GemmaRMSNorm(K).cuda()
        norm.weight.data.copy_(self.norm_w)
        norm.fp8_out_gemm_shape = (N, K)
        x, res = self._rows(m), self._rows(m)
        ref = self._aiter_ref(x, res, m, kernel)

        def run():
            (q, scale), _ = add_norm_hip.gemma_norm_fp8(norm, x, res, None, "fp8")
            self.assertEqual(q.zeroed_gemm_out.shape, (m, N))
            out = fp8_utils.aiter_w8a8_block_fp8_linear(
                input=q,
                weight=self.wq,
                block_size=[128, 128],
                weight_scale=self.ws,
                input_scale=scale,
            )
            self.assertFalse(hasattr(q, "zeroed_gemm_out"))
            return out

        def pick(m, n, k):
            return kernel, 2

        with (
            mock.patch.object(add_norm_hip, "asm_split_k_kernel", pick),
            mock.patch.object(fp8_utils_hip, "asm_split_k_kernel", pick),
        ):
            self.assertTrue(torch.equal(run(), ref))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            out.fill_(7.0)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out, ref))

    def test_zeroed_output_is_bf16_only(self):
        """The asm kernel accumulates bf16: an fp16 norm hands no zeroed output, and a
        linear asked for another dtype keeps aiter's own launch."""
        from sglang.srt.layers.layer_boundary.residual import add_norm_hip
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        from sglang.srt.layers.quantization import fp8_utils_hip

        m, kernel = 256, _kernel("48x128")
        norm = GemmaRMSNorm(K).cuda()
        norm.fp8_out_gemm_shape = (N, K)
        x, res = self._rows(m).half(), self._rows(m).half()

        def pick(m, n, k):
            return kernel, 2

        launched = []

        def gemm(xq, wq, x_scale, w_scale, dtype):
            launched.append(dtype)

        with (
            mock.patch.object(add_norm_hip, "asm_split_k_kernel", pick),
            mock.patch.object(fp8_utils_hip, "asm_split_k_kernel", pick),
        ):
            (q, scale), _ = add_norm_hip.gemma_norm_fp8(norm, x, res, None, "fp8")
            self.assertFalse(hasattr(q, "zeroed_gemm_out"))
            zeroed = torch.zeros(m, N, device="cuda", dtype=torch.bfloat16)
            fp8_utils_hip._gemm_into_zeroed(
                q, self.wq, scale, self.ws, torch.float32, gemm=gemm, out=zeroed
            )
        self.assertEqual(launched, [torch.float32])


if __name__ == "__main__":
    unittest.main()
