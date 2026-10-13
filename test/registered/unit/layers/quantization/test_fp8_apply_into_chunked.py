"""Fp8LinearMethod.apply_into: caller-owned output, whole and row-chunked.

The chunked GEMM+AR overlap writes each row chunk's GEMM straight into a
slice of one preallocated buffer and claims bit-identical results to the
unchunked forward. That claim rests on two properties pinned here:

1. ``apply_into(layer, x, out)`` produces exactly ``apply(layer, x)`` (the
   deepgemm path writes into ``out`` in place; other dispatches copy).
2. Running row chunks independently reproduces the full-M result bit for bit
   (per-token-group quant depends only on the row; the deepgemm K-loop
   accumulates in a fixed order per row -- no split-K).
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

if torch.cuda.is_available():
    from sglang.srt.layers.chunked_gemm_ar import compute_chunk_plan
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

N, K = 512, 256
BLOCK = 128


def _make_layer(device):
    torch.manual_seed(7)
    weight = (torch.randn(N, K, device=device) * 0.05).to(torch.float8_e4m3fn)
    weight_scale_inv = (
        torch.rand(N // BLOCK, K // BLOCK, device=device, dtype=torch.float32) * 0.01
        + 0.005
    )

    class _Layer(torch.nn.Module):
        pass

    layer = _Layer()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
    return layer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFp8ApplyIntoChunked(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        cls.method = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[BLOCK, BLOCK],
            )
        )
        cls.layer = _make_layer(cls.device)

    def _x(self, m):
        torch.manual_seed(m)
        return torch.randn(m, K, device=self.device, dtype=torch.bfloat16)

    def test_apply_into_matches_apply(self):
        for m in (64, 1900, 4096):
            x = self._x(m)
            ref = self.method.apply(self.layer, x)
            out = torch.empty_like(ref)
            ret = self.method.apply_into(self.layer, x, out)
            self.assertTrue(ret.data_ptr() == out.data_ptr())
            self.assertTrue(torch.equal(ref, out), f"m={m}")

    def test_apply_into_with_bias(self):
        x = self._x(1024)
        bias = torch.randn(N, device=self.device, dtype=torch.bfloat16)
        ref = self.method.apply(self.layer, x, bias=bias)
        out = torch.empty_like(ref)
        self.method.apply_into(self.layer, x, out, bias=bias)
        self.assertTrue(torch.equal(ref, out))

    def _run_chunked(self, m, plan):
        x = self._x(m)
        ref = self.method.apply(self.layer, x)
        out = torch.empty_like(ref)
        for lo, hi in plan:
            self.method.apply_into(self.layer, x[lo:hi], out[lo:hi])
        self.assertTrue(torch.equal(ref, out), f"m={m} plan={plan}")

    def test_row_chunks_bit_exact(self):
        # Hand-picked boundaries, including tile-misaligned ones.
        self._run_chunked(4096, [(0, 2048), (2048, 3072), (3072, 4096)])
        self._run_chunked(4000, [(0, 2560), (2560, 3333), (3333, 4000)])

    def test_geometric_plan_bit_exact(self):
        m = 9000
        plan = compute_chunk_plan(m, 4, 0.36)
        self.assertIsNotNone(plan)
        self._run_chunked(m, plan)


if __name__ == "__main__":
    unittest.main()
