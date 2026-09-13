import math
import unittest

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache,
    dequantize_k_cache_paged,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
    FP8_DTYPE,
    tilelang_sparse_fwd,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _torch_sparse_attention(q, kv, indices, scale, d_v):
    rows = indices[:, 0]
    valid = rows >= 0
    selected = kv[rows.clamp_min(0), 0]
    scores = torch.einsum("thd,tkd->thk", q.float(), selected.float()) * scale
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(valid[:, None, :], probs, 0)
    return torch.einsum("thk,tkd->thd", probs, selected[..., :d_v].float()).to(
        torch.bfloat16
    )


@unittest.skipIf(
    not torch.cuda.is_available() or torch.version.hip is None, "ROCm required"
)
class TestTileLangDSAZeroRope(CustomTestCase):
    def _run_case(self, tokens, topk, use_fp8, padded, d_v=256, d_tail=0):
        torch.manual_seed(7)
        heads, slots = 64, 2112
        dim = d_v + d_tail
        q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(slots, 1, dim, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(topk, device="cuda", dtype=torch.int32)
        indices = (
            indices.remainder(slots).view(1, 1, topk).expand(tokens, -1, -1).clone()
        )
        if padded:
            indices[..., 2051:] = -1

        if use_fp8:
            q = q.to(FP8_DTYPE)
            kv = kv.to(FP8_DTYPE)

        scale = 1.0 / math.sqrt(dim)
        expected = _torch_sparse_attention(q, kv, indices, scale, d_v)
        actual = tilelang_sparse_fwd(q, kv, indices, scale, d_v=d_v)
        if actual.ndim == 4:
            actual = actual.squeeze(0)

        self.assertEqual(actual.shape, (tokens, heads, d_v))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.20 if use_fp8 else 0.04,
            rtol=0.12 if use_fp8 else 0.04,
        )
        repeated = tilelang_sparse_fwd(q, kv, indices, scale, d_v=d_v)
        if repeated.ndim == 4:
            repeated = repeated.squeeze(0)
        torch.testing.assert_close(actual, repeated, atol=0, rtol=0)

    def test_glm_bf16_zero_rope(self):
        for tokens in (1, 8, 17):
            for topk, padded in ((2048, False), (2112, True)):
                with self.subTest(tokens=tokens, topk=topk, padded=padded):
                    self._run_case(tokens, topk, use_fp8=False, padded=padded)

    def test_glm_fp8_zero_rope(self):
        for tokens in (1, 8, 17):
            for topk, padded in ((2048, False), (2112, True)):
                with self.subTest(tokens=tokens, topk=topk, padded=padded):
                    self._run_case(tokens, topk, use_fp8=True, padded=padded)

    def test_deepseek_tail64_regression(self):
        for use_fp8 in (False, True):
            with self.subTest(use_fp8=use_fp8):
                self._run_case(
                    tokens=1,
                    topk=2048,
                    use_fp8=use_fp8,
                    padded=False,
                    d_v=512,
                    d_tail=64,
                )

    def test_scaled_cache_round_trip_layouts(self):
        for dim_nope, dim_rope, packed_width in ((256, 0, 264), (512, 64, 656)):
            with self.subTest(dim_nope=dim_nope, dim_rope=dim_rope):
                source = torch.randn(
                    8,
                    1,
                    1,
                    dim_nope + dim_rope,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                packed = quantize_k_cache(source, dv=dim_nope)
                self.assertEqual(packed.shape[-1], packed_width)
                restored = dequantize_k_cache(packed, dv=dim_nope)
                self.assertEqual(restored.shape, source.shape)
                torch.testing.assert_close(restored, source, atol=0.08, rtol=0.08)

                pages = torch.tensor([7, 1, 1, 4], device="cuda", dtype=torch.int32)
                gathered = dequantize_k_cache_paged(packed, pages)
                expected = restored.view(8, 1, -1)[pages]
                torch.testing.assert_close(gathered, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
