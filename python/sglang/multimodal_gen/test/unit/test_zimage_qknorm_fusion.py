import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.zimage import (
    ZImageRMSNorm,
    zimage_native_qk_rmsnorm,
)


def _eager_qk_norm(x: torch.Tensor, norm: ZImageRMSNorm, head_dim: int) -> torch.Tensor:
    # Exact eager fallback chain from apply_qk_norm.
    return norm(x.reshape(-1, head_dim)).view(x.shape)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestZImageQkNormFusion(unittest.TestCase):
    HEAD_DIM = 128

    def _make_qkv_views(self, batch, seq, heads, kv_heads, scale, seed=0):
        torch.manual_seed(seed)
        total = (heads + 2 * kv_heads) * self.HEAD_DIM
        qkv = (torch.randn(batch, seq, total, device="cuda") * scale).to(torch.bfloat16)
        q = qkv[..., : heads * self.HEAD_DIM].view(batch, seq, heads, self.HEAD_DIM)
        k = qkv[..., heads * self.HEAD_DIM : (heads + kv_heads) * self.HEAD_DIM].view(
            batch, seq, kv_heads, self.HEAD_DIM
        )
        return q, k

    def _make_norm(self, seed):
        torch.manual_seed(seed)
        norm = ZImageRMSNorm(self.HEAD_DIM, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(self.HEAD_DIM) * 0.5 + 1.0)
        return norm.to(device="cuda", dtype=torch.bfloat16)

    def test_bit_exact_across_scales(self):
        norm_q = self._make_norm(1)
        norm_k = self._make_norm(2)
        for scale in (0.005, 0.02, 0.5, 1.0, 4.0, 30.0, 200.0):
            q, k = self._make_qkv_views(1, 4352, 30, 30, scale, seed=int(scale * 1000))
            fused = zimage_native_qk_rmsnorm(q, k, norm_q, norm_k, self.HEAD_DIM)
            self.assertIsNotNone(fused, f"fused path unsupported at scale {scale}")
            q_out, k_out = fused
            self.assertTrue(q_out.is_contiguous())
            self.assertTrue(k_out.is_contiguous())
            self.assertTrue(
                torch.equal(q_out, _eager_qk_norm(q, norm_q, self.HEAD_DIM)),
                f"q mismatch at scale {scale}",
            )
            self.assertTrue(
                torch.equal(k_out, _eager_qk_norm(k, norm_k, self.HEAD_DIM)),
                f"k mismatch at scale {scale}",
            )

    def test_bit_exact_batched_and_gqa(self):
        norm_q = self._make_norm(3)
        norm_k = self._make_norm(4)
        q, k = self._make_qkv_views(2, 257, 30, 10, 1.0, seed=7)
        fused = zimage_native_qk_rmsnorm(q, k, norm_q, norm_k, self.HEAD_DIM)
        self.assertIsNotNone(fused)
        q_out, k_out = fused
        self.assertTrue(torch.equal(q_out, _eager_qk_norm(q, norm_q, self.HEAD_DIM)))
        self.assertTrue(torch.equal(k_out, _eager_qk_norm(k, norm_k, self.HEAD_DIM)))

    def test_unsupported_head_dim_falls_back(self):
        norm_q = ZImageRMSNorm(64, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
        norm_k = ZImageRMSNorm(64, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(zimage_native_qk_rmsnorm(q, k, norm_q, norm_k, 64))

    def test_unsupported_dtype_falls_back(self):
        norm_q = self._make_norm(5).to(dtype=torch.float16)
        norm_k = self._make_norm(6).to(dtype=torch.float16)
        q = torch.randn(1, 32, 4, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 32, 4, 128, device="cuda", dtype=torch.float16)
        self.assertIsNone(zimage_native_qk_rmsnorm(q, k, norm_q, norm_k, 128))


if __name__ == "__main__":
    unittest.main()
