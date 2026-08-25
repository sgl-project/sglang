import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.zimage import (
    ZImageRMSNorm,
    zimage_native_qk_rmsnorm,
)

HEAD_DIM = 128


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestZImageQkNormFusion(unittest.TestCase):
    def _make_norm(self, seed):
        torch.manual_seed(seed)
        norm = ZImageRMSNorm(HEAD_DIM, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(HEAD_DIM) * 0.5 + 1.0)
        return norm.to(device="cuda", dtype=torch.bfloat16)

    def test_bit_exact_on_strided_qkv_view(self):
        norm_q, norm_k = self._make_norm(1), self._make_norm(2)
        heads, kv_heads = 4, 4
        total = (heads + 2 * kv_heads) * HEAD_DIM
        qkv = (torch.randn(1, 64, total, device="cuda")).to(torch.bfloat16)
        q = qkv[..., : heads * HEAD_DIM].view(1, 64, heads, HEAD_DIM)
        k = qkv[..., heads * HEAD_DIM : (heads + kv_heads) * HEAD_DIM].view(
            1, 64, kv_heads, HEAD_DIM
        )

        fused = zimage_native_qk_rmsnorm(q, k, norm_q, norm_k, HEAD_DIM)
        self.assertIsNotNone(fused)
        q_out, k_out = fused
        self.assertTrue(q_out.is_contiguous() and k_out.is_contiguous())
        self.assertTrue(
            torch.equal(q_out, norm_q(q.reshape(-1, HEAD_DIM)).view(q.shape))
        )
        self.assertTrue(
            torch.equal(k_out, norm_k(k.reshape(-1, HEAD_DIM)).view(k.shape))
        )

    def test_unsupported_head_dim_falls_back(self):
        norm = ZImageRMSNorm(64, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(zimage_native_qk_rmsnorm(q, k, norm, norm, 64))


if __name__ == "__main__":
    unittest.main()
