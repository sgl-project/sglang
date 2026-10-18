"""aiter fused QK-norm + RoPE + KV-cache write used by Qwen3 decode on ROCm.

``aiter_fused_qk_norm_rope_set_kv`` hands the kernel sglang's NHD
(slots, num_kv_heads, head_dim) pool buffers. Guards: q and the K/V written at
``out_cache_loc`` match QK-norm + RoPE computed with the reference modules, and
no other cache slot is written.
"""

import os
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-a-test-1-gpu-small-amd")

HEAD_DIM = 128
EPS = 1e-6
MAX_POS = 8192
NUM_SLOTS = 4096
# Qwen3-VL text config: interleaved 3D mRoPE over a 128-dim head.
MROPE_SCALING = {
    "rope_type": "default",
    "mrope_section": [24, 20, 20],
    "mrope_interleaved": True,
}
# bf16 carries ~8 mantissa bits; q/k go through a norm and a rotation.
TOL = 2e-2


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip, "ROCm aiter kernel"
)
class TestQwen3AiterFusedQKNormRopeSetKV(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_USE_AITER", "1")
        from sglang.srt.models import qwen3

        if not (qwen3._has_fused_qk_norm_mrope or qwen3._has_fused_qk_norm_rope):
            raise unittest.SkipTest("aiter fused QK-norm + RoPE kernels unavailable")
        cls.qwen3 = qwen3
        cls.device = torch.device("cuda", 0)

    def _norm(self):
        from sglang.srt.layers.layernorm import RMSNorm

        norm = RMSNorm(HEAD_DIM, eps=EPS).to(self.device, torch.bfloat16)
        norm.weight.data.copy_(1.0 + 0.1 * torch.randn(HEAD_DIM, device=self.device))
        return norm

    def _rope(self):
        from sglang.srt.layers.rotary_embedding import get_rope

        return get_rope(HEAD_DIM, HEAD_DIM, MAX_POS, 1000000).to(self.device)

    def _mrope(self):
        from sglang.srt.layers.rotary_embedding import get_rope

        return get_rope(
            HEAD_DIM, HEAD_DIM, MAX_POS, 5000000, rope_scaling=MROPE_SCALING
        ).to(self.device)

    def _check(self, rotary_emb, positions, num_heads, num_kv_heads):
        num_tokens = positions.shape[-1]
        q_norm, k_norm = self._norm(), self._norm()
        qkv = torch.randn(
            num_tokens,
            (num_heads + 2 * num_kv_heads) * HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k_cache = torch.zeros(
            NUM_SLOTS, num_kv_heads, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        v_cache = torch.zeros_like(k_cache)
        slots = torch.randperm(NUM_SLOTS - 1, device=self.device)[:num_tokens] + 1
        scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")

        q = self.qwen3.aiter_fused_qk_norm_rope_set_kv(
            qkv,
            q_norm,
            k_norm,
            rotary_emb,
            positions,
            k_cache,
            v_cache,
            slots,
            num_heads,
            num_kv_heads,
            HEAD_DIM,
            scale,
            scale,
        )

        q_ref, k_ref, v_ref = qkv.split(
            [num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM, num_kv_heads * HEAD_DIM],
            dim=-1,
        )
        q_ref = q_norm.forward_native(q_ref.reshape(-1, HEAD_DIM)).view(num_tokens, -1)
        k_ref = k_norm.forward_native(k_ref.reshape(-1, HEAD_DIM)).view(num_tokens, -1)
        q_ref, k_ref = rotary_emb.forward_native(positions, q_ref, k_ref)

        torch.testing.assert_close(q, q_ref, atol=TOL, rtol=TOL)
        torch.testing.assert_close(
            k_cache[slots],
            k_ref.view(num_tokens, num_kv_heads, HEAD_DIM),
            atol=TOL,
            rtol=TOL,
        )
        torch.testing.assert_close(
            v_cache[slots],
            v_ref.reshape(num_tokens, num_kv_heads, HEAD_DIM),
            atol=0,
            rtol=0,
        )
        untouched = torch.ones(NUM_SLOTS, dtype=torch.bool, device=self.device)
        untouched[slots] = False
        self.assertFalse(k_cache[untouched].any().item(), "K written to other slots")
        self.assertFalse(v_cache[untouched].any().item(), "V written to other slots")

    def test_rope_nhd_cache(self):
        if not self.qwen3._has_fused_qk_norm_rope:
            self.skipTest("aiter fused_qk_norm_rope kernel unavailable")
        torch.manual_seed(0)
        rotary_emb = self._rope()
        # (32, 8): Qwen3-8B at TP1; (8, 2): Qwen3-8B at TP4.
        for num_heads, num_kv_heads in ((32, 8), (8, 2)):
            for num_tokens in (1, 7, 64, 256):
                with self.subTest(heads=(num_heads, num_kv_heads), tokens=num_tokens):
                    positions = torch.randint(
                        0, MAX_POS, (num_tokens,), device=self.device
                    )
                    self._check(rotary_emb, positions, num_heads, num_kv_heads)

    def test_mrope_nhd_cache(self):
        if not self.qwen3._has_fused_qk_norm_mrope:
            self.skipTest("aiter fused_qk_norm_mrope kernel unavailable")
        torch.manual_seed(0)
        rotary_emb = self._mrope()
        # (32, 8): Qwen3-VL-8B at TP1; (16, 2): Qwen3-VL-32B at TP4.
        for num_heads, num_kv_heads in ((32, 8), (16, 2)):
            for num_tokens in (1, 7, 64, 256):
                with self.subTest(heads=(num_heads, num_kv_heads), tokens=num_tokens):
                    positions = torch.randint(
                        0, MAX_POS, (3, num_tokens), device=self.device
                    )
                    self._check(rotary_emb, positions, num_heads, num_kv_heads)

    def test_cos_sin_cache_cast_is_reused(self):
        rotary_emb = self._mrope()
        self.assertEqual(rotary_emb.cos_sin_cache.dtype, torch.float32)
        first = rotary_emb.cos_sin_cache_as(torch.bfloat16)
        self.assertIs(rotary_emb.cos_sin_cache_as(torch.bfloat16), first)
        torch.testing.assert_close(
            first, rotary_emb.cos_sin_cache.to(torch.bfloat16), atol=0, rtol=0
        )
        rotary_emb._ensure_cos_sin_cache_length(2 * MAX_POS)
        grown = rotary_emb.cos_sin_cache_as(torch.bfloat16)
        self.assertIsNot(grown, first)
        self.assertEqual(grown.shape, rotary_emb.cos_sin_cache.shape)


if __name__ == "__main__":
    unittest.main()
