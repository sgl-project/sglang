import unittest

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    qwen3_apply_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.models.dits.ideogram import (
    _ideogram_rope,
    _ideogram_swiglu,
)


def _make_qk_cos_sin(batch, seq, heads, head_dim, device):
    q = torch.randn(batch, seq, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seq, heads, head_dim, device=device, dtype=torch.bfloat16)
    # Qwen3-style full-span duplicated-halves cos/sin, row-broadcast over heads.
    half = head_dim // 2
    freqs = torch.randn(batch, seq, 1, half, device=device) * 3.0
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
    return q, k, cos, sin


class TestIdeogramRopeFusion(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_rope_matches_eager(self):
        torch.manual_seed(0)
        for batch, seq, heads, head_dim in [(2, 257, 16, 128), (1, 64, 3, 64)]:
            q, k, cos, sin = _make_qk_cos_sin(batch, seq, heads, head_dim, "cuda")
            q_ref, k_ref = qwen3_apply_rotary_pos_emb(q, k, cos, sin)
            q_fused, k_fused = _ideogram_rope(q, k, cos, sin)
            self.assertTrue(torch.equal(q_ref, q_fused))
            self.assertTrue(torch.equal(k_ref, k_fused))

    def test_eager_fallback_cpu(self):
        torch.manual_seed(1)
        q, k, cos, sin = _make_qk_cos_sin(1, 9, 2, 32, torch.device("cpu"))
        q_ref, k_ref = qwen3_apply_rotary_pos_emb(q, k, cos, sin)
        q_out, k_out = _ideogram_rope(q, k, cos, sin)
        self.assertTrue(torch.equal(q_ref, q_out))
        self.assertTrue(torch.equal(k_ref, k_out))


class TestIdeogramSwigluFusion(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_swiglu_matches_eager(self):
        torch.manual_seed(0)
        a = torch.randn(2, 513, 3584, device="cuda", dtype=torch.bfloat16) * 4
        b = torch.randn(2, 513, 3584, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(torch.equal(F.silu(a) * b, _ideogram_swiglu(a, b)))

    def test_eager_fallback_cpu(self):
        torch.manual_seed(1)
        a = torch.randn(3, 8, dtype=torch.float32)
        b = torch.randn(3, 8, dtype=torch.float32)
        self.assertTrue(torch.equal(F.silu(a) * b, _ideogram_swiglu(a, b)))


if __name__ == "__main__":
    unittest.main()
