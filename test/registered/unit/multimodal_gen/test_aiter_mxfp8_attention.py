"""MXFP8 diffusion attention on gfx1250 (aiter fmha_fwd_mxfp8_asm); skipped elsewhere."""

import importlib.util
import unittest

import torch

from sglang.srt.utils import is_gfx1250_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

HEAD_SIZE = 128
_ON_GFX1250 = (
    importlib.util.find_spec("aiter") is not None
    and torch.cuda.is_available()
    and is_gfx1250_supported()
)


def _qkv(seq_q: int, seq_k: int, num_heads: int = 4):
    torch.manual_seed(0)
    shape_q = (1, seq_q, num_heads, HEAD_SIZE)
    shape_kv = (1, seq_k, num_heads, HEAD_SIZE)
    q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _reference(q, k, v):
    out = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (q, k, v)), scale=HEAD_SIZE**-0.5
    )
    return out.transpose(1, 2)


@unittest.skipUnless(_ON_GFX1250, "fmha_fwd_mxfp8_asm ships for gfx1250 only")
class TestGfx1250MXFP8Attention(unittest.TestCase):
    def setUp(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_mxfp8 import (
            mxfp8_attention,
        )

        self.mxfp8_attention = mxfp8_attention

    def test_matches_fp32_reference_with_and_without_kv_padding(self):
        # 1000 keys pad to 1024 and need the LSE correction; 512 keys do not.
        for seq_q, seq_k in [(1000, 1000), (4000, 512)]:
            with self.subTest(seq_q=seq_q, seq_k=seq_k):
                q, k, v = _qkv(seq_q, seq_k)
                out = self.mxfp8_attention(q, k, v, HEAD_SIZE**-0.5)
                cos = torch.nn.functional.cosine_similarity(
                    out.float().flatten(), _reference(q, k, v).flatten(), dim=0
                )
                self.assertGreater(cos.item(), 0.995)

    def test_runs_inside_an_inductor_compiled_function(self):
        q, k, v = _qkv(256, 1000)

        def block(q, k, v):
            return self.mxfp8_attention(q * 1.0, k, v, HEAD_SIZE**-0.5) + 1.0

        torch._dynamo.reset()
        out = torch.compile(block)(q, k, v)
        torch.testing.assert_close(out, block(q, k, v), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
