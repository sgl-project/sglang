"""HIP query RoPE fused into the K-store launch preserves both outputs."""

import math
import unittest
from itertools import product

import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import set_batched_rope
from sglang.kernels.ops.attention.dsv4.elementwise import (
    fused_k_norm_rope_flashmla,
    fused_rope_inplace,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, stage="jit-kernel-unit", runner_config="amd")

HEAD_DIM, ROPE_DIM, NOPE_DIM = 512, 64, 448


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFusedKNormRopeFlashMLA(CustomTestCase):
    @unittest.skipUnless(
        is_hip() and is_gfx95_supported(),
        "the query rope rides the HIP K launch; its bitwise parity with the flat rope"
        " kernel is claimed on gfx950 only",
    )
    def test_query_rope_in_the_k_launch(self):
        """With `q` the K launch must rope every query head's trailing ROPE_DIM bitwise
        like the flat rope kernel, leave the cache bytes and the nope part untouched,
        and rope rows without a slot."""
        dev = "cuda"
        page_size = 256
        for layout, (num_tokens, heads, pos_dtype, seed) in product(
            (KVLayout.V4, KVLayout.V41),
            ((1, 16, torch.int64, 0), (300, 16, torch.int32, 2)),
        ):
            with self.subTest(layout=layout, num_tokens=num_tokens, heads=heads):
                torch.manual_seed(seed)
                kv = torch.randn(num_tokens, HEAD_DIM, device=dev, dtype=torch.bfloat16)
                weight = (1 + 0.1 * torch.randn(HEAD_DIM, device=dev)).to(
                    torch.bfloat16
                )
                angles = torch.rand(8192, ROPE_DIM // 2, device=dev) * 2 * math.pi
                freqs_cis = torch.polar(torch.ones_like(angles), angles)
                positions = torch.randint(0, 8192, (num_tokens,), device=dev).to(
                    pos_dtype
                )
                out_loc = torch.randperm(4 * page_size, device=dev)[:num_tokens]
                out_loc = out_loc.to(torch.int32)
                if num_tokens > 2:
                    out_loc[1] = -1
                page_bytes = layout.page_bytes(page_size)
                cache = torch.zeros(4, page_bytes, device=dev, dtype=torch.uint8)
                cache_q = cache.clone()
                q = (torch.randn(num_tokens, heads, HEAD_DIM, device=dev) * 3).to(
                    torch.bfloat16
                )
                expected = q.clone()
                # The model's standalone query rope (batched flat kernel).
                set_batched_rope(True)
                fused_rope_inplace(
                    expected[..., -ROPE_DIM:], None, freqs_cis, positions
                )
                got = q.clone()
                fused_k_norm_rope_flashmla(
                    kv,
                    weight,
                    1e-6,
                    freqs_cis,
                    positions,
                    out_loc,
                    cache,
                    page_size,
                    layout=layout,
                )
                fused_k_norm_rope_flashmla(
                    kv,
                    weight,
                    1e-6,
                    freqs_cis,
                    positions,
                    out_loc,
                    cache_q,
                    page_size,
                    q=got,
                    layout=layout,
                )
                self.assertTrue(torch.equal(got, expected))
                self.assertTrue(torch.equal(got[..., :NOPE_DIM], q[..., :NOPE_DIM]))
                self.assertTrue(torch.equal(cache_q, cache))


if __name__ == "__main__":
    unittest.main()
