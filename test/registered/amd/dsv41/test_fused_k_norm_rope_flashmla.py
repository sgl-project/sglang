"""The fused K RMSNorm + RoPE + fp8 store into the FlashMLA paged cache must match a torch reference of the same quantization."""

import math
import unittest

import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import set_batched_rope
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged_ref,
)
from sglang.kernels.ops.attention.dsv4.elementwise import (
    fused_k_norm_rope_flashmla,
    fused_rope_inplace,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=40, stage="jit-kernel-unit", runner_config="amd")

HEAD_DIM, ROPE_DIM, NOPE_DIM, BLOCK = 512, 64, 448, 64
FP8_MAX = 448.0


def _reference(kv, weight, eps, freqs_cis, positions):
    x = kv.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    pairs = x[:, NOPE_DIM:].view(-1, ROPE_DIM // 2, 2)
    f = freqs_cis[positions]
    rope = torch.stack(
        [
            pairs[..., 0] * f.real - pairs[..., 1] * f.imag,
            pairs[..., 0] * f.imag + pairs[..., 1] * f.real,
        ],
        dim=-1,
    ).reshape(-1, ROPE_DIM)
    nope = x[:, :NOPE_DIM].view(-1, NOPE_DIM // BLOCK, BLOCK)
    amax = nope.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    nope_q = (nope / scale).to(torch.float8_e4m3fn).float() * scale
    return torch.cat([nope_q.view(-1, NOPE_DIM), rope.to(torch.bfloat16).float()], -1)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFusedKNormRopeFlashMLA(CustomTestCase):
    def _run(self, num_tokens, page_size, seed):
        torch.manual_seed(seed)
        dev = "cuda"
        kv = torch.randn(num_tokens, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        # Force one element per block into the top fp8 binade after scaling.
        kv[:, ::BLOCK] *= 4.0
        weight = (1 + 0.1 * torch.randn(HEAD_DIM, device=dev)).to(torch.bfloat16)
        eps = 1e-20
        angles = torch.rand(4096, ROPE_DIM // 2, device=dev) * 2 * math.pi
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        positions = torch.randint(0, 4096, (num_tokens,), device=dev)
        out_loc = torch.randperm(2 * page_size, device=dev)[:num_tokens].to(torch.int32)
        page_bytes = -(-584 * page_size // 576) * 576
        cache = torch.zeros(2, page_bytes, device=dev, dtype=torch.uint8)

        fused_k_norm_rope_flashmla(
            kv, weight, eps, freqs_cis, positions, out_loc, cache, page_size
        )
        got = dequantize_k_cache_paged_ref(cache, out_loc, page_size)
        got = got.float().reshape(num_tokens, -1)[:, :HEAD_DIM]
        expected = _reference(kv, weight, eps, freqs_cis, positions)
        return got, expected

    def test_matches_torch_quantization(self):
        for num_tokens, page_size, seed in ((300, 256, 1), (7, 64, 2)):
            with self.subTest(num_tokens=num_tokens, page_size=page_size):
                got, expected = self._run(num_tokens, page_size, seed)
                # payload and scale are exact; the rope tail is bf16 on both sides, its only freedom fp32 rounding
                torch.testing.assert_close(
                    got[:, :NOPE_DIM], expected[:, :NOPE_DIM], atol=0.0, rtol=0.0
                )
                torch.testing.assert_close(
                    got[:, NOPE_DIM:], expected[:, NOPE_DIM:], atol=2e-2, rtol=2e-2
                )

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
        for num_tokens, heads, pos_dtype, seed in (
            (1, 16, torch.int64, 0),
            (300, 16, torch.int32, 2),
        ):
            with self.subTest(num_tokens=num_tokens, heads=heads):
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
                page_bytes = -(-584 * page_size // 576) * 576
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
                    kv, weight, 1e-6, freqs_cis, positions, out_loc, cache, page_size
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
                )
                self.assertTrue(torch.equal(got, expected))
                self.assertTrue(torch.equal(got[..., :NOPE_DIM], q[..., :NOPE_DIM]))
                self.assertTrue(torch.equal(cache_q, cache))


if __name__ == "__main__":
    unittest.main()
