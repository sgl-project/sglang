"""AITER fused FP8 DSA writer parity on GLM-5.2 geometry."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.fp8_fused_writer_hip import (
    aiter_fused_fp8_qk_write,
    prepare_aiter_rope_caches,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

HEADS = 32
HEAD_DIM = 128
ROPE_DIM = 64
PAGE_SIZE = 64


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    return (
        str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"
    )


def _ue8m0_quant(x: torch.Tensor):
    """CPU/GPU reference for per-1x128 ue8m0 FP8 used by the fused writer."""
    fp8_max = 224.0 if str(fp8_dtype).endswith("fnuz") else 448.0
    amax = x.detach().float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))
    y = (x.float() / scale).clamp(-fp8_max, fp8_max)
    return y.to(fp8_dtype), scale.squeeze(-1)


@unittest.skipUnless(_gfx950(), "gfx950 only")
class TestFp8FusedDsaWriter(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.norm = LayerNorm(HEAD_DIM, dtype=torch.float32).cuda()
        cls.rope = get_rope_wrapper(
            ROPE_DIM,
            rotary_dim=ROPE_DIM,
            max_position=4096,
            base=1_000_000,
            is_neox_style=False,
            device="cuda",
        )
        if not (hasattr(cls.rope, "cos_cache") and hasattr(cls.rope, "sin_cache")):
            raise unittest.SkipTest("AITER split cos/sin RoPE caches are required")
        cls.cos_cache, cls.sin_cache = prepare_aiter_rope_caches(
            cls.rope.cos_cache,
            cls.rope.sin_cache,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        cls.rope.cos_cache = cls.rope.cos_cache.to(device="cuda", dtype=torch.bfloat16)
        cls.rope.sin_cache = cls.rope.sin_cache.to(device="cuda", dtype=torch.bfloat16)

    def _inputs(self, tokens: int):
        torch.manual_seed(2026 + tokens)
        q = torch.randn(tokens, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(tokens, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        weights = torch.randn(tokens, HEADS, dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(tokens, dtype=torch.int64, device="cuda")
        slots = torch.arange(tokens, dtype=torch.int64, device="cuda")
        pages = max(1, (tokens + PAGE_SIZE - 1) // PAGE_SIZE)
        return q, k, weights, positions, slots, pages

    def _cache(self, pages: int, fill: int = 0):
        return torch.full(
            (pages, PAGE_SIZE, HEAD_DIM + 4),
            fill,
            dtype=torch.uint8,
            device="cuda",
        )

    def _separate_no_hadamard(self, q, k, weights, positions, slots, cache):
        q_ref = q.clone()
        k_ref = self.norm(k)
        q_rope = q_ref[..., :ROPE_DIM]
        k_rope = k_ref[..., :ROPE_DIM]
        self.rope(positions, q_rope, k_rope)
        q_fp8, q_scale = _ue8m0_quant(q_ref)

        from aiter.ops.cache import indexer_k_quant_and_cache

        indexer_k_quant_and_cache(
            k_ref,
            cache.view(fp8_dtype),
            slots,
            128,
            "ue8m0",
            preshuffle=True,
        )
        weights_out = (
            weights.float() * q_scale.squeeze(-1) * (HEAD_DIM**-0.5) * (HEADS**-0.5)
        )
        return q_fp8, weights_out

    def _fused(self, q, k, weights, positions, slots, cache):
        q_out = torch.empty_like(q, dtype=fp8_dtype)
        weights_out = torch.empty_like(weights, dtype=torch.float32)
        aiter_fused_fp8_qk_write(
            q,
            q_out,
            weights,
            weights_out,
            k,
            cache.view(fp8_dtype),
            slots,
            self.norm.weight,
            self.norm.bias,
            positions,
            self.cos_cache,
            self.sin_cache,
            self.norm.variance_epsilon,
            128,
            "ue8m0",
            (HEAD_DIM**-0.5) * (HEADS**-0.5),
            preshuffle=True,
            is_neox=False,
            compute_all_q_rope=True,
        )
        return q_out, weights_out

    def test_matches_separate_no_hadamard_path(self):
        for tokens in (1, 4, 8, 24, 48):
            with self.subTest(tokens=tokens):
                q, k, weights, positions, slots, pages = self._inputs(tokens)
                ref_cache = self._cache(pages)
                got_cache = self._cache(pages)
                _, ref_weights = self._separate_no_hadamard(
                    q, k, weights, positions, slots, ref_cache
                )
                got_q, got_weights = self._fused(
                    q, k, weights, positions, slots, got_cache
                )
                self.assertTrue(torch.isfinite(got_q.float()).all())
                self.assertTrue(torch.isfinite(got_weights).all())
                torch.testing.assert_close(
                    got_weights, ref_weights, rtol=2e-2, atol=2e-2
                )
                self.assertTrue(torch.equal(got_cache, ref_cache))

    def test_negative_slot_computes_q_but_does_not_write_k(self):
        q, k, weights, positions, slots, pages = self._inputs(4)
        slots.fill_(-1)
        cache = self._cache(pages, fill=0xA5)
        before = cache.clone()
        q_out, weights_out = self._fused(q, k, weights, positions, slots, cache)
        self.assertTrue(torch.equal(cache, before))
        self.assertTrue(torch.isfinite(q_out.float()).all())
        self.assertTrue(torch.isfinite(weights_out).all())


if __name__ == "__main__":
    unittest.main()
