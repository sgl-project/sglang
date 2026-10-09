"""DSA packed-cache and HIP TileLang kernels must accept GLM's 256+0 geometry without changing 512+64."""

import math
import unittest

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    _infer_dsa_dims,
    dequantize_k_cache,
    dequantize_k_cache_paged,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the zero-tail specialization only exists in the HIP TileLang kernels
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestPackedRowInference(CustomTestCase):
    def test_packed_width_maps_to_one_layout(self):
        """A 64-wide RoPE tail adds 128 bytes and a NoPE tile adds 132, so no width is ambiguous."""
        for packed_width, layout in {
            264: (256, 0),
            392: (256, 64),
            528: (512, 0),
            656: (512, 64),
        }.items():
            self.assertEqual(_infer_dsa_dims(packed_width), layout)
        with self.assertRaises(ValueError):
            _infer_dsa_dims(265)


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestScaledCacheLayouts(CustomTestCase):
    def test_quant_dequant_round_trip_and_paged_gather(self):
        """A 264-byte row must dequantize to 256 columns and gather by page like the 656-byte row."""
        torch.manual_seed(7)
        for dim_nope, dim_rope in ((256, 0), (512, 64)):
            with self.subTest(dim_nope=dim_nope, dim_rope=dim_rope):
                source = torch.randn(
                    8, 1, 1, dim_nope + dim_rope, device="cuda", dtype=torch.bfloat16
                )
                packed = quantize_k_cache(source, dv=dim_nope)
                restored = dequantize_k_cache(packed, dv=dim_nope)
                torch.testing.assert_close(restored, source, atol=0.08, rtol=0.08)

                pages = torch.tensor([7, 1, 1, 4], device="cuda", dtype=torch.int32)
                gathered = dequantize_k_cache_paged(packed, pages)
                expected = restored.view(8, 1, -1)[pages]
                torch.testing.assert_close(gathered, expected, atol=0, rtol=0)


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


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "the zero-tail TileLang specialization is compiled for gfx950",
)
class TestTileLangDSAZeroRope(CustomTestCase):
    def _assert_matches_torch(self, use_fp8, d_v, d_tail):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            FP8_DTYPE,
            tilelang_sparse_fwd,
        )

        torch.manual_seed(7)
        tokens, heads, topk = 17, 64, 2112
        dim = d_v + d_tail
        q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(topk, 1, dim, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(topk, device="cuda", dtype=torch.int32)
        indices = indices.view(1, 1, topk).expand(tokens, -1, -1).clone()
        indices[..., 2051:] = -1  # padded rows must be masked, not gathered from slot 0
        if use_fp8:
            q = q.to(FP8_DTYPE)
            kv = kv.to(FP8_DTYPE)

        scale = 1.0 / math.sqrt(dim)
        expected = _torch_sparse_attention(q, kv, indices, scale, d_v)
        # the HIP combine kernel returns [batch=1, tokens, heads, d_v]
        actual = tilelang_sparse_fwd(q, kv, indices, scale, d_v=d_v).squeeze(0)
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.20 if use_fp8 else 0.04,
            rtol=0.12 if use_fp8 else 0.04,
        )

    def test_bf16_zero_rope_matches_torch(self):
        """Before the fix the BF16 partial kernel emitted zero-extent tail copies and failed to compile."""
        self._assert_matches_torch(use_fp8=False, d_v=256, d_tail=0)

    def test_fp8_zero_rope_matches_torch(self):
        """Before the fix the FP8 partial kernel asserted d_v == 512 and read four NoPE tiles."""
        self._assert_matches_torch(use_fp8=True, d_v=256, d_tail=0)

    def test_tail64_layout_unchanged_by_zero_tail_specialization(self):
        """The 512+64 path was rewritten into has_tail/num_main_tiles branches and must still match."""
        for use_fp8 in (False, True):
            with self.subTest(use_fp8=use_fp8):
                self._assert_matches_torch(use_fp8=use_fp8, d_v=512, d_tail=64)


if __name__ == "__main__":
    unittest.main()
