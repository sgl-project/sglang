"""CPU tests for blockwise FP8 online weight quantization helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    per_block_no_cast_to_fp8,
    quant_weight_sf_fp32,
)
from sglang.srt.utils.common import ceil_div
from sglang.test.test_utils import CustomTestCase

BLOCK_SIZE = [128, 128]


def _dequant_blocks(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Reconstruct the (padded) tensor from fp8 values and per-block fp32 scales."""
    m, n = q.shape
    pm = ceil_div(m, 128) * 128
    pn = ceil_div(n, 128) * 128
    q_padded = torch.zeros((pm, pn), dtype=torch.float32)
    q_padded[:m, :n] = q.float()
    q_view = q_padded.view(pm // 128, 128, pn // 128, 128)
    sf_view = sf.view(pm // 128, 1, pn // 128, 1)
    deq = (q_view * sf_view).view(pm, pn)
    return deq[:m, :n]


class TestPerBlockNoCastToFp8(CustomTestCase):
    def test_output_dtype_and_scale_shape_aligned(self):
        x = torch.randn(256, 512, dtype=torch.bfloat16)
        q, sf = per_block_no_cast_to_fp8(x)

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(sf.dtype, torch.float32)
        self.assertEqual(sf.shape, (2, 4))
        self.assertTrue(q.is_contiguous())

    def test_scale_shape_unaligned_dims(self):
        # 200 -> 2 blocks of 128, 300 -> 3 blocks of 128
        x = torch.randn(200, 300, dtype=torch.bfloat16)
        q, sf = per_block_no_cast_to_fp8(x)

        self.assertEqual(q.shape, (200, 300))
        self.assertEqual(sf.shape, (2, 3))

    def test_scale_matches_per_block_amax(self):
        x = torch.randn(128, 128, dtype=torch.bfloat16)
        _, sf = per_block_no_cast_to_fp8(x)

        expected = x.abs().float().amax().clamp(1e-4) / 448.0
        self.assertEqual(sf.shape, (1, 1))
        self.assertTrue(torch.allclose(sf.flatten(), expected.flatten(), atol=1e-6))

    def test_roundtrip_dequant_accuracy(self):
        torch.manual_seed(0)
        x = torch.randn(256, 256, dtype=torch.bfloat16)
        q, sf = per_block_no_cast_to_fp8(x)

        deq = _dequant_blocks(q, sf)
        # fp8 e4m3 has ~2 decimal digits of precision; per-block scaling keeps
        # relative error bounded. Compare against the bf16 reference.
        ref = x.float()
        rel = (deq - ref).abs() / ref.abs().clamp(min=1e-2)
        self.assertLess(rel.mean().item(), 0.1)

    def test_zero_block_clamped(self):
        x = torch.zeros(128, 128, dtype=torch.bfloat16)
        q, sf = per_block_no_cast_to_fp8(x)

        # amax is clamped to 1e-4, so scale is 1e-4 / 448 (non-zero, finite).
        self.assertTrue(torch.isfinite(sf).all())
        self.assertGreater(sf.min().item(), 0.0)
        self.assertTrue((q.float() == 0).all())


class TestQuantWeightSfFp32(CustomTestCase):
    def test_2d_weight(self):
        n, k = 256, 384
        w = torch.randn(n, k, dtype=torch.bfloat16)
        out_w, out_s = quant_weight_sf_fp32(w, BLOCK_SIZE)

        self.assertEqual(out_w.dtype, torch.float8_e4m3fn)
        self.assertEqual(out_w.shape, (n, k))
        self.assertEqual(out_s.shape, (ceil_div(n, 128), ceil_div(k, 128)))

    def test_3d_weight_batch_dims_preserved(self):
        e, n, k = 4, 256, 512
        w = torch.randn(e, n, k, dtype=torch.bfloat16)
        out_w, out_s = quant_weight_sf_fp32(w, BLOCK_SIZE)

        self.assertEqual(out_w.shape, (e, n, k))
        self.assertEqual(out_s.shape, (e, ceil_div(n, 128), ceil_div(k, 128)))

    def test_rejects_non_bf16(self):
        w = torch.randn(128, 128, dtype=torch.float16)
        with self.assertRaises(AssertionError):
            quant_weight_sf_fp32(w, BLOCK_SIZE)

    def test_rejects_unsupported_block_size(self):
        w = torch.randn(128, 128, dtype=torch.bfloat16)
        with self.assertRaises(AssertionError):
            quant_weight_sf_fp32(w, [128, 256])


if __name__ == "__main__":
    unittest.main(verbosity=3)
