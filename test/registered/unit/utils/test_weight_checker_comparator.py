# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for sglang/srt/utils/weight_checker_comparator.py."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from sglang.srt.utils.weight_checker_comparator import (
    ComparableWeight,
    Fp8BlockComparable,
    compare_weights,
    select_comparable_weight,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare_quant_pair(expect_q, expect_s, actual_q, actual_s):
    return compare_weights(
        Fp8BlockComparable(expect_q, expect_s), Fp8BlockComparable(actual_q, actual_s)
    )


def _build_fp8_quant_pair(device: str = "cuda"):
    """Returns (qweight, fp32 scale, ue8m0-packed int32 scale) for one random weight."""
    weight_bf16 = torch.randn((256, 128), dtype=torch.bfloat16, device=device)
    block_size = [128, 128]
    qweight, sf_fp32 = quant_weight_ue8m0(
        weight_dequant=weight_bf16, weight_block_size=block_size
    )
    sf_packed_int32 = transform_scale_ue8m0(sf_fp32, mn=qweight.shape[-2])
    return qweight, sf_fp32, sf_packed_int32


# ---------------------------------------------------------------------------
# _quant_ulp
# ---------------------------------------------------------------------------


class TestQuantUlp(CustomTestCase):

    def test_matches_bruteforce_spacing_for_fp8(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            all_bits = torch.arange(256, dtype=torch.uint8).view(dtype)
            vals = all_bits.to(torch.float32)
            magnitudes = torch.unique(vals[torch.isfinite(vals) & (vals >= 0)])
            # Brute-force ULP: spacing to the next representable magnitude
            # (the largest magnitude reuses the spacing below it).
            spacing = magnitudes[1:] - magnitudes[:-1]
            expected = torch.cat([spacing, spacing[-1:]])
            got = ComparableWeight._quant_ulp(magnitudes.to(dtype))
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# compare_weights
# ---------------------------------------------------------------------------


class TestCompareQuantPair(CustomTestCase):
    """Chunked dequantized-space comparison of block-quantized pairs."""

    @staticmethod
    def _quantize(weight: torch.Tensor, scale_margin: float):
        """Blockwise 128x128 fp8 quantization with a tweakable scale convention."""
        n, k = weight.shape
        blocks = weight.float().view(n // 128, 128, k // 128, 128).permute(0, 2, 1, 3)
        scale = blocks.abs().amax(dim=(-1, -2)) / 448.0 * scale_margin
        q = (blocks / scale[:, :, None, None]).to(torch.float8_e4m3fn)
        q = q.permute(0, 2, 1, 3).reshape(n, k)
        return q, scale

    def setUp(self):
        torch.manual_seed(0)
        self.weight = torch.randn(256, 256, device="cuda") * 0.02
        self.e_q, self.e_s = self._quantize(self.weight, 1.0)
        self.a_q, self.a_s = self._quantize(self.weight, 1.001)

    def test_identical_pair_is_equal(self):
        equal, max_err, mean_err, num_exceed = _compare_quant_pair(
            self.e_q, self.e_s, self.e_q.clone(), self.e_s.clone()
        )
        self.assertTrue(equal)
        self.assertEqual((max_err, mean_err, num_exceed), (0.0, 0.0, 0))

    def test_ue8m0_packed_scale_equals_unpacked_scale(self):
        qweight, sf_fp32, sf_packed_int32 = _build_fp8_quant_pair()
        equal, *_ = _compare_quant_pair(qweight, sf_packed_int32, qweight, sf_fp32)
        self.assertTrue(equal)

    def test_two_quantizations_stay_within_ulp_tolerance(self):
        equal, max_err, mean_err, num_exceed = _compare_quant_pair(
            self.e_q, self.e_s, self.a_q, self.a_s
        )
        self.assertFalse(equal)
        self.assertGreater(max_err, 0.0)
        self.assertEqual(num_exceed, 0)

    def test_corruption_and_fp8_nan_exceed_tolerance(self):
        bad_q = self.a_q.clone().view(torch.uint8)
        bad_q[::50] += 8  # jumps a full binade; some bytes become fp8 NaN
        equal, max_err, mean_err, num_exceed = _compare_quant_pair(
            self.e_q, self.e_s, bad_q.view(torch.float8_e4m3fn), self.a_s
        )
        self.assertFalse(equal)
        self.assertGreater(num_exceed, 0)

    def test_chunked_result_matches_unchunked(self):
        reference = _compare_quant_pair(self.e_q, self.e_s, self.a_q, self.a_s)
        with patch("sglang.srt.utils.weight_checker_comparator.CHUNK_NUMEL", 128 * 128):
            chunked = _compare_quant_pair(self.e_q, self.e_s, self.a_q, self.a_s)
        self.assertEqual(chunked, reference)

    @staticmethod
    def _quantize_partial(weight: torch.Tensor, scale_margin: float):
        """128x128 block quant where the last block per dim may be partial."""
        n, k = weight.shape
        s_n, s_k = -(-n // 128), -(-k // 128)
        q = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=weight.device)
        scale = torch.empty(s_n, s_k, device=weight.device)
        for i in range(s_n):
            for j in range(s_k):
                blk = weight[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128].float()
                s = blk.abs().amax() / 448.0 * scale_margin
                s = s if s > 0 else weight.new_ones(())
                scale[i, j] = s
                q[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = (blk / s).to(
                    torch.float8_e4m3fn
                )
        return q, scale

    def test_partial_last_block_infers_true_block_size(self):
        # fused_qkv_a_proj_with_mqa out-dim is not a multiple of 128 (e.g. 2112 =
        # 16*128 + 64), so the last row-block is partial. ceil(dim/num_blocks)
        # would infer 125, misaligning scales; the true block size is 128.
        n, k = 3 * 128 + 64, 256
        weight = torch.randn(n, k, device="cuda") * 0.02
        e_q, e_s = self._quantize_partial(weight, 1.0)
        a_q, a_s = self._quantize_partial(weight, 1.001)
        self.assertEqual(list(e_s.shape), [4, 2])  # ceil(448/128)=4, 256/128=2
        self.assertEqual(Fp8BlockComparable._infer_block_size(e_q, e_s), [128, 128])
        equal, _, _, num_exceed = _compare_quant_pair(e_q, e_s, a_q, a_s)
        self.assertFalse(equal)
        self.assertEqual(num_exceed, 0)

    def test_3d_expert_tensor(self):
        q3 = self.e_q.reshape(2, 128, 256).contiguous()
        s3 = self.e_s.reshape(2, 1, 2)
        equal, *_ = _compare_quant_pair(q3, s3, q3.clone(), s3.clone())
        self.assertTrue(equal)


# ---------------------------------------------------------------------------
# select_comparable_weight
# ---------------------------------------------------------------------------


class TestSelectComparableWeight(CustomTestCase):

    def test_returns_none_when_not_a_quant_method(self):
        self.assertIsNone(select_comparable_weight(None))

    def test_returns_none_for_raw_safe_method(self):
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        # unquantized / int4 / mxfp8 all route to raw (None).
        fake = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
        self.assertIsNone(select_comparable_weight(fake))

    def test_raises_on_nvfp4(self):
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptFp4LinearMethod,
        )

        # nvfp4 has no ComparableWeight yet -> must raise, not silently raw-compare.
        fake = ModelOptFp4LinearMethod.__new__(ModelOptFp4LinearMethod)
        with self.assertRaises(NotImplementedError):
            select_comparable_weight(fake)


if __name__ == "__main__":
    unittest.main()
