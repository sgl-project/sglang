"""
Unit tests for sglang.srt.hardware_backend.npu.attention.mla_preprocess.
"""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    is_fia_nz,
    is_mla_preprocess_enabled,
    round_up,
    trans_rope_weight,
    transdata,
)


class TestRoundUp(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(round_up(16, 16), 16)
        self.assertEqual(round_up(32, 16), 32)

    def test_round_up_to_block(self):
        self.assertEqual(round_up(1, 16), 16)
        self.assertEqual(round_up(10, 16), 16)
        self.assertEqual(round_up(17, 16), 32)

    def test_zero_value(self):
        self.assertEqual(round_up(0, 16), 0)

    def test_zero_align_returns_zero(self):
        self.assertEqual(round_up(100, 0), 0)
        self.assertEqual(round_up(0, 0), 0)

    def test_negative_align(self):
        # Pin current behavior for negative align.
        self.assertEqual(round_up(10, -4), 8)


class TestTransdata(unittest.TestCase):
    def test_aligned_32x32_default_block(self):
        mat = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
        nz = transdata(mat)
        self.assertEqual(nz.shape, (2, 32, 16))

    def test_aligned_element_mapping(self):
        rows, cols = 16, 32
        mat = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
        bs = (16, 16)
        nz = transdata(mat, block_size=bs)

        # NZ layout: element (r, c) lands at nz[c // bs1, r, c % bs1].
        for r in range(rows):
            for c in range(cols):
                expected = mat[r, c]
                actual = nz[c // bs[1], r, c % bs[1]]
                self.assertEqual(
                    actual.item(),
                    expected.item(),
                    f"mismatch at (r={r}, c={c})",
                )

    def test_padding_for_non_aligned_shape(self):
        mat = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        nz = transdata(mat, block_size=(16, 16))
        # c_blocks = ceil(10/16) = 1, r_padded = 16, bs1 = 16
        self.assertEqual(nz.shape, (1, 16, 16))

        # Original elements are preserved in the padded NZ layout.
        for r in range(10):
            for c in range(10):
                self.assertEqual(
                    nz[c // 16, r, c % 16].item(),
                    mat[r, c].item(),
                    f"padded mismatch at (r={r}, c={c})",
                )

    def test_non_square_matrix(self):
        mat = torch.arange(16 * 48, dtype=torch.float32).reshape(16, 48)
        nz = transdata(mat, block_size=(16, 16))
        # c_blocks = 48//16 = 3, r = 16, bs1 = 16
        self.assertEqual(nz.shape, (3, 16, 16))

    def test_custom_block_size(self):
        mat = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
        nz = transdata(mat, block_size=(4, 4))
        self.assertEqual(nz.shape, (2, 8, 4))

    def test_3d_input_raises(self):
        mat = torch.arange(2 * 16 * 16, dtype=torch.float32).reshape(2, 16, 16)
        with self.assertRaises(RuntimeError):
            transdata(mat, block_size=(16, 16))


class TestTransRopeWeight(unittest.TestCase):
    def test_basic_reorder(self):
        # 8 rows, rope_dim=4 → last 4 rows are the RoPE region.
        weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4).clone()
        original = weight.clone()
        result = trans_rope_weight(weight, rope_dim=4)

        # RoPE region (rows 4-7) reordered: even indices first → [4,6,5,7].
        rope_region = original[4:8]
        expected_rope = torch.stack(
            [rope_region[0], rope_region[2], rope_region[1], rope_region[3]]
        )
        self.assertTrue(torch.equal(result[4:8], expected_rope))

    def test_non_rope_region_unchanged(self):
        weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4).clone()
        original = weight.clone()
        trans_rope_weight(weight, rope_dim=4)
        self.assertTrue(torch.equal(weight[0:4], original[0:4]))

    def test_rope_dim_zero_reorders_all(self):
        weight = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8).clone()
        original = weight.clone()
        result = trans_rope_weight(weight, rope_dim=0)
        # Rows [0,1,2,3] → [0,2,1,3] (even first, then odd)
        expected = torch.stack([original[0], original[2], original[1], original[3]])
        self.assertTrue(torch.equal(result, expected))

    def test_full_rope_dim(self):
        weight = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2).clone()
        original = weight.clone()
        result = trans_rope_weight(weight, rope_dim=4)
        # rows [0,1,2,3] → [0,2,1,3]
        expected = torch.stack([original[0], original[2], original[1], original[3]])
        self.assertTrue(torch.equal(result, expected))

    def test_3d_weight(self):
        # Shape (2, 8, 4): 8 rows per expert, rope_dim=4
        weight = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4).clone()
        original = weight.clone()
        result = trans_rope_weight(weight, rope_dim=4)

        for e in range(2):
            rope_region = original[e, 4:8]
            expected_rope = torch.stack(
                [rope_region[0], rope_region[2], rope_region[1], rope_region[3]]
            )
            self.assertTrue(torch.equal(result[e, 4:8], expected_rope))


class TestIsMlaPreprocessEnabled(unittest.TestCase):
    def setUp(self):
        is_mla_preprocess_enabled.cache_clear()

    def tearDown(self):
        is_mla_preprocess_enabled.cache_clear()

    def test_not_set_returns_false(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_NPU_USE_MLAPO", None)
            self.assertFalse(is_mla_preprocess_enabled())

    def test_set_to_one_returns_true(self):
        with patch.dict(os.environ, {"SGLANG_NPU_USE_MLAPO": "1"}):
            self.assertTrue(is_mla_preprocess_enabled())

    def test_set_to_zero_returns_false(self):
        with patch.dict(os.environ, {"SGLANG_NPU_USE_MLAPO": "0"}):
            self.assertFalse(is_mla_preprocess_enabled())

    def test_set_to_true_returns_true(self):
        with patch.dict(os.environ, {"SGLANG_NPU_USE_MLAPO": "true"}):
            self.assertTrue(is_mla_preprocess_enabled())


class TestIsFiaNz(unittest.TestCase):
    def setUp(self):
        is_mla_preprocess_enabled.cache_clear()
        is_fia_nz.cache_clear()

    def tearDown(self):
        is_mla_preprocess_enabled.cache_clear()
        is_fia_nz.cache_clear()

    def test_not_set_returns_false(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_USE_FIA_NZ", None)
            os.environ.pop("SGLANG_NPU_USE_MLAPO", None)
            self.assertFalse(is_fia_nz())

    def test_fia_nz_with_mlapo_returns_true(self):
        with patch.dict(
            os.environ,
            {"SGLANG_USE_FIA_NZ": "1", "SGLANG_NPU_USE_MLAPO": "1"},
        ):
            self.assertTrue(is_fia_nz())

    def test_fia_nz_without_mlapo_raises(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_NPU_USE_MLAPO", None)
            os.environ["SGLANG_USE_FIA_NZ"] = "1"
            with self.assertRaises(AssertionError):
                is_fia_nz()


if __name__ == "__main__":
    unittest.main()
