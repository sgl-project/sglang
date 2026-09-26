"""Unit tests for the INT8 HiCache record codec.

Covers the packed layout, quantiser bounds and the degenerate inputs that a
running server actually produces (all-zero heads, near-zero heads, outliers,
mixed magnitudes). No CUDA and no SGLang runtime needed beyond plain torch, so
this runs anywhere the codec imports.
"""

import unittest

import torch

from sglang.srt.mem_cache.pool_host.int8_codec import (
    ALIGNMENT_BYTES,
    AMAX_FLOOR,
    PADDING_BYTES,
    PAYLOAD_BYTES,
    QUANT_MAX,
    ROW_BYTES,
    SCALE_BYTES,
    SCALE_FLOOR,
    SCALE_OFFSET,
    bytes_per_token,
    check_layout,
    compute_scales,
    decode_records,
    encode_rows,
    quantize_rows,
    write_record,
)
from sglang.test.test_utils import CustomTestCase

HEAD_NUM = 8
HEAD_DIM = 128


def _rand(rows, *, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((rows, HEAD_NUM, HEAD_DIM), generator=g) * scale).to(
        torch.bfloat16
    )


class TestRecordLayout(CustomTestCase):
    def test_row_is_1152_bytes_and_mover_aligned(self):
        self.assertEqual(ROW_BYTES, 1152)
        self.assertEqual(ROW_BYTES, 9 * ALIGNMENT_BYTES)
        self.assertEqual(ROW_BYTES % ALIGNMENT_BYTES, 0)

    def test_sub_regions(self):
        self.assertEqual(PAYLOAD_BYTES, 1024)
        self.assertEqual(SCALE_BYTES, 16)
        self.assertEqual(PADDING_BYTES, 112)
        self.assertEqual(SCALE_OFFSET, 1024)
        self.assertEqual(PAYLOAD_BYTES + SCALE_BYTES + PADDING_BYTES, ROW_BYTES)

    def test_bytes_per_token_matches_project_numbers(self):
        # Qwen3-8B TP=1: 36 layers, 8 KV heads, head_dim 128.
        self.assertEqual(bytes_per_token(36), 82_944)
        baseline = 2 * 36 * 8 * 128 * 2
        self.assertEqual(baseline, 147_456)
        self.assertAlmostEqual(baseline / bytes_per_token(36), 1.7777778, places=6)

    def test_check_layout_accepts_qwen3_8b(self):
        check_layout(HEAD_NUM, HEAD_DIM, 2)

    def test_check_layout_rejects_wrong_geometry(self):
        for head_num, head_dim in ((16, 128), (8, 64), (4, 128)):
            with self.assertRaises(ValueError):
                check_layout(head_num, head_dim, 2)

    def test_check_layout_rejects_non_two_byte_source(self):
        with self.assertRaises(ValueError):
            check_layout(HEAD_NUM, HEAD_DIM, 1)


class TestQuantiser(CustomTestCase):
    def test_payload_stays_in_symmetric_int8_range(self):
        payload, _ = encode_rows(_rand(64, seed=1))
        self.assertEqual(payload.dtype, torch.int8)
        self.assertGreaterEqual(int(payload.min()), -QUANT_MAX)
        self.assertLessEqual(int(payload.max()), QUANT_MAX)

    def test_scale_is_absmax_over_127(self):
        x = _rand(64, seed=2)
        _, scales = encode_rows(x)
        expected = (x.abs().amax(dim=-1).clamp_min(AMAX_FLOOR) / QUANT_MAX).to(
            torch.bfloat16
        )
        self.assertTrue(torch.equal(scales, expected))

    def test_absmax_element_quantises_to_full_scale(self):
        x = torch.zeros((16, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
        x[:, :, 1] = 0.25
        payload, _ = encode_rows(x)
        self.assertEqual(int(payload[:, :, 1].abs().min()), QUANT_MAX)

    def test_zero_head_gets_safe_scale_and_encodes_zero(self):
        x = torch.zeros((4, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
        payload, scales = encode_rows(x)
        self.assertTrue(bool((scales > 0).all()))
        self.assertTrue(bool(torch.isfinite(scales).all()))
        self.assertTrue(bool((payload == 0).all()))
        records = write_record(x, torch.zeros((4, ROW_BYTES), dtype=torch.uint8))
        restored = decode_records(
            records, head_num=HEAD_NUM, head_dim=HEAD_DIM, dtype=torch.bfloat16
        )
        self.assertTrue(torch.equal(restored, x))

    def test_scale_floor_is_normal_not_subnormal(self):
        # A subnormal scale would round differently per backend.
        self.assertGreater(SCALE_FLOOR, 2.0**-126)
        self.assertGreater(float(torch.tensor(SCALE_FLOOR, dtype=torch.bfloat16)), 0)

    def test_tiny_head_encodes_zero_without_nan(self):
        x = torch.full((2, HEAD_NUM, HEAD_DIM), 1e-40, dtype=torch.bfloat16)
        payload, scales = encode_rows(x)
        self.assertTrue(bool(torch.isfinite(scales).all()))
        self.assertTrue(bool((payload == 0).all()))

    def test_mixed_zero_and_nonzero_heads_coexist(self):
        x = _rand(8, seed=3)
        x[:, 0, :] = 0.0
        x[:, 5, :] = 0.0
        payload, scales = encode_rows(x)
        self.assertTrue(bool((payload[:, 0, :] == 0).all()))
        self.assertTrue(bool((payload[:, 5, :] == 0).all()))
        self.assertTrue(bool((payload[:, 1, :] != 0).any()))
        self.assertTrue(bool((scales > 0).all()))

    def test_float32_quotient_matches_exact_float64_reference(self):
        """The quotient must be correctly rounded; bf16 division would not be."""
        x = _rand(256, seed=4)
        payload, scales = encode_rows(x)
        reference = (
            torch.round(x.double() / scales.double().unsqueeze(-1))
            .clamp_(-QUANT_MAX, QUANT_MAX)
            .to(torch.int8)
        )
        self.assertEqual(int((payload != reference).sum()), 0)

    def test_bf16_quotient_would_differ(self):
        """Guard the reason the widening exists (regression on this decision)."""
        x = _rand(512, seed=5)
        scales = compute_scales(x)
        bf16_q = torch.round(x / scales.unsqueeze(-1)).clamp_(-QUANT_MAX, QUANT_MAX)
        f32_q = quantize_rows(x, scales).float()
        self.assertGreater(int((bf16_q != f32_q).sum()), 0)


class TestReconstructionBound(CustomTestCase):
    """``|x_hat - x| <= (0.5 + 2**-8) * s + 2**-8 * |x_hat|``.

    The ``0.5 * s`` part is exact round-to-nearest. The ``2**-8 * s`` part is the
    BF16 scale's own rounding error, amplified by ``|x / s| <= 127`` (this is why
    the clamp to 127 is required). The ``2**-8 * |x_hat|`` part is the BF16
    output dtype, since L1 attention consumes BF16, and it dominates at large
    magnitudes.
    """

    def _check(self, x, label):
        records = write_record(x, torch.zeros((x.shape[0], ROW_BYTES), dtype=torch.uint8))
        restored = decode_records(
            records, head_num=HEAD_NUM, head_dim=HEAD_DIM, dtype=torch.bfloat16
        )
        _, scales = encode_rows(x)
        s = scales.float().unsqueeze(-1)
        bound = (0.5 + 2**-8) * s + 2**-8 * restored.float().abs()
        err = (restored.float() - x.float()).abs()
        violations = int((err > bound).sum())
        self.assertEqual(
            violations, 0, f"{label}: {violations} elements exceed the bound"
        )

    def test_random(self):
        self._check(_rand(128, seed=6), "random")

    def test_small_and_large(self):
        self._check(_rand(64, seed=7, scale=1e-3), "small")
        self._check(_rand(64, seed=8, scale=1e3), "large")

    def test_mixed_magnitudes(self):
        x = _rand(64, seed=9)
        x[0, :, :] = 0.0
        x[1, 0, :] = 1e-30
        x[2, 1, :] = 1e4
        x[3] *= 1e-8
        self._check(x, "mixed")

    def test_exact_quantiser_part_is_within_half_scale(self):
        x = _rand(128, seed=10)
        payload, scales = encode_rows(x)
        exact = payload.double() * scales.double().unsqueeze(-1)
        s = scales.float().unsqueeze(-1)
        excess = (exact.float() - x.float()).abs() - s / 2
        self.assertLessEqual(float(excess.max()), 1e-6)


class TestRecordPacking(CustomTestCase):
    def test_payload_and_scales_land_at_the_right_offsets(self):
        x = _rand(32, seed=11)
        records = write_record(x, torch.zeros((32, ROW_BYTES), dtype=torch.uint8))
        payload, scales = encode_rows(x)
        self.assertTrue(
            torch.equal(records[:, :PAYLOAD_BYTES], payload.reshape(32, -1).view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(
                records[:, SCALE_OFFSET : SCALE_OFFSET + SCALE_BYTES],
                scales.reshape(32, -1).view(torch.uint8),
            )
        )

    def test_padding_is_untouched(self):
        x = _rand(16, seed=12)
        records = write_record(x, torch.zeros((16, ROW_BYTES), dtype=torch.uint8))
        self.assertTrue(bool((records[:, PAYLOAD_BYTES + SCALE_BYTES :] == 0).all()))

    def test_write_record_reuses_persistent_staging_without_growth(self):
        x = _rand(8, seed=13)
        buffer = torch.zeros((64, ROW_BYTES), dtype=torch.uint8)
        before = buffer.data_ptr()
        out = write_record(x, buffer)
        self.assertEqual(buffer.data_ptr(), before)
        self.assertEqual(out.shape, (8, ROW_BYTES))
        # Only the first 8 rows are touched.
        self.assertTrue(bool((buffer[8:] == 0).all()))

    def test_records_are_self_delimiting_at_row_stride(self):
        x = _rand(5, seed=14)
        records = write_record(x, torch.zeros((5, ROW_BYTES), dtype=torch.uint8))
        arena = records.reshape(-1)
        for token in range(5):
            start = token * ROW_BYTES
            slot = arena[start : start + ROW_BYTES].unsqueeze(0)
            decoded = decode_records(
                slot, head_num=HEAD_NUM, head_dim=HEAD_DIM, dtype=torch.bfloat16
            )
            expected = decode_records(
                records[token : token + 1],
                head_num=HEAD_NUM,
                head_dim=HEAD_DIM,
                dtype=torch.bfloat16,
            )
            self.assertTrue(torch.equal(decoded, expected), f"slot {token}")


class TestDeterminism(CustomTestCase):
    def test_encode_is_deterministic(self):
        x = _rand(64, seed=15)
        a = write_record(x, torch.zeros((64, ROW_BYTES), dtype=torch.uint8))
        b = write_record(x, torch.zeros((64, ROW_BYTES), dtype=torch.uint8))
        self.assertTrue(torch.equal(a, b))

    def test_partial_batch_leaves_no_stale_bytes(self):
        """A second, smaller write must not leak the first write's tail."""
        big = _rand(32, seed=16)
        small = _rand(4, seed=17)
        buffer = torch.zeros((32, ROW_BYTES), dtype=torch.uint8)
        write_record(big, buffer)
        write_record(small, buffer)
        first_four = decode_records(
            buffer[:4], head_num=HEAD_NUM, head_dim=HEAD_DIM, dtype=torch.bfloat16
        )
        expected = decode_records(
            write_record(small, torch.zeros((4, ROW_BYTES), dtype=torch.uint8)),
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
        )
        self.assertTrue(torch.equal(first_four, expected))


if __name__ == "__main__":
    unittest.main()
