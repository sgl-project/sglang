"""Round trips of the routed-experts wire encodings."""

import unittest

import numpy as np

from sglang.srt.state_capturer.routed_experts_codec import (
    BASE64_INT32,
    ZSTD,
    decode_routed_experts,
    encode_routed_experts,
    encode_routed_experts_per_request,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _routing(num_tokens, num_layers, topk, num_experts, dtype, seed=0):
    """Sorted rows of distinct expert ids, as the capturer hands them over."""
    rng = np.random.default_rng(seed)
    rows = np.argsort(rng.random((num_tokens, num_layers, num_experts)), axis=-1)
    return np.sort(rows[..., :topk], axis=-1).astype(dtype)


class TestRoutedExpertsCodec(CustomTestCase):
    def _roundtrip(self, rows, encoding):
        payload = encode_routed_experts(rows, encoding)
        return decode_routed_experts(payload, encoding, rows.shape[0], rows.shape[1])

    def test_base64_int32_widens_captured_rows(self):
        for dtype in (np.uint8, np.int16, np.int32):
            rows = _routing(7, 3, 4, 64, dtype)
            out = self._roundtrip(rows, BASE64_INT32)
            self.assertEqual(out.dtype, np.int32)
            np.testing.assert_array_equal(out, rows)

    def test_zstd_keeps_the_captured_width(self):
        cases = (
            (128, np.uint8, np.uint8),
            (384, np.int16, np.uint16),
            (384, np.int32, np.int32),
        )
        for num_experts, dtype, wire_dtype in cases:
            rows = _routing(33, 5, 8, num_experts, dtype)
            out = self._roundtrip(rows, ZSTD)
            self.assertEqual(out.dtype, wire_dtype, f"{dtype=}")
            self.assertTrue(out.flags.c_contiguous)
            np.testing.assert_array_equal(out, rows)

    def test_zstd_multithreaded_frame(self):
        # large enough to cross the threaded-compression threshold
        rows = _routing(12000, 48, 8, 128, np.uint8)
        np.testing.assert_array_equal(self._roundtrip(rows, ZSTD), rows)

    def test_zstd_is_far_smaller_than_base64_int32(self):
        rows = _routing(2000, 16, 8, 128, np.uint8)
        payload = encode_routed_experts(rows, ZSTD)
        self.assertLess(
            len(payload), len(encode_routed_experts(rows, BASE64_INT32)) / 4
        )

    def test_empty(self):
        rows = np.empty((0, 4, 8), dtype=np.uint8)
        for encoding in (BASE64_INT32, ZSTD):
            self.assertEqual(self._roundtrip(rows, encoding).shape[:2], (0, 4))

    def test_per_request_keeps_none(self):
        out = encode_routed_experts_per_request(
            [_routing(3, 2, 2, 8, np.uint8), None], ZSTD
        )
        self.assertIsNone(out[1])
        self.assertIsNone(encode_routed_experts_per_request(None, ZSTD))

    def test_unknown_encoding(self):
        with self.assertRaises(ValueError):
            encode_routed_experts(_routing(2, 2, 2, 8, np.uint8), "lz4")


if __name__ == "__main__":
    unittest.main()
