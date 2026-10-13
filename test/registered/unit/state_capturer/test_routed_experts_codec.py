"""The routed-experts wire format round-trips exactly and rejects anything malformed."""

import struct
import unittest

import numpy as np
import pybase64

from sglang.srt.state_capturer.routed_experts_codec import (
    RoutedExpertsEncoder,
    decode_routed_experts,
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
    encoder = RoutedExpertsEncoder()

    def _roundtrip(self, rows):
        return decode_routed_experts(self.encoder.encode(rows))

    def test_round_trip_keeps_the_captured_dtype(self):
        for num_experts, dtype in ((128, np.uint8), (384, np.int16), (384, np.int32)):
            rows = _routing(33, 5, 8, num_experts, dtype)
            out = self._roundtrip(rows)
            self.assertEqual(out.dtype, rows.dtype)
            self.assertTrue(out.flags.c_contiguous and out.flags.writeable)
            np.testing.assert_array_equal(out, rows)

    def test_threaded_frame_and_compressor_reuse(self):
        # large enough to cross the threaded-compression threshold
        big = _routing(12000, 48, 8, 128, np.uint8)
        small = _routing(10, 48, 8, 128, np.uint8, seed=1)
        for rows in (big, small, big):
            np.testing.assert_array_equal(self._roundtrip(rows), rows)

    def test_a_quarter_of_base64_int32_or_less(self):
        rows = _routing(2000, 16, 8, 128, np.uint8)
        base64_int32_chars = rows.size * 4 * 4 / 3
        self.assertLess(len(self.encoder.encode(rows)), base64_int32_chars / 4)

    def test_empty(self):
        rows = np.empty((0, 4, 8), dtype=np.uint8)
        self.assertEqual(self._roundtrip(rows).shape, (0, 4, 8))

    def test_per_request_keeps_none(self):
        rows = _routing(3, 2, 2, 8, np.uint8)
        out = self.encoder.encode_per_request([rows, None])
        self.assertIsNone(out[1])
        self.assertIsNone(self.encoder.encode_per_request(None))

    def test_rejects_non_host_dtype(self):
        with self.assertRaises(ValueError):
            self.encoder.encode(_routing(2, 2, 2, 8, np.int64))

    def test_malformed_payloads_raise_value_error(self):
        rows = _routing(20, 4, 8, 128, np.uint8)
        data = pybase64.b64decode(self.encoder.encode(rows))
        header = struct.Struct("<4sB3sIHH")
        _, version, dtype_str, tokens, layers, topk = header.unpack_from(data)
        frame = data[header.size :]
        cases = {
            "pre-v1 base64 int32": rows.astype(np.int32).tobytes(),
            "bad magic": b"XXXX" + data[4:],
            "truncated frame": data[:-5],
            "header promises more": header.pack(
                b"SGRE", version, dtype_str, tokens + 1, layers, topk
            )
            + frame,
            "unsupported dtype": header.pack(
                b"SGRE", version, b"<f8", tokens, layers, topk
            )
            + frame,
            "short header": data[:10],
        }
        for name, raw in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                decode_routed_experts(pybase64.b64encode(raw).decode())
        with self.assertRaises(ValueError):
            decode_routed_experts("!!!!")


if __name__ == "__main__":
    unittest.main()
