"""Round trips of the routed-experts wire encodings."""

import unittest

import numpy as np
import torch

from sglang.srt.state_capturer.routed_experts_codec import (
    BASE64_INT32,
    ZSTD,
    decode_routed_experts,
    encode_routed_experts,
    encode_routed_experts_per_request,
    host_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _routing(num_tokens, num_layers, topk, num_experts, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.argsort(rng.random((num_tokens, num_layers, num_experts)), axis=-1)
    return torch.from_numpy(rows[..., :topk].astype(np.int32))


class TestRoutedExpertsCodec(CustomTestCase):
    def _roundtrip(self, topk, encoding):
        payload = encode_routed_experts(topk, encoding)
        return decode_routed_experts(payload, encoding, topk.shape[0], topk.shape[1])

    def test_base64_int32_is_raw_buffer(self):
        topk = _routing(7, 3, 4, 64)
        out = self._roundtrip(topk, BASE64_INT32)
        self.assertEqual(out.dtype, np.int32)
        np.testing.assert_array_equal(out, topk.numpy())

    def test_zstd_narrows_by_value_range(self):
        for num_experts, dtype in ((128, np.uint8), (256, np.uint8), (384, np.uint16)):
            topk = _routing(33, 5, 8, num_experts)
            out = self._roundtrip(topk, ZSTD)
            self.assertEqual(out.dtype, dtype, f"{num_experts=}")
            self.assertTrue(out.flags.c_contiguous)
            np.testing.assert_array_equal(out, topk.numpy())

    def test_zstd_keeps_negative_values(self):
        topk = _routing(5, 2, 4, 32)
        topk[0, 0, 0] = -1
        out = self._roundtrip(topk, ZSTD)
        self.assertEqual(out.dtype, np.int32)
        np.testing.assert_array_equal(out, topk.numpy())

    def test_zstd_multithreaded_frame(self):
        topk = _routing(12000, 48, 8, 128)  # > the threaded-compression threshold
        np.testing.assert_array_equal(self._roundtrip(topk, ZSTD), topk.numpy())

    def test_zstd_compresses_sorted_rows(self):
        topk = torch.sort(_routing(2000, 16, 8, 128), dim=-1).values
        payload = encode_routed_experts(topk, ZSTD)
        baseline = encode_routed_experts(topk, BASE64_INT32)
        self.assertLess(len(payload), len(baseline) / 4)
        np.testing.assert_array_equal(
            decode_routed_experts(payload, ZSTD, 2000, 16), topk.numpy()
        )

    def test_empty(self):
        topk = torch.empty((0, 4, 8), dtype=torch.int32)
        for encoding in (BASE64_INT32, ZSTD):
            out = self._roundtrip(topk, encoding)
            self.assertEqual(out.shape[:2], (0, 4))

    def test_per_request_keeps_none(self):
        topk = _routing(3, 2, 2, 8)
        out = encode_routed_experts_per_request([topk, None], ZSTD)
        self.assertIsNone(out[1])
        self.assertIsNone(encode_routed_experts_per_request(None, ZSTD))

    def test_narrowed_numpy_rows(self):
        topk = _routing(40, 4, 8, 128)
        for rows in (topk.numpy().astype(np.uint8), topk.numpy().astype(np.int16)):
            for encoding in (BASE64_INT32, ZSTD):
                out = decode_routed_experts(
                    encode_routed_experts(rows, encoding), encoding, 40, 4
                )
                np.testing.assert_array_equal(out, topk.numpy())
                if encoding == BASE64_INT32:
                    self.assertEqual(out.dtype, np.int32)

    def test_host_dtype(self):
        self.assertEqual(host_dtype(BASE64_INT32, 128), torch.int32)
        self.assertEqual(host_dtype(ZSTD, None), torch.int32)
        self.assertEqual(host_dtype(ZSTD, 256), torch.uint8)
        self.assertEqual(host_dtype(ZSTD, 288), torch.int16)
        self.assertEqual(host_dtype(ZSTD, 1 << 16), torch.int32)

    def test_unknown_encoding(self):
        with self.assertRaises(ValueError):
            encode_routed_experts(_routing(2, 2, 2, 8), "lz4")


if __name__ == "__main__":
    unittest.main()
