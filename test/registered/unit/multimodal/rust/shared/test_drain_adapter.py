"""Unit tests for native MM scheduler-input tensor wrapping."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.rust_server import MmProcessorHost  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNativeMmDrainAdapter(CustomTestCase):
    def setUp(self):
        self.host = MmProcessorHost.__new__(MmProcessorHost)
        self.host._native = {
            "feature_dim": 6,
            "image_token_id": 10,
            "vision_start_token_id": 11,
            "vision_end_token_id": 12,
            "video_token_id": 13,
        }

    def build(self):
        features = np.arange(30, dtype=np.float32)
        output = self.host.build_native_mm(
            (
                features,
                [(1, 2, 2), (1, 1, 1)],
                [101, 202],
                [(2, 5), (8, 8)],
                np.arange(30, dtype=np.int64),
                -3,
            )
        )
        return output, features

    def test_wraps_and_slices_native_buffers(self):
        output, features = self.build()
        self.assertEqual(
            [tuple(item.feature.shape) for item in output.mm_items], [(4, 6), (1, 6)]
        )
        self.assertEqual([item.hash for item in output.mm_items], [101, 202])
        self.assertEqual(
            [item.offsets for item in output.mm_items], [[(2, 5)], [(8, 8)]]
        )
        self.assertEqual(tuple(output.mrope_positions.shape), (3, 10))
        self.assertEqual(output.mrope_position_delta.item(), -3)
        self.assertEqual(
            (output.im_start_id, output.im_token_id, output.im_end_id), (11, 10, 12)
        )
        features[0] = 99
        self.assertEqual(output.mm_items[0].feature[0, 0].item(), 99)

    def test_optional_pad_values_use_precomputed_hashes(self):
        with patch.dict(os.environ, {"SGLANG_MM_PRECOMPUTE_HASH": "1"}):
            output, _ = self.build()
            self.assertTrue(all(item.pad_value is not None for item in output.mm_items))


if __name__ == "__main__":
    unittest.main()
