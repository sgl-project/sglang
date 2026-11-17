"""Unit tests for watermark config probability/file bounds and detector edge cases."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest

from sglang.srt.sampling.watermarking.config import (
    WatermarkConfigError,
    load_watermark_config,
)
from sglang.srt.sampling.watermarking.detector import (
    WatermarkDetector,
    WatermarkStatistics,
    detect,
)
from sglang.test.test_utils import CustomTestCase


class TestLoadWatermarkConfigBounds(CustomTestCase):
    def test_rejects_out_of_bounds_mixing_probability(self):
        for bad in (0.0, 1.0, -0.1, 1.1, True, False):
            with self.subTest(mixing_probability=bad):
                with self.assertRaises(WatermarkConfigError):
                    load_watermark_config(json.dumps({"mixing_probability": bad}))

    def test_rejects_out_of_bounds_max_probability(self):
        for bad in (0.0, -0.5, 1.1, True, False):
            with self.subTest(max_probability=bad):
                with self.assertRaises(WatermarkConfigError):
                    load_watermark_config(json.dumps({"max_probability": bad}))

    def test_accepts_valid_probability_bounds(self):
        cfg = load_watermark_config(
            json.dumps({"mixing_probability": 0.25, "max_probability": 1.0})
        )
        self.assertEqual(cfg.mixing_probability, 0.25)
        self.assertEqual(cfg.max_probability, 1.0)

    def test_rejects_oversized_file_before_utf8_decode(self):
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".json", delete=False
        ) as tmp:
            # Write > 4096 bytes with a split multibyte UTF-8 sequence around byte 4096
            tmp.write(b"{" + b" " * 4094 + "é".encode("utf-8") + b"}")
            tmp_path = tmp.name
        try:
            with self.assertRaisesRegex(WatermarkConfigError, "exceeds 4096 bytes"):
                load_watermark_config(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestWatermarkDetectorEdgeCases(CustomTestCase):
    KEY = "0x0123456789ABCDEF"

    def test_detect_forwards_max_contexts(self):
        result = detect(
            [1, 2, 3, 4, 5],
            key=self.KEY,
            context_window=1,
            max_contexts=2,
        )
        self.assertTrue(result.prefix_limit_reached)
        self.assertEqual(result.key_a_all_positions.num_contexts, 2)

    def test_rejects_bytes_and_bytearray_token_ids(self):
        detector = WatermarkDetector(self.KEY, context_window=2)
        for bad in (b"hello", bytearray(b"hello")):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(
                    ValueError, "token IDs must be unsigned 32-bit integers"
                ):
                    detector.detect_tokens(bad)
                with self.assertRaisesRegex(
                    ValueError, "token IDs must be unsigned 32-bit integers"
                ):
                    detector.detect_tokens([1, 2, 3], prompt_token_ids=bad)

    def test_statistics_from_non_positive_scores(self):
        stats = WatermarkStatistics.from_scores([-0.5, 0.0])
        self.assertEqual(stats.num_contexts, 2)
        self.assertEqual(stats.p_value, 1.0)
        self.assertEqual(stats.log_p_value, 0.0)


if __name__ == "__main__":
    unittest.main()
