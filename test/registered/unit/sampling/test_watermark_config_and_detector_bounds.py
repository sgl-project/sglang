"""Unit tests for watermark config probability and file size bounds."""

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
from sglang.test.test_utils import CustomTestCase


def _load_config(source: str):
    try:
        return load_watermark_config(source, algorithm="gumbel")
    except TypeError:
        return load_watermark_config(source)


class TestLoadWatermarkConfigBounds(CustomTestCase):
    def test_rejects_out_of_bounds_mixing_probability(self):
        for bad in (0.0, 1.0, -0.1, 1.1, True, False):
            with self.subTest(mixing_probability=bad):
                with self.assertRaises(WatermarkConfigError):
                    _load_config(json.dumps({"mixing_probability": bad}))

    def test_rejects_out_of_bounds_max_probability(self):
        for bad in (0.0, -0.5, 1.1, True, False):
            with self.subTest(max_probability=bad):
                with self.assertRaises(WatermarkConfigError):
                    _load_config(json.dumps({"max_probability": bad}))

    def test_accepts_valid_probability_bounds(self):
        cfg = _load_config(
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
                _load_config(tmp_path)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
