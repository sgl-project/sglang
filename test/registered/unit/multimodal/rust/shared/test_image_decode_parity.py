"""Model-independent image decode parity for native Rust MM."""

import io
import unittest

import numpy as np
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

try:
    from sglang.srt.multimodal import _core

    DECODE = _core.common.image_decode_rgb
except (AttributeError, ImportError):
    DECODE = None


def encode(array, mode, fmt):
    buffer = io.BytesIO()
    Image.fromarray(array, mode).save(buffer, format=fmt)
    return buffer.getvalue()


@unittest.skipUnless(DECODE, "sglang-mm decode binding not built")
class TestRustImageDecode(CustomTestCase):
    def test_png_modes_match_pil(self):
        rgb = np.random.default_rng(1).integers(0, 256, (19, 23, 3), dtype=np.uint8)
        cases = [
            ("RGB", rgb),
            ("L", rgb[..., 0]),
            ("RGBA", np.dstack((rgb, np.full(rgb.shape[:2], 127, np.uint8)))),
        ]
        for mode, array in cases:
            with self.subTest(mode=mode):
                data = encode(array, mode, "PNG")
                height, width, pixels = DECODE(data)
                expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
                self.assertEqual((height, width), expected.shape[:2])
                np.testing.assert_array_equal(
                    np.asarray(pixels).reshape(expected.shape), expected
                )

    def test_jpeg_matches_with_decoder_tolerance(self):
        image = np.random.default_rng(2).integers(0, 256, (31, 29, 3), dtype=np.uint8)
        data = encode(image, "RGB", "JPEG")
        height, width, pixels = DECODE(data)
        expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        actual = np.asarray(pixels).reshape(height, width, 3)
        self.assertLessEqual(np.abs(actual.astype(int) - expected.astype(int)).max(), 3)

    def test_corrupt_image_fails(self):
        with self.assertRaises(ValueError):
            DECODE(b"not an image")


if __name__ == "__main__":
    unittest.main()
