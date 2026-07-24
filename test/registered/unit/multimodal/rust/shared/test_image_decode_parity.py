"""Model-independent image decode parity for native Rust MM."""

import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import load_core  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CORE = load_core()
DECODE = CORE and CORE.common.image_decode_rgb


def encode(image, fmt, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@unittest.skipUnless(DECODE, "sglang-mm decode binding not built")
class TestRustImageDecode(CustomTestCase):
    def assert_matches_pil(self, data, tolerance=0):
        height, width, pixels = DECODE(data)
        expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        self.assertEqual((height, width), expected.shape[:2])
        actual = np.asarray(pixels).reshape(expected.shape)
        diff = np.abs(actual.astype(int) - expected.astype(int))
        self.assertLessEqual(diff.max(), tolerance)

    def test_png_modes_match_pil(self):
        rgb = np.random.default_rng(1).integers(0, 256, (19, 23, 3), dtype=np.uint8)
        cases = [
            ("RGB", Image.fromarray(rgb)),
            ("L", Image.fromarray(rgb[..., 0])),
            ("RGBA", Image.fromarray(np.dstack((rgb, rgb[..., 0])))),
            ("P", Image.fromarray(rgb).quantize(colors=16)),
        ]
        for mode, image in cases:
            with self.subTest(mode=mode):
                self.assert_matches_pil(encode(image, "PNG"))

    def test_jpeg_modes_match_with_decoder_tolerance(self):
        rgb = np.random.default_rng(2).integers(0, 256, (31, 29, 3), dtype=np.uint8)
        image = Image.fromarray(rgb)
        exif = image.getexif()
        exif[274] = 6  # EXIF orientation: neither decoder applies it
        cases = [
            ("RGB", encode(image, "JPEG")),
            ("L", encode(Image.fromarray(rgb[..., 0]), "JPEG")),
            ("CMYK", encode(image.convert("CMYK"), "JPEG")),
            ("EXIF-rotated", encode(image, "JPEG", exif=exif)),
        ]
        for mode, data in cases:
            with self.subTest(mode=mode):
                self.assert_matches_pil(data, tolerance=3)

    def test_unsupported_inputs_fail(self):
        rgb = Image.fromarray(
            np.random.default_rng(3).integers(0, 256, (9, 9, 3), dtype=np.uint8)
        )
        gray16 = Image.fromarray(np.zeros((9, 9), dtype=np.uint16))
        cases = {
            "corrupt": b"not an image",
            # PIL-only formats and bit depths must error (the request driver
            # then falls back to the Python path) — never silently diverge.
            "gif": encode(rgb, "GIF"),
            "webp": encode(rgb, "WEBP"),
            "png16": encode(gray16, "PNG"),
        }
        for name, data in cases.items():
            with self.subTest(input=name):
                with self.assertRaises(ValueError):
                    DECODE(data)


if __name__ == "__main__":
    unittest.main()
