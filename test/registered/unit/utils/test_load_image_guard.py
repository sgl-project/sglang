import unittest
from io import BytesIO

from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.utils.common import load_image
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def _png_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestLoadImageDecodeGuard(CustomTestCase):
    """`load_image` must reject decompression-bomb-sized images before the
    expensive full decode, using SGLANG_IMAGE_MAX_DECODE_PIXELS as the limit.
    """

    def test_image_within_limit_loads(self):
        image_bytes = _png_bytes(64, 64)
        with envs.SGLANG_IMAGE_MAX_DECODE_PIXELS.override(1_000_000):
            image, _ = load_image(image_bytes)
        self.assertEqual((image.width, image.height), (64, 64))

    def test_image_over_limit_raises(self):
        image_bytes = _png_bytes(64, 64)  # 4096 pixels
        with envs.SGLANG_IMAGE_MAX_DECODE_PIXELS.override(100):
            with self.assertRaises(ValueError):
                load_image(image_bytes)

    def test_limit_disabled_with_zero(self):
        image_bytes = _png_bytes(64, 64)
        with envs.SGLANG_IMAGE_MAX_DECODE_PIXELS.override(0):
            image, _ = load_image(image_bytes)
        self.assertEqual((image.width, image.height), (64, 64))


if __name__ == "__main__":
    unittest.main()
