"""Unit tests for media URL scheme resolvers registered with ``srt.utils.common``.

Deployments can register a fetcher for URL schemes the image loaders don't know
(``s3://``, ``gs://``, ...). Each loader entrypoint must consult the registry
before the bare-base64 fallback, and unregistered schemes must keep falling
back to base64.

No server, no model loading -- pure CPU.
"""

import io
import unittest

from PIL import Image

from sglang.srt.multimodal.cache.identity import _read_media_bytes
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class TestMediaUrlResolvers(CustomTestCase):
    def setUp(self):
        self.png = _png_bytes()
        self.seen_urls: list[str] = []
        self.addCleanup(common._media_url_resolvers.clear)
        common._media_url_resolvers.clear()

        def resolve(url: str) -> bytes:
            self.seen_urls.append(url)
            return self.png

        common.register_media_url_resolver("fake", resolve)

    def test_get_image_bytes_routes_registered_scheme(self):
        self.assertEqual(common.get_image_bytes("fake://bucket/img"), self.png)
        self.assertEqual(self.seen_urls, ["fake://bucket/img"])

    def test_load_image_decodes_registered_scheme(self):
        image, _ = common.load_image("fake://bucket/img.png", gpu_image_decode=False)
        self.assertEqual(image.size, (4, 4))

    def test_cache_identity_reads_registered_scheme(self):
        self.assertEqual(_read_media_bytes("fake://bucket/img.png"), self.png)

    def test_unregistered_scheme_still_fails_as_base64(self):
        with self.assertRaises(ValueError):
            common.get_image_bytes("other://bucket/img")
        self.assertEqual(self.seen_urls, [])


if __name__ == "__main__":
    unittest.main()
