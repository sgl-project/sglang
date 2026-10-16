"""Unit tests for media URL scheme resolvers registered with ``srt.utils.common``.

Deployments can register a fetcher for URL schemes the media loaders don't know
(``s3://``, ``gs://``, ...). Each loader entrypoint must consult the registry
before the bare-base64 fallback, and unregistered schemes must keep falling
back to base64.

No server, no model loading -- pure CPU.
"""

import io
import struct
import unittest
import wave

from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.cache.identity import _read_media_bytes  # noqa: E402
from sglang.srt.multimodal.processors.inkling import _resolve_media_item  # noqa: E402
from sglang.srt.utils import common  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_AUDIO_SAMPLE_RATE = 16000
_AUDIO_SAMPLES = 1600


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_AUDIO_SAMPLE_RATE)
        wav.writeframes(struct.pack(f"<{_AUDIO_SAMPLES}h", *([0] * _AUDIO_SAMPLES)))
    return buf.getvalue()


class TestMediaUrlResolvers(CustomTestCase):
    def setUp(self):
        self.payload = _png_bytes()
        self.seen_urls: list[str] = []
        self.addCleanup(common._media_url_resolvers.clear)
        common._media_url_resolvers.clear()

        def resolve(url: str) -> bytes:
            self.seen_urls.append(url)
            return self.payload

        common.register_media_url_resolver("fake", resolve)

    def test_get_image_bytes_routes_registered_scheme(self):
        self.assertEqual(common.get_image_bytes("fake://bucket/img"), self.payload)
        self.assertEqual(self.seen_urls, ["fake://bucket/img"])

    def test_scheme_lookup_ignores_case(self):
        common.register_media_url_resolver("Loud", common._media_url_resolvers["fake"])
        self.assertEqual(common.get_image_bytes("FAKE://bucket/img"), self.payload)
        self.assertEqual(common.get_image_bytes("loud://bucket/img"), self.payload)
        self.assertEqual(self.seen_urls, ["FAKE://bucket/img", "loud://bucket/img"])

    def test_load_image_decodes_registered_scheme(self):
        image, _ = common.load_image("fake://bucket/img.png", gpu_image_decode=False)
        self.assertEqual(image.size, (4, 4))

    def test_cache_identity_reads_registered_scheme(self):
        self.assertEqual(_read_media_bytes("fake://bucket/img.png"), self.payload)

    def test_get_video_bytes_routes_registered_scheme(self):
        self.assertEqual(common.get_video_bytes("fake://bucket/clip.mp4"), self.payload)
        self.assertEqual(self.seen_urls, ["fake://bucket/clip.mp4"])

    def test_load_audio_decodes_registered_scheme(self):
        self.payload = _wav_bytes()
        audio = common.load_audio("fake://bucket/clip.wav", sr=_AUDIO_SAMPLE_RATE)
        self.assertEqual(len(audio), _AUDIO_SAMPLES)
        self.assertEqual(self.seen_urls, ["fake://bucket/clip.wav"])

    def test_inkling_media_item_routes_registered_scheme(self):
        self.assertEqual(_resolve_media_item("fake://bucket/img.png"), self.payload)
        self.assertEqual(
            _resolve_media_item({"url": "fake://bucket/img.png"}), self.payload
        )

    def test_unregistered_scheme_still_fails_as_base64(self):
        with self.assertRaises(ValueError):
            common.get_image_bytes("other://bucket/img")
        with self.assertRaises(ValueError):
            common.get_video_bytes("other://bucket/clip.mp4")
        self.assertEqual(self.seen_urls, [])


if __name__ == "__main__":
    unittest.main()
