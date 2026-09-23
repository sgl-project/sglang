"""Bad-media classification for the Inkling multimodal processor.

Inkling resolves request media through its own ``_resolve_media_item`` and decodes
image bytes through ``_encode_image_bytes`` instead of
``BaseMultimodalProcessor._load_single_item``. Client-supplied media that cannot
be resolved or decoded must raise ``ValueError`` (the 400 path, classified by
``CLIENT_MEDIA_EXCEPTIONS``), not bubble up as raw connection / decode errors
that the server reports as HTTP 500. See issue #40897.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import base64
import binascii
import unittest
from unittest.mock import patch

import requests
from sglang.srt.multimodal.inkling.image_processing import _encode_image_bytes
from sglang.srt.multimodal.processors.inkling import _resolve_media_item
from sglang.test.test_utils import CustomTestCase


class TestResolveMediaItem(CustomTestCase):
    def test_empty_url_is_client_error(self):
        for item in ("", {"url": ""}, {"url": "   "}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                _resolve_media_item(item)

    def test_unreachable_http_url_is_client_error(self):
        # A connection failure while fetching the remote media is the client's
        # bad URL, not a server fault; the raw ConnectionError must be wrapped.
        cases = (
            requests.exceptions.ConnectionError("dns failure"),
            requests.exceptions.Timeout("read timed out"),
            requests.exceptions.TooManyRedirects("loop"),
        )
        for exc in cases:
            with self.subTest(exc_type=type(exc).__name__), patch(
                "sglang.srt.multimodal.processors.inkling.download_remote_media",
                side_effect=exc,
            ):
                with self.assertRaises(ValueError) as ctx:
                    _resolve_media_item(
                        {"url": "https://unreachable.example/x.png"}
                    )
                self.assertIs(ctx.exception.__cause__, exc)

    def test_invalid_data_base64_is_client_error(self):
        # binascii.Error is a ValueError subclass, so this was already on the
        # 400 path; lock it in so the resolution refactor does not regress it.
        with self.assertRaises(binascii.Error):
            _resolve_media_item("data:image/png;base64,!!!not-base64!!!")

    def test_valid_data_url_round_trip(self):
        payload = b"\x89PNG\r\n\x1a\n"
        url = "data:image/png;base64," + base64.b64encode(payload).decode()
        self.assertEqual(_resolve_media_item(url), payload)

    def test_remote_fetch_result_passes_through(self):
        with patch(
            "sglang.srt.multimodal.processors.inkling.download_remote_media",
            return_value=b"remote-bytes",
        ):
            self.assertEqual(
                _resolve_media_item("https://media.host/x.png"), b"remote-bytes"
            )

    def test_local_path_unchanged(self):
        # Plain file paths are left for the per-modality byte loader.
        self.assertEqual(_resolve_media_item("/local/img.png"), "/local/img.png")
        self.assertEqual(_resolve_media_item(b"already-bytes"), b"already-bytes")


class TestEncodeImageBytes(CustomTestCase):
    def test_undecodable_image_bytes_is_client_error(self):
        with self.assertRaises(ValueError) as ctx:
            _encode_image_bytes(
                b"definitely not an image",
                patch_size=14,
                rescale_image_frac=None,
                rescale_image_max_upscaled_long_edge=None,
            )
        self.assertIn("Could not decode image bytes", str(ctx.exception))

    def test_undecodable_image_bytes_cause_is_pil_error(self):
        from PIL import UnidentifiedImageError

        with self.assertRaises(ValueError) as ctx:
            _encode_image_bytes(
                b"definitely not an image",
                patch_size=14,
                rescale_image_frac=None,
                rescale_image_max_upscaled_long_edge=None,
            )
        self.assertIsInstance(ctx.exception.__cause__, UnidentifiedImageError)


if __name__ == "__main__":
    unittest.main()
