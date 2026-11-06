"""Client media problems surface as ValueError (HTTP 400), not server errors."""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

import pybase64
import requests
from PIL import Image

from sglang.srt.multimodal.cache.identity import snapshot_media
from sglang.srt.utils.common import (
    CLIENT_MEDIA_EXCEPTIONS,
    get_image_bytes,
    load_image,
    media_url_timeout,
    read_local_media_file,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _image_bytes(fmt: str, size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format=fmt)
    return buf.getvalue()


class TestMediaURLTimeout(unittest.TestCase):
    def test_unset_keeps_each_type_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_MEDIA_URL_TIMEOUT", None)
            os.environ.pop("REQUEST_TIMEOUT", None)
            self.assertEqual(media_url_timeout(3), 3.0)
            self.assertEqual(media_url_timeout(10), 10.0)

    def test_legacy_request_timeout_is_honored(self):
        with patch.dict(os.environ, {"REQUEST_TIMEOUT": "12"}):
            os.environ.pop("SGLANG_MEDIA_URL_TIMEOUT", None)
            self.assertEqual(media_url_timeout(3), 12.0)

    def test_new_variable_wins(self):
        with patch.dict(
            os.environ, {"REQUEST_TIMEOUT": "12", "SGLANG_MEDIA_URL_TIMEOUT": "30.5"}
        ):
            self.assertEqual(media_url_timeout(3), 30.5)

    def test_image_download_uses_it(self):
        with (
            patch.dict(os.environ, {"SGLANG_MEDIA_URL_TIMEOUT": "7"}),
            patch(
                "sglang.srt.utils.common.download_remote_media", return_value=b"x"
            ) as download,
        ):
            get_image_bytes("https://example.com/a.png")
        download.assert_called_once_with("https://example.com/a.png", timeout=7.0)


class TestLocalMediaFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "a.png")
        with open(self.path, "wb") as f:
            f.write(_image_bytes("PNG"))

    def test_reads_regular_file(self):
        self.assertEqual(get_image_bytes(self.path), _image_bytes("PNG"))
        self.assertEqual(get_image_bytes("file://" + self.path), _image_bytes("PNG"))

    def test_missing_file_is_client_error(self):
        with self.assertRaises(ValueError):
            read_local_media_file(os.path.join(self.dir.name, "missing.png"))

    def test_non_regular_file_is_rejected(self):
        with self.assertRaises(ValueError):
            read_local_media_file(self.dir.name)
        if os.path.exists("/dev/zero"):
            with self.assertRaises(ValueError):
                get_image_bytes("file:///dev/zero")

    def test_can_be_disabled(self):
        with patch.dict(os.environ, {"SGLANG_MEDIA_ALLOW_LOCAL_FILES": "0"}):
            with self.assertRaises(ValueError):
                get_image_bytes(self.path)
            with self.assertRaises(ValueError):
                snapshot_media(self.path)
            with self.assertRaisesRegex(ValueError, "not allowed"):
                snapshot_media("file://" + self.path)

    def test_raw_base64_starting_with_slash_is_not_a_path(self):
        jpeg = _image_bytes("JPEG")
        encoded = pybase64.b64encode(jpeg).decode()
        self.assertTrue(encoded.startswith("/9j/"))
        self.assertEqual(get_image_bytes(encoded), jpeg)


class TestImageDecodeErrors(unittest.TestCase):
    def test_decompression_bomb_is_client_error(self):
        self.assertIn(Image.DecompressionBombError, CLIENT_MEDIA_EXCEPTIONS)
        data = _image_bytes("PNG", size=(64, 64))
        with patch.object(Image, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(ValueError):
                load_image(data, gpu_image_decode=False)

    def test_common_formats_decode(self):
        for fmt in ("PNG", "JPEG", "GIF", "TIFF", "BMP", "WEBP"):
            with self.subTest(fmt=fmt):
                image, _ = load_image(_image_bytes(fmt), gpu_image_decode=False)
                self.assertEqual(image.size, (8, 8))

    def test_heic_decodes(self):
        try:
            import pillow_heif
        except ImportError:
            self.skipTest("pillow-heif is not installed")
        heif = pillow_heif.from_pillow(Image.new("RGB", (8, 8), (0, 200, 0)))
        buf = io.BytesIO()
        heif.save(buf, format="HEIF")
        image, _ = load_image(buf.getvalue(), gpu_image_decode=False)
        self.assertEqual(image.size, (8, 8))


class TestSnapshotMediaErrors(unittest.TestCase):
    def test_download_failures_are_client_errors(self):
        for error in (
            requests.exceptions.Timeout("slow"),
            requests.exceptions.ConnectionError("down"),
            requests.exceptions.HTTPError("404"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch("sglang.srt.utils.get_image_bytes", side_effect=error),
            ):
                with self.assertRaises(ValueError):
                    snapshot_media("https://example.com/" + "a" * 200)


if __name__ == "__main__":
    unittest.main()
