"""Unit tests for bad-input classification in ``BaseMultimodalProcessor._load_single_item``.

Media the client supplied but that cannot be fetched or decoded must raise
``ValueError``; anything else must stay a ``RuntimeError``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import base64
import binascii
import io
import traceback
import unittest
from unittest.mock import MagicMock, patch

import requests
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils.common import CLIENT_MEDIA_EXCEPTIONS, ImageData
from sglang.test.test_utils import CustomTestCase

MODALITIES = (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO)


class _StubProcessor(BaseMultimodalProcessor):
    # gpu_image_decode=False keeps the decode on PIL so the test needs no GPU. The
    # abstract methods are never called: only the _load_single_item classmethod is.
    gpu_image_decode = False


def _session_raising(exc):
    session = MagicMock()
    session.get.side_effect = exc
    return session


class TestBadInputIsClientError(CustomTestCase):
    def _assert_client_error(self, data, modality):
        with self.assertRaises(ValueError):
            _StubProcessor._load_single_item(data, modality)

    def test_unfetchable_url_every_modality(self):
        # All three loaders fetch through get_mm_http_session(); HTTPError,
        # ConnectionError and Timeout all subclass RequestException.
        for exc in (
            requests.exceptions.HTTPError("404 from media host"),
            requests.exceptions.ConnectionError("dns failure"),
            requests.exceptions.Timeout("read timed out"),
        ):
            for modality in MODALITIES:
                with self.subTest(exc=type(exc).__name__, modality=modality):
                    with patch(
                        "sglang.srt.utils.common.get_mm_http_session",
                        return_value=_session_raising(exc),
                    ):
                        self._assert_client_error("https://media.host/clip", modality)

    def test_invalid_base64(self):
        self._assert_client_error("!!!not-base64!!!", Modality.IMAGE)

    def test_undecodable_image_bytes(self):
        # PIL raises UnidentifiedImageError, an OSError -- not a ValueError.
        self._assert_client_error(b"definitely not an image", Modality.IMAGE)

    def test_lazy_pil_decode_failure(self):
        malformed_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJSwAAAABJRU5ErkJggg=="
        )
        for media in (malformed_png, Image.open(io.BytesIO(malformed_png))):
            with self.subTest(media_type=type(media).__name__):
                with self.assertRaisesRegex(
                    ValueError, "Could not decode image"
                ) as ctx:
                    _StubProcessor._load_single_item(media, Modality.IMAGE)
                self.assertIsInstance(ctx.exception.__cause__.__cause__, OSError)

    def test_undecodable_audio_bytes(self):
        # soundfile raises LibsndfileError, a RuntimeError -- not a ValueError.
        self._assert_client_error(b"definitely not audio", Modality.AUDIO)

    def test_undecodable_video_bytes(self):
        # Decoder is patched so no codec backend needs to be installed.
        with patch(
            "sglang.srt.utils.common.VideoDecoderWrapper",
            side_effect=RuntimeError("invalid data found when processing input"),
        ):
            self._assert_client_error(b"definitely not a video", Modality.VIDEO)


class TestServerFaultStaysServerError(CustomTestCase):
    """``load_video`` catches the decoder broadly; these are the exclusions."""

    def _assert_server_error(self, side_effect):
        with patch(
            "sglang.srt.utils.common.VideoDecoderWrapper", side_effect=side_effect
        ):
            with self.assertRaises(RuntimeError):
                _StubProcessor._load_single_item(b"payload", Modality.VIDEO)

    def test_missing_decoder_backend(self):
        self._assert_server_error(ImportError("no decoder backend installed"))

    def test_decoder_oom(self):
        self._assert_server_error(MemoryError("out of memory"))

    def test_image_source_os_error(self):
        with patch(
            "sglang.srt.utils.common.get_image_bytes",
            side_effect=OSError("too many open files"),
        ):
            with self.assertRaisesRegex(RuntimeError, "too many open files"):
                _StubProcessor._load_single_item("file:///image.png", Modality.IMAGE)


class TestBadInputDoesNotLeakMedia(CustomTestCase):
    """Loader failures must not expose request media in formatted exceptions."""

    def test_malformed_url_preserves_sanitized_client_error(self):
        url = "https://[invalid"

        with self.assertRaises(ValueError) as ctx:
            _StubProcessor._load_single_item(url, Modality.IMAGE)

        formatted = "".join(traceback.format_exception(ctx.exception))
        self.assertNotIn(url, formatted)
        self.assertIn("Error while loading image data <url scheme=https>", formatted)
        self.assertIsInstance(ctx.exception.__cause__, ValueError)
        self.assertIn("<url scheme=https>", str(ctx.exception.__cause__))
        self.assertIsNone(ctx.exception.__context__)

    def test_authenticated_url_is_redacted_from_client_error(self):
        url = (
            "https://media-user:media-password@example.com/private.jpg"
            "?token=MOCK_PRIVATE_TOKEN"
        )
        loader_error = requests.HTTPError(f"404 Client Error for url: {url}")

        with patch(
            "sglang.srt.multimodal.processors.base_processor.load_image",
            side_effect=loader_error,
        ):
            with self.assertRaises(ValueError) as ctx:
                _StubProcessor._load_single_item(ImageData(url=url), Modality.IMAGE)

        formatted = "".join(traceback.format_exception(ctx.exception))
        self.assertNotIn(url, formatted)
        self.assertNotIn("media-password", formatted)
        self.assertNotIn("MOCK_PRIVATE_TOKEN", formatted)
        self.assertIn("<url scheme=https>", formatted)
        self.assertIn("404 Client Error", formatted)
        self.assertIsInstance(ctx.exception.__cause__, requests.HTTPError)
        self.assertIsNone(ctx.exception.__context__)

    def test_data_uri_is_redacted_from_client_error(self):
        encoded = base64.b64encode(b"MOCK_PRIVATE_MEDIA").decode()
        data_uri = f"data:image/jpeg;base64,{encoded}"
        loader_error = binascii.Error(f"Non-base64 digit found in {data_uri}")

        with patch(
            "sglang.srt.multimodal.processors.base_processor.load_image",
            side_effect=loader_error,
        ):
            with self.assertRaises(ValueError) as ctx:
                _StubProcessor._load_single_item(data_uri, Modality.IMAGE)

        formatted = "".join(traceback.format_exception(ctx.exception))
        self.assertNotIn(data_uri, formatted)
        self.assertNotIn(encoded, formatted)
        self.assertNotIn("data:image/jpeg;base64", formatted)
        self.assertIn(
            f"<data-uri mime=image/jpeg encoded_length={len(encoded)}>", formatted
        )
        self.assertIn("Non-base64 digit found", formatted)
        self.assertIsInstance(ctx.exception.__cause__, binascii.Error)
        self.assertIsNone(ctx.exception.__context__)


class TestClientMediaExceptions(CustomTestCase):
    def test_tuple_covers_the_documented_families(self):
        for exc_type in (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            binascii.Error,  # invalid base64
        ):
            with self.subTest(exc_type=exc_type.__name__):
                self.assertTrue(issubclass(exc_type, CLIENT_MEDIA_EXCEPTIONS))


if __name__ == "__main__":
    unittest.main()
