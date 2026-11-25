"""Unit tests for bad-input classification in ``BaseMultimodalProcessor`` media loading.

Media the client supplied but that cannot be fetched or decoded must raise
``ValueError``; anything else must stay a ``RuntimeError``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import asyncio
import base64
import binascii
import errno
import io
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import requests
from PIL import Image

from sglang.srt.multimodal.media_io import CLIENT_MEDIA_EXCEPTIONS, ImageData
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
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
                        "sglang.srt.multimodal.media_io.get_mm_http_session",
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

    def test_decompression_bomb_image(self):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        with (
            patch.object(Image, "MAX_IMAGE_PIXELS", 16),
            self.assertRaisesRegex(ValueError, "decompression bomb") as ctx,
        ):
            _StubProcessor._load_single_item(buf.getvalue(), Modality.IMAGE)
        self.assertIsInstance(ctx.exception.__cause__, Image.DecompressionBombError)

    def test_undecodable_audio_bytes(self):
        # soundfile raises LibsndfileError, a RuntimeError -- not a ValueError.
        self._assert_client_error(b"definitely not audio", Modality.AUDIO)

    def test_undecodable_video_bytes(self):
        # Decoder is patched so no codec backend needs to be installed.
        with patch(
            "sglang.srt.multimodal.media_io.VideoDecoderWrapper",
            side_effect=RuntimeError("invalid data found when processing input"),
        ):
            self._assert_client_error(b"definitely not a video", Modality.VIDEO)


class TestServerFaultStaysServerError(CustomTestCase):
    """``load_video`` catches the decoder broadly; these are the exclusions."""

    def _assert_server_error(self, side_effect):
        with patch(
            "sglang.srt.multimodal.media_io.VideoDecoderWrapper",
            side_effect=side_effect,
        ):
            with self.assertRaises(RuntimeError):
                _StubProcessor._load_single_item(b"payload", Modality.VIDEO)

    def test_missing_decoder_backend(self):
        self._assert_server_error(ImportError("no decoder backend installed"))

    def test_decoder_oom(self):
        self._assert_server_error(MemoryError("out of memory"))

    def test_image_source_os_error(self):
        with patch(
            "sglang.srt.multimodal.media_io.get_image_bytes",
            side_effect=OSError(errno.EMFILE, "too many open files"),
        ):
            with self.assertRaisesRegex(RuntimeError, "(?i)too many open files"):
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
        self.assertIsNone(ctx.exception.__context__)

    def test_authenticated_url_is_redacted_from_client_error(self):
        url = (
            "https://media-user:media-password@example.com/private.jpg"
            "?token=MOCK_PRIVATE_TOKEN"
        )
        response = requests.Response()
        response.status_code = 404
        loader_error = requests.HTTPError("fixture detail", response=response)

        with patch(
            "sglang.srt.multimodal.multimodal_processor_mixin.load_image",
            side_effect=loader_error,
        ):
            with self.assertRaises(ValueError) as ctx:
                _StubProcessor._load_single_item(ImageData(url=url), Modality.IMAGE)

        formatted = "".join(traceback.format_exception(ctx.exception))
        self.assertNotIn(url, formatted)
        self.assertNotIn("media-password", formatted)
        self.assertNotIn("MOCK_PRIVATE_TOKEN", formatted)
        self.assertIn("<url scheme=https>", formatted)
        self.assertIn("HTTP 404", formatted)
        self.assertIsInstance(ctx.exception.__cause__, requests.HTTPError)
        self.assertIsNone(ctx.exception.__context__)

    def test_encoder_redacts_decoder_details(self):
        from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
        from sglang.srt.disaggregation.encoder.server import BadRequestError

        processor = object.__new__(EncoderPreprocessor)
        processor.use_image_processor_gpu = False
        with (
            patch(
                "sglang.srt.disaggregation.encoder.preprocessor.load_image",
                side_effect=ValueError("fixture-private-caption"),
            ),
            self.assertRaises(BadRequestError) as ctx,
        ):
            processor._load_single_item(b"fixture", Modality.IMAGE)
        formatted = "".join(traceback.format_exception(ctx.exception))
        self.assertNotIn("fixture-private-caption", formatted)
        self.assertIn("<bytes length=7>", formatted)

    def test_data_uri_is_redacted_from_client_error(self):
        encoded = base64.b64encode(b"MOCK_PRIVATE_MEDIA").decode()
        data_uri = f"data:image/jpeg;base64,{encoded}"
        loader_error = binascii.Error(f"Non-base64 digit found in {data_uri}")

        with patch(
            "sglang.srt.multimodal.multimodal_processor_mixin.load_image",
            side_effect=loader_error,
        ):
            with self.assertRaises(ValueError) as ctx:
                _StubProcessor._load_single_item(data_uri, Modality.IMAGE)

        formatted = "".join(traceback.format_exception(ctx.exception))
        self.assertNotIn(data_uri, formatted)
        self.assertNotIn(encoded, formatted)
        self.assertNotIn("data:image/jpeg;base64", formatted)
        self.assertIn(f"<data-uri length={len(encoded)}>", formatted)
        self.assertIn("Invalid base64 media", formatted)
        self.assertIsInstance(ctx.exception.__cause__, binascii.Error)
        self.assertIsNone(ctx.exception.__context__)


class TestClientMediaExceptions(CustomTestCase):
    def test_tuple_covers_the_documented_families(self):
        for exc_type in (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            binascii.Error,  # invalid base64
            Image.DecompressionBombError,
        ):
            with self.subTest(exc_type=exc_type.__name__):
                self.assertTrue(issubclass(exc_type, CLIENT_MEDIA_EXCEPTIONS))


class _LoaderStub(_StubProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class TestPlaceholderMismatchIsClientError(CustomTestCase):
    def test_more_placeholders_than_images(self):
        """A prompt with more image placeholders than images must be a 400, not a 500."""
        processor = object.__new__(_LoaderStub)
        tokens = MultimodalSpecialTokens(image_token="<img>").build(None)
        with ThreadPoolExecutor(max_workers=1) as processor.io_executor:
            with self.assertRaisesRegex(
                ValueError, "2 image placeholders but 1 image inputs"
            ):
                asyncio.run(
                    processor.legacy_load_mm_data(
                        prompt="a <img> b <img> c",
                        multimodal_tokens=tokens,
                        image_data=[Image.new("RGB", (8, 8))],
                    )
                )


if __name__ == "__main__":
    unittest.main()
