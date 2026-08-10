"""Unit tests for bad-input classification in ``BaseMultimodalProcessor._load_single_item``.

Media the client supplied but that cannot be fetched or decoded must raise
``ValueError``; anything else must stay a ``RuntimeError``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import binascii
import unittest
from unittest.mock import MagicMock, patch

import requests

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils.common import CLIENT_MEDIA_EXCEPTIONS
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
