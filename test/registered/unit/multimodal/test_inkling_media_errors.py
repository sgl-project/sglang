"""Unit tests for client-error classification in Inkling's multimodal fetch/decode path."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import errno
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import requests
import soundfile as sf
from PIL import Image

from sglang.srt.multimodal.inkling.feature_extraction import (
    _decode_audio,
    _load_audio_bytes,
)
from sglang.srt.multimodal.inkling.image_processing import (
    _encode_image_bytes,
    _load_image_bytes,
)
from sglang.srt.multimodal.inkling.image_processing_rust import _pil_decode
from sglang.srt.multimodal.processors.inkling import _resolve_media_item
from sglang.test.test_utils import CustomTestCase


def _session_raising(exc):
    session = MagicMock()
    session.get.side_effect = exc
    return session


def _png_bytes(width: int = 16, height: int = 16) -> bytes:
    arr = (np.random.RandomState(0).rand(height, width, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _broken_chunk_png_bytes() -> bytes:
    """A structurally-valid PNG with a corrupted IDAT chunk length byte.

    PIL raises SyntaxError (not OSError) for this, matching the fixture in
    test_common.py::TestLoadImage.
    """
    broken = bytearray(_png_bytes())
    idat_pos = broken.index(b"IDAT")
    broken[idat_pos - 1] -= 6
    return bytes(broken)


class TestResolveMediaItemIsClientError(CustomTestCase):
    """_resolve_media_item() -- empty/unreachable media URL (image or audio)."""

    def test_empty_url_string_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "empty media URL"):
            _resolve_media_item("")

    def test_empty_url_in_mapping_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "empty media URL"):
            _resolve_media_item({"url": ""})

    def test_empty_url_attribute_raises_value_error(self):
        class _ImageData:
            url = ""

        with self.assertRaisesRegex(ValueError, "empty media URL"):
            _resolve_media_item(_ImageData())

    def test_whitespace_only_url_raises_value_error(self):
        # A client sending "   " instead of "" must be rejected the same way,
        # not silently fall through to being treated as a plain file path.
        for whitespace_url in ("   ", "\t", "\n", " \t\n "):
            with self.subTest(url=repr(whitespace_url)):
                with self.assertRaisesRegex(ValueError, "empty media URL"):
                    _resolve_media_item({"url": whitespace_url})

    def test_unfetchable_url_every_request_exception(self):
        # download_remote_media() fetches through get_mm_http_session(); HTTPError,
        # ConnectionError and Timeout all subclass RequestException.
        for exc in (
            requests.exceptions.HTTPError("404 from media host"),
            requests.exceptions.ConnectionError("dns failure"),
            requests.exceptions.Timeout("read timed out"),
        ):
            with self.subTest(exc=type(exc).__name__):
                with patch(
                    "sglang.srt.utils.common.get_mm_http_session",
                    return_value=_session_raising(exc),
                ):
                    with self.assertRaises(ValueError) as ctx:
                        _resolve_media_item({"url": "https://media.host/clip.png"})
                    # the raw RequestException must not leak past the boundary
                    self.assertNotIsInstance(
                        ctx.exception, requests.exceptions.RequestException
                    )
                    self.assertIsInstance(ctx.exception.__cause__, type(exc))

    def test_long_url_error_message_is_truncated(self):
        # A "data:" URL's base64 payload (or a pathologically long http(s) URL)
        # can be megabytes; the error message (logged, and echoed in the 400
        # response) must stay bounded. Route through http(s) + a mocked fetch
        # failure so this doesn't depend on the payload happening to be valid
        # base64 (a long run of "A"s decodes cleanly and wouldn't raise).
        huge_url = "https://media.host/" + ("a" * 1_000_000)
        with patch(
            "sglang.srt.utils.common.get_mm_http_session",
            return_value=_session_raising(requests.exceptions.ConnectionError("x")),
        ):
            with self.assertRaises(ValueError) as ctx:
                _resolve_media_item({"url": huge_url})
        self.assertLess(len(str(ctx.exception)), 200)

    def test_non_url_item_passes_through_unchanged(self):
        # Non-media, non-string, non-Mapping items (already-resolved bytes, PIL
        # images, etc.) must not be touched by the empty-url / fetch guards.
        sentinel = b"already-resolved-bytes"
        self.assertIs(_resolve_media_item(sentinel), sentinel)

    def test_plain_path_still_passes_through(self):
        # A local path / file:// URI is intentionally not resolved here -- it's
        # handled by the per-modality byte loader downstream -- so it must not
        # be mistaken for an "empty" or "unfetchable" input.
        self.assertEqual(
            _resolve_media_item("/tmp/some-local-image.png"),
            "/tmp/some-local-image.png",
        )


class TestEncodeImageBytesIsClientError(CustomTestCase):
    """_encode_image_bytes() (image_processing.py) -- plain-Python image decode path."""

    def _encode(self, image_bytes: bytes):
        return _encode_image_bytes(
            image_bytes,
            patch_size=14,
            rescale_image_frac=None,
            rescale_image_max_upscaled_long_edge=None,
        )

    def test_undecodable_image_bytes_raises_value_error(self):
        # PIL raises UnidentifiedImageError, an OSError -- not a ValueError --
        # for bytes it cannot identify as any supported image format.
        with self.assertRaises(ValueError) as ctx:
            self._encode(b"definitely not an image")
        self.assertIn("Could not decode image", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_truncated_image_raises_value_error(self):
        # A recognizable-but-corrupt header (truncated PNG signature) must also
        # be reclassified, not just a fully unrecognized format.
        with self.assertRaisesRegex(ValueError, "Could not decode image"):
            self._encode(b"\x89PNG\r\n\x1a\n")

    def test_broken_chunk_png_raises_value_error(self):
        # PIL raises SyntaxError (not OSError) for a structurally-valid PNG
        # with a corrupted chunk; must be reclassified too.
        with self.assertRaises(ValueError) as ctx:
            self._encode(_broken_chunk_png_bytes())
        self.assertIsInstance(ctx.exception.__cause__, SyntaxError)

    def test_valid_image_still_decodes(self):
        # Regression guard: the try/except must not change behavior for valid input.
        result = self._encode(_png_bytes())
        self.assertEqual(result.shape[-1], 3)


class TestPilDecodeIsClientError(CustomTestCase):
    """_pil_decode() (image_processing_rust.py) -- the default image decode path."""

    def test_undecodable_image_bytes_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _pil_decode(b"definitely not an image")
        self.assertIn("Could not decode image", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_broken_chunk_png_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _pil_decode(_broken_chunk_png_bytes())
        self.assertIsInstance(ctx.exception.__cause__, SyntaxError)

    def test_valid_image_still_decodes(self):
        result = _pil_decode(_png_bytes())
        self.assertEqual(result.shape[-1], 3)


class TestLoadImageBytesIsClientError(CustomTestCase):
    """_load_image_bytes() (image_processing.py) -- shared by both image decode paths."""

    def test_missing_local_path_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _load_image_bytes("/nonexistent/path/that/does/not/exist.png")
        self.assertIn("Could not read image from path", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_path_component_that_is_not_a_directory_raises_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            not_a_directory = Path(temp_dir) / "file"
            not_a_directory.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "Could not read image from path"):
                _load_image_bytes(str(not_a_directory / "child.png"))

    def test_directory_path_raises_value_error(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(ValueError, "Could not read image from path"),
        ):
            _load_image_bytes(temp_dir)

    def test_overlong_path_raises_value_error(self):
        # A raw base64 payload sent without the "data:" prefix falls through
        # _resolve_media_item() as a "plain path" and ends up here.
        with self.assertRaises(ValueError) as ctx:
            _load_image_bytes("a" * 300)
        self.assertIn("Could not read image from path", str(ctx.exception))
        self.assertLess(len(str(ctx.exception)), 200)

    def test_server_os_error_is_not_reclassified_as_client_error(self):
        with (
            patch(
                "builtins.open",
                side_effect=OSError(errno.EMFILE, "too many open files"),
            ),
            self.assertRaises(OSError) as ctx,
        ):
            _load_image_bytes("/any/image.png")
        self.assertEqual(ctx.exception.errno, errno.EMFILE)


class TestLoadAudioBytesIsClientError(CustomTestCase):
    """_load_audio_bytes() (feature_extraction.py) -- same gap as the image loader
    above, for the audio modality."""

    def test_missing_local_path_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _load_audio_bytes("/nonexistent/path/that/does/not/exist.wav")
        self.assertIn("Could not read audio from path", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_path_component_that_is_not_a_directory_raises_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            not_a_directory = Path(temp_dir) / "file"
            not_a_directory.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "Could not read audio from path"):
                _load_audio_bytes(str(not_a_directory / "child.wav"))

    def test_directory_path_raises_value_error(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(ValueError, "Could not read audio from path"),
        ):
            _load_audio_bytes(temp_dir)

    def test_overlong_path_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _load_audio_bytes("a" * 300)
        self.assertIn("Could not read audio from path", str(ctx.exception))
        self.assertLess(len(str(ctx.exception)), 200)

    def test_server_os_error_is_not_reclassified_as_client_error(self):
        with (
            patch(
                "builtins.open",
                side_effect=OSError(errno.EMFILE, "too many open files"),
            ),
            self.assertRaises(OSError) as ctx,
        ):
            _load_audio_bytes("/any/audio.wav")
        self.assertEqual(ctx.exception.errno, errno.EMFILE)


class TestDecodeAudioIsClientError(CustomTestCase):
    """_decode_audio() (feature_extraction.py) -- undecodable audio bytes.
    soundfile raises LibsndfileError, a RuntimeError subclass, so it must be
    explicitly caught and reclassified."""

    def test_undecodable_audio_bytes_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _decode_audio(b"definitely not audio", sample_rate=16000)
        self.assertIn("Could not decode audio", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, sf.LibsndfileError)

    def test_valid_audio_still_decodes(self):
        # Regression guard: a 1-second 440Hz tone should still decode without
        # raising, and resampling (source rate == target rate here, so it's a
        # no-op) must not be affected either.
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, tone, sample_rate, format="WAV")

        result = _decode_audio(buf.getvalue(), sample_rate=sample_rate)
        self.assertEqual(result.shape[0], sample_rate)


if __name__ == "__main__":
    unittest.main()
