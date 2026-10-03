"""Unit tests for client-error classification in Inkling's own multimodal
fetch/decode path.

``InklingMultimodalProcessor`` overrides ``process_mm_data_async()`` with its
own media-resolution step (``_resolve_media_item``, in
``sglang.srt.multimodal.processors.inkling``) and its own image decode step
(``_encode_image_bytes``, in ``sglang.srt.multimodal.inkling.image_processing``).
Neither goes through ``BaseMultimodalProcessor._load_single_item`` /
``common.py::load_image``, so the ``CLIENT_MEDIA_EXCEPTIONS`` classification
added there (see ``test_base_processor_bad_input.py`` /
``test_base_processor_image_decode.py``) never covered Inkling. An
unfetchable/empty ``image_url`` or an undecodable image must raise
``ValueError`` (-> HTTP 400 in serving_base.py), not surface as an
unclassified exception (-> HTTP 500).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

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


class TestResolveMediaItemIsClientError(CustomTestCase):
    """_resolve_media_item() — empty/unreachable image_url."""

    def test_empty_url_string_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "empty image_url"):
            _resolve_media_item("")

    def test_empty_url_in_mapping_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "empty image_url"):
            _resolve_media_item({"url": ""})

    def test_empty_url_attribute_raises_value_error(self):
        class _ImageData:
            url = ""

        with self.assertRaisesRegex(ValueError, "empty image_url"):
            _resolve_media_item(_ImageData())

    def test_whitespace_only_url_raises_value_error(self):
        # A client sending "   " instead of "" must be rejected the same way,
        # not silently fall through to being treated as a plain file path.
        for whitespace_url in ("   ", "\t", "\n", " \t\n "):
            with self.subTest(url=repr(whitespace_url)):
                with self.assertRaisesRegex(ValueError, "empty image_url"):
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

    def test_invalid_base64_data_url_raises_value_error(self):
        with self.assertRaises(ValueError):
            _resolve_media_item({"url": "data:image/png;base64,!!!not-base64!!!"})

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
    """_encode_image_bytes() — unsupported/undecodable image format."""

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
        with self.assertRaisesRegex(ValueError, "Could not decode image"):
            self._encode(b"definitely not an image")

    def test_truncated_image_raises_value_error(self):
        # A recognizable-but-corrupt header (truncated PNG signature) must also
        # be reclassified, not just a fully unrecognized format.
        png_signature_only = b"\x89PNG\r\n\x1a\n"
        with self.assertRaisesRegex(ValueError, "Could not decode image"):
            self._encode(png_signature_only)

    def test_value_error_cause_is_the_original_os_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._encode(b"definitely not an image")
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_valid_image_still_decodes(self):
        # Regression guard: the try/except must not change behavior for valid
        # input. A 16x16 RGB PNG should encode without raising.
        import io

        import numpy as np
        from PIL import Image

        arr = (np.random.RandomState(0).rand(16, 16, 3) * 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")

        result = self._encode(buf.getvalue())
        self.assertEqual(result.shape[-1], 3)


class TestPilDecodeIsClientError(CustomTestCase):
    """image_processing_rust.py::_pil_decode() -- the decode path actually used by
    default, since SGLANG_INKLING_RS_MM_PREPROCESS defaults to True and routes
    Inkling to InklingRustImageProcessor instead of the plain-Python processor
    covered by TestEncodeImageBytesIsClientError above."""

    def test_undecodable_image_bytes_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Could not decode image"):
            _pil_decode(b"definitely not an image")

    def test_value_error_cause_is_the_original_os_error(self):
        with self.assertRaises(ValueError) as ctx:
            _pil_decode(b"definitely not an image")
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_valid_image_still_decodes(self):
        # Regression guard: a 16x16 RGB PNG should still decode without raising.
        import io

        import numpy as np
        from PIL import Image

        arr = (np.random.RandomState(0).rand(16, 16, 3) * 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")

        result = _pil_decode(buf.getvalue())
        self.assertEqual(result.shape[-1], 3)


class TestLoadImageBytesIsClientError(CustomTestCase):
    """image_processing.py::_load_image_bytes() -- shared by both the plain-Python
    and Rust-accelerated processors, and runs *before* either decode step above.
    A garbage image_url that isn't empty/data:/http(s):// falls through
    _resolve_media_item() unresolved and lands here as a "plain path"."""

    def test_missing_local_path_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Could not read image from path"):
            _load_image_bytes("/nonexistent/path/that/does/not/exist.png")

    def test_value_error_cause_is_the_original_os_error(self):
        with self.assertRaises(ValueError) as ctx:
            _load_image_bytes("/nonexistent/path/that/does/not/exist.png")
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
    """feature_extraction.py::_load_audio_bytes() -- same gap as
    TestLoadImageBytesIsClientError, but for the audio modality."""

    def test_missing_local_path_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Could not read audio from path"):
            _load_audio_bytes("/nonexistent/path/that/does/not/exist.wav")

    def test_value_error_cause_is_the_original_os_error(self):
        with self.assertRaises(ValueError) as ctx:
            _load_audio_bytes("/nonexistent/path/that/does/not/exist.wav")
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
    """feature_extraction.py::_decode_audio() -- mirrors the image-decode bug
    (unclassified exception -> HTTP 500) but for undecodable audio bytes.
    soundfile raises LibsndfileError, a RuntimeError subclass -- not OSError,
    not ValueError -- so it must be explicitly caught and reclassified."""

    def test_undecodable_audio_bytes_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Could not decode audio"):
            _decode_audio(b"definitely not audio", sample_rate=16000)

    def test_value_error_cause_is_libsndfile_error(self):
        import soundfile as sf

        with self.assertRaises(ValueError) as ctx:
            _decode_audio(b"definitely not audio", sample_rate=16000)
        self.assertIsInstance(ctx.exception.__cause__, sf.LibsndfileError)

    def test_valid_audio_still_decodes(self):
        # Regression guard: a 1-second 440Hz tone should still decode without
        # raising, and resampling (source rate == target rate here, so it's a
        # no-op) must not be affected either.
        import io

        import numpy as np
        import soundfile as sf

        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, tone, sample_rate, format="WAV")

        result = _decode_audio(buf.getvalue(), sample_rate=sample_rate)
        self.assertEqual(result.shape[0], sample_rate)


if __name__ == "__main__":
    unittest.main()
