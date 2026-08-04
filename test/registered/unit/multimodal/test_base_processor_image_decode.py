"""Unit tests for ``BaseMultimodalProcessor._load_single_item`` image decoding.

Regression test for the change that forces the (otherwise lazy) PIL decode inside
``_load_single_item`` — which runs in the ``io_executor`` worker thread — instead of
letting it fire lazily on the main event-loop thread later (inside
``pil_to_tensor``/``tobytes`` during processing). The behavior of the returned image
(mode, pixels) must be unchanged; only *when/where* the decode happens differs.

No server, no model loading — pure CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import concurrent.futures
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests
from PIL import Image, ImageOps

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils.common import _get_exif_orientation, load_image
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    # gpu_image_decode=False forces the PIL (CPU) path so the test needs no GPU and
    # exercises exactly the lazy-decode branch the fix targets. The abstract methods
    # are never called: we only invoke the _load_single_item classmethod.
    gpu_image_decode = False

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _png_bytes(mode: str = "RGB", size=(8, 8)) -> bytes:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    img = Image.fromarray(arr, "RGB").convert(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_decoded(img: Image.Image) -> bool:
    """A lazily-opened PIL image has no decoded core yet; ``load()`` populates it.
    PIL's ``.im`` property requires a completed load and raises otherwise."""
    try:
        return img.im is not None
    except Exception:
        return False


class TestLoadSingleItemImageDecode(CustomTestCase):
    def test_plain_open_is_lazy(self):
        # Documents why the fix matters: a bare Image.open is not decoded yet, so
        # without the fix the decode would land on the caller (main) thread.
        lazy = Image.open(io.BytesIO(_png_bytes()))
        self.assertFalse(_is_decoded(lazy))

    def test_load_single_item_forces_decode(self):
        img = _StubProcessor._load_single_item(_png_bytes("RGB"), Modality.IMAGE)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        # The fix: decode is forced inside _load_single_item, not lazily later.
        self.assertTrue(_is_decoded(img))

    def test_rgba_converted_to_rgb_and_decoded(self):
        img = _StubProcessor._load_single_item(_png_bytes("RGBA"), Modality.IMAGE)
        # Existing alpha-discard behavior preserved.
        self.assertEqual(img.mode, "RGB")
        self.assertTrue(_is_decoded(img))

    def test_pixels_match_reference(self):
        # Output must be bit-identical to the pre-fix path (open -> [convert]).
        data = _png_bytes("RGB")
        img = _StubProcessor._load_single_item(data, Modality.IMAGE)
        ref = Image.open(io.BytesIO(data)).convert("RGB")
        np.testing.assert_array_equal(np.asarray(img), np.asarray(ref))

    def test_fast_loader_preserves_invalid_input_as_value_error(self):
        processor = object.__new__(_StubProcessor)
        future = concurrent.futures.Future()
        future.set_exception(ValueError("invalid base64 image"))
        processor._submit_mm_data_loading_tasks_simple = Mock(
            side_effect=[[(Modality.IMAGE, 0, future)], [], []]
        )

        with self.assertRaisesRegex(ValueError, "invalid base64 image"):
            asyncio.run(
                processor.fast_load_mm_data(
                    prompt="<image>",
                    multimodal_tokens=Mock(),
                    image_data=["bad-image"],
                )
            )

    def test_unreachable_image_url_is_a_client_error(self):
        with patch(
            "sglang.srt.multimodal.processors.base_processor.load_image",
            side_effect=requests.ConnectionError("connection refused"),
        ):
            with self.assertRaisesRegex(ValueError, "connection refused"):
                _StubProcessor._load_single_item(
                    "https://127.0.0.1:1/not-an-image.png", Modality.IMAGE
                )

    def test_invalid_image_bytes_are_a_client_error(self):
        with self.assertRaisesRegex(ValueError, "cannot identify image file"):
            _StubProcessor._load_single_item(b"not an image", Modality.IMAGE)

    def test_unexpected_loader_bug_remains_a_server_error(self):
        with patch(
            "sglang.srt.multimodal.processors.base_processor.load_image",
            side_effect=TypeError("unexpected loader bug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected loader bug"):
                _StubProcessor._load_single_item(b"image", Modality.IMAGE)


def _make_jpeg_with_orientation(width: int, height: int, orientation: int) -> bytes:
    """Create a JPEG with the given EXIF orientation tag.

    The image content is a simple gradient so that different orientations
    produce visually distinct pixel arrangements.
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [x % 256, y % 256, (x + y) % 256]
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    # exif= expects an instance of PIL.ExifImage; build it via getexif
    exif = img.getexif()
    if orientation != 1:
        exif[0x0112] = orientation
        img.save(buf, format="JPEG", exif=exif)
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


class TestExifOrientation(CustomTestCase):
    """Verify that image loading applies EXIF orientation so the model sees the
    same pixels a human viewer sees.

    Covers the three code paths in ``sglang.srt.utils.common``:
    * ``_load_image`` CPU fallback (``Image.open`` + ``exif_transpose``)
    * ``load_image`` direct-PIL branch (``isinstance(image_file, Image.Image)``)
    * ``_get_exif_orientation`` helper (unit-tested without CUDA)
    """

    # ------------------------------------------------------------------
    # _get_exif_orientation helper
    # ------------------------------------------------------------------

    def test_get_orientation_no_exif(self):
        """A JPEG with no EXIF returns orientation 1."""
        data = _make_jpeg_with_orientation(16, 12, orientation=1)
        self.assertEqual(_get_exif_orientation(data), 1)

    def test_get_orientation_all_values(self):
        """All 8 EXIF orientation values are correctly read from raw bytes."""
        for orientation in range(1, 9):
            data = _make_jpeg_with_orientation(16, 12, orientation=orientation)
            self.assertEqual(
                _get_exif_orientation(data),
                orientation,
                f"orientation {orientation} not read correctly",
            )

    def test_get_orientation_non_jpeg_returns_1(self):
        """PNG inputs have no EXIF orientation and return 1."""
        data = _png_bytes("RGB")
        self.assertEqual(_get_exif_orientation(data), 1)

    def test_get_orientation_malformed_returns_1(self):
        """Malformed bytes never raise — they degrade to orientation 1."""
        self.assertEqual(_get_exif_orientation(b"not an image"), 1)
        self.assertEqual(_get_exif_orientation(b""), 1)
        # JPEG magic bytes but truncated body
        self.assertEqual(_get_exif_orientation(b"\xff\xd8\xff\xe0"), 1)

    def test_get_orientation_out_of_range_returns_1(self):
        """Orientation values outside 1-8 are clamped to 1 (no rotation)."""
        # Manually craft a JPEG with orientation=99 (invalid)
        img = Image.new("RGB", (8, 8), color=(128, 128, 128))
        exif = img.getexif()
        exif[0x0112] = 99
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        self.assertEqual(_get_exif_orientation(buf.getvalue()), 1)

    # ------------------------------------------------------------------
    # _load_image CPU path (no CUDA required)
    # ------------------------------------------------------------------

    def test_load_image_cpu_all_orientations_match_exif_transpose(self):
        """For every EXIF orientation value, ``_load_image`` on the CPU path
        produces pixels identical to ``PIL.ImageOps.exif_transpose`` applied
        to the same input. This single parity assertion eliminates every
        transpose/flip sign error."""
        from sglang.srt.utils.common import _load_image

        for orientation in range(1, 9):
            data = _make_jpeg_with_orientation(24, 16, orientation=orientation)
            result = _load_image(image_bytes=data, gpu_image_decode=False)
            self.assertIsInstance(result, Image.Image, f"orientation {orientation}")
            ref = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            np.testing.assert_array_equal(
                np.asarray(result),
                np.asarray(ref),
                err_msg=f"pixel mismatch for orientation {orientation}",
            )

    def test_load_image_cpu_no_exif_unchanged(self):
        """JPEG without EXIF orientation must be returned as-is (no rotation)."""
        from sglang.srt.utils.common import _load_image

        data = _make_jpeg_with_orientation(16, 12, orientation=1)
        result = _load_image(image_bytes=data, gpu_image_decode=False)
        ref = Image.open(io.BytesIO(data))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(ref))

    def test_load_image_cpu_malformed_exif_no_crash(self):
        """Malformed EXIF must not cause a crash — image is returned unrotated."""
        from sglang.srt.utils.common import _load_image

        # Non-image garbage bytes should still go through the CPU path
        # (Image.open will raise, but _load_image propagates that)
        with self.assertRaises(Exception):
            _load_image(image_bytes=b"not an image", gpu_image_decode=False)

    def test_load_image_cpu_corrupt_exif_preserves_image(self):
        """A valid JPEG with corrupt EXIF metadata must still decode and return
        unrotated pixels, not raise. This guards against the regression where
        exif_transpose itself raises on malformed EXIF."""
        from sglang.srt.utils.common import _load_image

        # Build a valid JPEG, then inject a corrupt APP1/EXIF segment
        img = Image.new("RGB", (12, 10), color=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        valid_jpeg = buf.getvalue()

        # Replace the image with one that has corrupt EXIF: prepend a malformed
        # APP1 segment (valid marker + Exif header + garbage TIFF IFD)
        import struct

        corrupt_exif = b"Exif\x00\x00"  # Exif header
        corrupt_exif += b"MM\x00\x2a"  # Big-endian byte order + TIFF magic
        corrupt_exif += b"\xff\xff\xff\xff"  # IFD offset = huge (corrupt)
        corrupt_exif += b"\x00" * 20  # garbage padding
        app1_len = len(corrupt_exif) + 2
        app1 = b"\xff\xe1" + struct.pack(">H", app1_len) + corrupt_exif
        corrupt_jpeg = b"\xff\xd8" + app1 + valid_jpeg[2:]

        # The image must decode without raising and return pixels (unrotated
        # since the corrupt EXIF orientation degrades to 1).
        result = _load_image(image_bytes=corrupt_jpeg, gpu_image_decode=False)
        self.assertIsInstance(result, Image.Image)

    def test_safe_exif_transpose_returns_image_on_failure(self):
        """``_safe_exif_transpose`` must return the original image when
        exif_transpose raises, never propagate the exception."""
        from sglang.srt.utils.common import _safe_exif_transpose

        # Use an image with orientation=6 so the code path actually reaches
        # exif_transpose (orientation 1 short-circuits before calling it).
        data = _make_jpeg_with_orientation(8, 8, orientation=6)
        img = Image.open(io.BytesIO(data))
        with patch(
            "sglang.srt.utils.common.ImageOps.exif_transpose",
            side_effect=RuntimeError("bad EXIF"),
        ):
            result = _safe_exif_transpose(img)
            self.assertIs(result, img)

    # ------------------------------------------------------------------
    # load_image direct-PIL branch
    # ------------------------------------------------------------------

    def test_load_image_direct_pil_applies_exif_transpose(self):
        """``load_image`` must normalize a directly-passed PIL Image, not just
        bytes/paths. This covers the third touch point that vLLM missed in
        their initial fix (PR #47566)."""
        for orientation in range(1, 9):
            data = _make_jpeg_with_orientation(24, 16, orientation=orientation)
            raw_img = Image.open(io.BytesIO(data))
            result, result_size = load_image(raw_img, gpu_image_decode=False)
            self.assertIsInstance(result, Image.Image)
            ref = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            np.testing.assert_array_equal(
                np.asarray(result),
                np.asarray(ref),
                err_msg=f"direct-PIL pixel mismatch for orientation {orientation}",
            )
            self.assertEqual(result_size, (ref.width, ref.height))

    def test_load_image_direct_pil_no_exif_unchanged(self):
        """A PIL Image with no EXIF orientation must be returned as-is."""
        img = Image.new("RGB", (16, 12), color=(100, 150, 200))
        result, result_size = load_image(img, gpu_image_decode=False)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(img))
        self.assertEqual(result_size, (16, 12))

    def test_safe_exif_transpose_short_circuits_no_exif(self):
        """When no EXIF orientation is present, ``_safe_exif_transpose`` must
        return the exact same object (no copy), preserving lazy/multi-frame
        image semantics."""
        from sglang.srt.utils.common import _safe_exif_transpose

        img = Image.new("RGB", (16, 12), color=(100, 150, 200))
        result = _safe_exif_transpose(img)
        self.assertIs(result, img)

    # ------------------------------------------------------------------
    # nvJPEG branch routing (mocked, so no CUDA is required)
    # ------------------------------------------------------------------

    def test_gpu_branch_skipped_for_nontrivial_orientation(self):
        """The nvJPEG branch returns raw sensor pixels and cannot apply EXIF
        orientation. When a non-trivial orientation is present, ``_load_image``
        must not call ``decode_jpeg`` at all and must return an already-rotated
        PIL Image instead.

        This is the regression guard for the primary path: on a CUDA server
        every JPEG is routed to nvJPEG, so a CPU-only fix would be a no-op for
        real phone photos."""
        from sglang.srt.utils import common as common_mod

        for orientation in range(2, 9):
            data = _make_jpeg_with_orientation(24, 16, orientation=orientation)
            with patch.object(
                common_mod, "is_jpeg_with_cuda", return_value=True
            ), patch.object(common_mod, "decode_jpeg") as mock_decode:
                result = common_mod._load_image(image_bytes=data, gpu_image_decode=True)
            mock_decode.assert_not_called()
            self.assertIsInstance(result, Image.Image)
            ref = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            np.testing.assert_array_equal(
                np.asarray(result),
                np.asarray(ref),
                err_msg=f"gpu-branch fallback mismatch for orientation {orientation}",
            )

    def test_gpu_branch_used_for_trivial_orientation(self):
        """The common case (orientation 1 or absent) must still take the fast
        nvJPEG path, so the fix costs nothing for unrotated images."""
        import torch

        from sglang.srt.utils import common as common_mod

        data = _make_jpeg_with_orientation(24, 16, orientation=1)
        sentinel = torch.zeros(3, 16, 24, dtype=torch.uint8)
        with patch.object(
            common_mod, "is_jpeg_with_cuda", return_value=True
        ), patch.object(
            common_mod, "decode_jpeg", return_value=sentinel
        ) as mock_decode:
            result = common_mod._load_image(image_bytes=data, gpu_image_decode=True)
        mock_decode.assert_called_once()
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
