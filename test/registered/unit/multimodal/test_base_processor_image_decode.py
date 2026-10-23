"""Unit tests for ``BaseMultimodalProcessor._load_single_item`` image decoding.

Regression test for the change that forces the (otherwise lazy) PIL decode inside
``_load_single_item`` — which runs in the ``io_executor`` worker thread — instead of
letting it fire lazily on the main event-loop thread later (inside
``pil_to_tensor``/``tobytes`` during processing). The behavior of the returned image
(mode, pixels) must be unchanged; only *when/where* the decode happens differs.

No server, no model loading — pure CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import asyncio
import concurrent.futures
import io
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests
import torch
from PIL import ExifTags, Image, ImageOps

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import common
from sglang.srt.utils.nvjpeg_decoder import _NvJpegDecoderPool
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


def _jpeg_bytes(size=(8, 8)) -> bytes:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=90, subsampling=2)
    return buf.getvalue()


def _encode_with_orientation(fmt: str, orientation: int, size=(16, 8)) -> bytes:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    img = Image.fromarray(arr, "RGB")
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    buf = io.BytesIO()
    img.save(buf, format=fmt, exif=exif)
    return buf.getvalue()


def _fake_torchcodec(decode_jpeg):
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.decode_jpeg = decode_jpeg
    torchcodec = types.ModuleType("torchcodec")
    torchcodec.decoders = decoders
    return {"torchcodec": torchcodec, "torchcodec.decoders": decoders}


def _is_decoded(img: Image.Image) -> bool:
    """A lazily-opened PIL image has no decoded core yet; ``load()`` populates it.
    PIL's ``.im`` property requires a completed load and raises otherwise."""
    try:
        return img.im is not None
    except Exception:
        return False


class TestLoadSingleItemImageDecode(CustomTestCase):
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

    def test_high_fidelity_gpu_jpeg_decoder_is_selected(self):
        data = _jpeg_bytes()
        expected = torch.zeros((3, 8, 8), dtype=torch.uint8)
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch(
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
                return_value=expected,
            ) as decode,
        ):
            image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")

        self.assertIs(image, expected)
        decode.assert_called_once_with(data)

    def test_high_fidelity_gpu_jpeg_decoder_falls_back_to_pil(self):
        data = _jpeg_bytes()
        common._warn_fancy_jpeg_fallback.cache_clear()
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch(
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
                side_effect=ImportError("nvImageCodec is unavailable"),
            ),
        ):
            image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")

        self.assertIsInstance(image, Image.Image)
        reference = Image.open(io.BytesIO(data))
        np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))

    def test_high_fidelity_decoder_uses_fancy_planar_rgb_and_reuses_pool(self):
        expected = torch.zeros((3, 8, 8), dtype=torch.uint8)
        fake_format = object()

        class FakeImage:
            def to_dlpack(self, *, cuda_stream):
                self.cuda_stream = cuda_stream
                return object()

        class FakeDecoder:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.instances.append(self)

            def decode(self, data, *, params, cuda_stream):
                self.call = (data, params, cuda_stream)
                return FakeImage()

        class FakeDecodeParams:
            def __init__(self, *, sample_format, apply_exif_orientation):
                self.sample_format = sample_format
                self.apply_exif_orientation = apply_exif_orientation

        fake_codec = SimpleNamespace(
            DecodeParams=FakeDecodeParams,
            Decoder=FakeDecoder,
            SampleFormat=SimpleNamespace(P_RGB=fake_format),
        )
        nvidia = types.ModuleType("nvidia")
        nvidia.nvimgcodec = fake_codec

        with (
            patch.dict(sys.modules, {"nvidia": nvidia}),
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=7),
            ),
            patch.object(torch, "from_dlpack", return_value=expected),
        ):
            pool = _NvJpegDecoderPool(device_id=2)
            self.assertIs(pool.decode(b"jpeg"), expected)
            self.assertIs(pool.decode(b"jpeg"), expected)

        self.assertEqual(len(FakeDecoder.instances), 1)
        decoder = FakeDecoder.instances[0]
        self.assertEqual(decoder.kwargs["device_id"], 2)
        self.assertEqual(decoder.kwargs["max_num_cpu_threads"], 1)
        self.assertIn(":fancy_upsampling=1", decoder.kwargs["options"])
        self.assertIs(pool._decode_params.sample_format, fake_format)
        self.assertFalse(pool._decode_params.apply_exif_orientation)

    def test_gpu_jpeg_decode_failure_falls_back_to_pil(self):
        data = _jpeg_bytes()
        common._warn_gpu_jpeg_fallback.cache_clear()
        decode = Mock(side_effect=RuntimeError("nvJPEG failed"))
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch.dict(sys.modules, _fake_torchcodec(decode)),
        ):
            image, _ = common.load_image(data, gpu_image_decode=True)

        self.assertIsInstance(image, Image.Image)
        reference = Image.open(io.BytesIO(data))
        np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))

    def test_gpu_jpeg_decode_without_torchcodec_falls_back_to_pil(self):
        """Hosts where torchcodec is missing or fails to load still serve JPEGs."""
        data = _jpeg_bytes()
        common._warn_gpu_jpeg_fallback.cache_clear()
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch.dict(sys.modules, {"torchcodec": None, "torchcodec.decoders": None}),
        ):
            image, _ = common.load_image(data, gpu_image_decode=True)

        self.assertIsInstance(image, Image.Image)


class TestExifOrientation(CustomTestCase):
    def test_pil_decode_applies_exif_orientation(self):
        """Non-regression test for https://github.com/sgl-project/sglang/issues/33464 (PIL path).

        Phone photos store landscape pixels plus an Orientation tag; the model
        must see the upright image a viewer shows, not the raw stored pixels.
        """
        for fmt in ("JPEG", "PNG", "WEBP"):
            for orientation in range(1, 9):
                with self.subTest(fmt=fmt, orientation=orientation):
                    data = _encode_with_orientation(fmt, orientation)
                    image = _StubProcessor._load_single_item(data, Modality.IMAGE)
                    reference = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
                    self.assertEqual(image.size, reference.size)
                    np.testing.assert_array_equal(
                        np.asarray(image), np.asarray(reference.convert("RGB"))
                    )

    def test_pil_input_is_oriented_without_mutating_caller_image(self):
        """Engine callers pass PIL images they still own; orientation must not
        rotate their object in place."""
        data = _encode_with_orientation("JPEG", 6, size=(16, 8))
        caller_image = Image.open(io.BytesIO(data))
        image, image_size = common.load_image(caller_image, gpu_image_decode=False)

        self.assertEqual(image.size, (8, 16))
        self.assertEqual(image_size, (8, 16))
        self.assertEqual(caller_image.size, (16, 8))
        self.assertEqual(caller_image.getexif()[ExifTags.Base.Orientation], 6)

    def test_unrotated_image_is_not_copied(self):
        """Pillow's exif_transpose copies even when there is nothing to undo."""
        data = _encode_with_orientation("JPEG", 1)
        image = Image.open(io.BytesIO(data))
        image.load()
        self.assertIs(common._apply_exif_orientation(image), image)

    def test_opted_out_processor_keeps_stored_orientation(self):
        """Processors whose GPU decoder ignores EXIF (nvImageCodec) opt out, so
        every decode path, torchcodec included, must leave the pixels as stored."""

        class _NoExifProcessor(_StubProcessor):
            gpu_image_decode = True
            apply_exif_orientation = False

        data = _encode_with_orientation("JPEG", 6, size=(16, 8))
        decode = Mock()
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch.dict(sys.modules, _fake_torchcodec(decode)),
        ):
            image = _NoExifProcessor._load_single_item(data, Modality.IMAGE)

        decode.assert_not_called()
        self.assertEqual(image.size, (16, 8))

    def test_malformed_exif_is_treated_as_no_rotation(self):
        data = _jpeg_bytes(size=(16, 8))
        with patch.object(
            Image.Image, "getexif", side_effect=SyntaxError("corrupt EXIF")
        ):
            image, _ = common.load_image(data, gpu_image_decode=False)
        self.assertEqual(image.size, (16, 8))


if __name__ == "__main__":
    unittest.main()
