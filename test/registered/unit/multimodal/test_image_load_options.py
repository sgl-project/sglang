"""The image loader's options come from mm_process_config, with the model's class as the default.

    legacy processor class / token-space strategy class attrs  ──default──▶  resolve_*  ◀──override── mm_process_config
    (gpu_image_decode, rgb_conversion)                                           │        (image_decoder, image_rgb_conversion)
                                                                                 ▼  resolved once per processor instance
                        _load_single_item(gpu_image_decode=..., rgb_conversion=...)    and passed as arguments
No server, no model loading — pure CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import io
import unittest
from types import SimpleNamespace

from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    resolve_gpu_image_decode,
    resolve_rgb_conversion,
    to_rgb,
)
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    gpu_image_decode = True
    rgb_conversion = "convert"

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class _BareTokenizer:
    def encode(self, text):
        return []

    def decode(self, token_ids):
        return ""


def _rgba_png(color=(10, 20, 30, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _through_png(image: Image.Image, **save_kwargs) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **save_kwargs)
    return Image.open(io.BytesIO(buffer.getvalue()))


class TestImageLoadOptions(CustomTestCase):
    def test_config_overrides_the_class_default(self):
        self.assertIs(resolve_gpu_image_decode({}, True), True)
        self.assertIs(resolve_gpu_image_decode({"image_decoder": "pil"}, True), False)
        self.assertEqual(
            resolve_gpu_image_decode({"image_decoder": "nvjpeg_fancy"}, False),
            "nvjpeg_fancy",
        )
        self.assertEqual(resolve_rgb_conversion({}, "convert"), "convert")
        self.assertEqual(
            resolve_rgb_conversion({"image_rgb_conversion": "white"}, "smart"), "white"
        )
        with self.assertRaises(ValueError):
            resolve_gpu_image_decode({"image_decoder": "cuda"}, True)
        with self.assertRaises(ValueError):
            resolve_rgb_conversion({"image_rgb_conversion": "black"}, "convert")

    def test_white_composites_transparent_pixels_and_convert_drops_alpha(self):
        image = Image.open(io.BytesIO(_rgba_png()))
        self.assertEqual(to_rgb(image, "white").getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(to_rgb(image, "convert").getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(to_rgb(image, "convert").mode, "RGB")

    def test_white_composites_only_an_alpha_channel(self):
        # qwen_vl_utils pastes onto white only for mode RGBA; a grey+alpha image and a palette PNG
        # with a transparent entry go through convert("RGB"), which drops their transparency.
        grey_alpha = _through_png(Image.new("LA", (4, 4), (50, 0)))
        self.assertEqual(grey_alpha.mode, "LA")
        self.assertEqual(to_rgb(grey_alpha, "white").getpixel((0, 0)), (50, 50, 50))
        palette = Image.new("P", (4, 4), 0)
        palette.putpalette([10, 20, 30] + [0] * 765)
        palette = _through_png(palette, transparency=0)
        self.assertEqual((palette.mode, palette.info["transparency"]), ("P", 0))
        self.assertEqual(to_rgb(palette, "white").getpixel((0, 0)), (10, 20, 30))

    def test_a_legacy_processor_resolves_the_config_over_its_class(self):
        # The class says nvJPEG + convert; the server's mm_process_config says PIL + white, and wins.
        reset_context()
        self.addCleanup(reset_context)
        server_args = ServerArgs(
            model_path="dummy",
            mm_process_config={"image_decoder": "pil", "image_rgb_conversion": "white"},
            mm_processor_worker_num=1,
        )
        publish(server_args, role="test")
        hf_config = SimpleNamespace(
            model_type="stub", architectures=[], to_dict=lambda: {"model_type": "stub"}
        )
        processor = _StubProcessor(
            hf_config,
            server_args,
            _BareTokenizer(),
            transport_mode=None,
            skip_mm_pool=True,
        )
        self.assertIs(processor.gpu_image_decode, False)
        self.assertEqual(processor.rgb_conversion, "white")

    def test_the_loader_takes_the_options_as_arguments(self):
        # The class says nvJPEG + convert; the arguments say PIL + white, and win.
        loaded = _StubProcessor._load_single_item(
            _rgba_png(), Modality.IMAGE, gpu_image_decode=False, rgb_conversion="white"
        )
        self.assertIsInstance(loaded, Image.Image)
        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.getpixel((0, 0)), (255, 255, 255))
        # Without arguments the class defaults apply (PIL here because the PNG is not a JPEG).
        loaded = _StubProcessor._load_single_item(_rgba_png(), Modality.IMAGE)
        self.assertEqual(loaded.getpixel((0, 0)), (10, 20, 30))


if __name__ == "__main__":
    unittest.main()
