"""A token-space strategy's image loader options come from mm_process_config, with the strategy as the default.

    strategy class attrs  ──default──▶  resolve_* in TokenSpaceProcessStrategy.__init__  ◀──override── mm_process_config
    (gpu_image_decode,                                    │                                (image_decoder,
     rgb_conversion)                                      ▼                                 image_rgb_conversion)
                                 strategy.load_image(source) -> load_image_item(..., rgb_conversion) -> to_rgb
    legacy processors keep their class attributes; the two keys are rejected without --enable-token-space-processor
No server, no model loading — pure CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import io
import unittest
from types import SimpleNamespace

from PIL import Image

from sglang.srt.arg_groups.serving_hook import handle_multimodal
from sglang.srt.managers.multimodal_processor import get_mm_processor_cls
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    resolve_gpu_image_decode,
    resolve_rgb_conversion,
    to_rgb,
)
from sglang.test.test_utils import CustomTestCase


class _WhiteStrategy(TokenSpaceProcessStrategy):
    rgb_conversion = "white"


def _strategy(strategy_class=TokenSpaceProcessStrategy, **mm_process_config):
    return strategy_class(SimpleNamespace(), None, mm_process_config=mm_process_config)


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
        # JSON null means the default; a wrong name or a non-string is a ValueError, never a TypeError.
        self.assertIs(resolve_gpu_image_decode({"image_decoder": None}, True), True)
        self.assertEqual(
            resolve_rgb_conversion({"image_rgb_conversion": None}, "smart"), "smart"
        )
        for bad in ({"image_decoder": "cuda"}, {"image_decoder": ["pil"]}):
            with self.assertRaises(ValueError):
                resolve_gpu_image_decode(bad, True)
        for bad in ({"image_rgb_conversion": "black"}, {"image_rgb_conversion": 1}):
            with self.assertRaises(ValueError):
                resolve_rgb_conversion(bad, "convert")

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

    def test_the_strategy_loads_with_its_class_default_and_the_config_wins(self):
        # The base strategy drops alpha; a model's `white` default composites it; the config overrides either.
        self.assertEqual(
            _strategy().load_image(_rgba_png()).getpixel((0, 0)), (10, 20, 30)
        )
        white = _strategy(_WhiteStrategy)
        self.assertEqual(
            white.load_image(_rgba_png()).getpixel((0, 0)), (255, 255, 255)
        )
        self.assertEqual(white.load_image(_rgba_png()).mode, "RGB")
        overridden = _strategy(
            _WhiteStrategy, image_decoder="pil", image_rgb_conversion="convert"
        )
        self.assertIs(overridden.gpu_image_decode, False)
        self.assertEqual(
            overridden.load_image(_rgba_png()).getpixel((0, 0)), (10, 20, 30)
        )
        with self.assertRaises(ValueError):
            _strategy(image_rgb_conversion="black")

    def test_the_keys_need_the_token_space_processor(self):
        # Legacy processors never read the keys, so a server without the token-space processor rejects them,
        # and with it a bad value is rejected at argument resolution, before any process starts.
        with self.assertRaises(ValueError):
            handle_multimodal(
                ServerArgs(
                    model_path="dummy", mm_process_config={"image_decoder": "pil"}
                )
            )
        with self.assertRaises(ValueError):
            handle_multimodal(
                ServerArgs(
                    model_path="dummy",
                    mm_process_config={"image_rgb_conversion": "White"},
                    enable_token_space_processor=True,
                )
            )
        handle_multimodal(
            ServerArgs(
                model_path="dummy",
                mm_process_config={
                    "image_decoder": "pil",
                    "image_rgb_conversion": None,
                },
                enable_token_space_processor=True,
            )
        )

    def test_a_model_without_a_strategy_rejects_the_keys(self):
        # With the token-space processor on, a model that still gets a legacy processor would ignore the keys.
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                mm_process_config={"image_decoder": "pil"},
                enable_token_space_processor=True,
            ),
            role="test",
        )
        hf_config = SimpleNamespace(
            model_type="stub", architectures=["NoStrategyForConditionalGeneration"]
        )
        with self.assertRaisesRegex(ValueError, "token-space processor only"):
            get_mm_processor_cls(hf_config)


if __name__ == "__main__":
    unittest.main()
