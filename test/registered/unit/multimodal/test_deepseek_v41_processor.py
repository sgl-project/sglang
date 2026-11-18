"""DeepSeek-V4.1 image placeholders in a tokenized prompt."""

import asyncio
import base64
import io
import unittest
from types import SimpleNamespace

from PIL import Image

from sglang.srt.multimodal.deepseek_v41_image_processing import (
    image_token_types,
    plan_image_grid,
)
from sglang.srt.multimodal.media_io import ImageData
from sglang.srt.multimodal.processors.deepseek_v41 import DeepseekV41ImageProcessor
from sglang.srt.parser.deepseek_recipe_renderer import IMAGE_TOKEN, DeepSeekRecipe
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SIZES = {"wide": (640, 320), "tall": (320, 640), "square": (560, 560)}


def _tokenizer_file():
    """The V4.1 tokenizer; only outside CI does an unreachable Hub skip the test."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    except Exception as error:  # noqa: BLE001
        if is_in_ci():
            raise
        raise unittest.SkipTest(f"the tokenizer is not available: {error}")


def _png(name):
    buffer = io.BytesIO()
    Image.new("RGB", SIZES[name], (10, 120, 200)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _image(name):
    return {"type": "image_url", "image_url": {"url": _png(name)}}


def _text(value):
    return {"type": "text", "text": value}


class TestDeepseekV41ImageProcessor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from transformers import PreTrainedTokenizerFast

        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_file=_tokenizer_file())
        cls.image_token_id = cls.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        cls.recipe = DeepSeekRecipe.from_hf_tokenizer("dsv41", cls.tokenizer)

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        args = ServerArgs(
            model_path="dummy", image_processor_backend="pil", mm_process_config={}
        )
        publish(args, role="tokenizer")
        # Vision values from the V4.1 checkpoint config.
        self.config = SimpleNamespace(
            image_token_id=self.image_token_id,
            vision_patch_size=14,
            vision_downsample_ratio=3,
            vision_max_n_token=1024,
            vision_min_pixels=295936,
            vision_max_wh_ratio=None,
        )
        self.processor = DeepseekV41ImageProcessor(
            self.config, args, self.tokenizer, "default"
        )

    def _prompt(self, content, tools=None):
        message = {"role": "system", "content": content}
        if tools:
            message["tools"] = tools
        urls = []
        ids = self.recipe.encode(
            {
                "messages": [message, {"role": "user", "content": "hi"}],
                "thinking": False,
                "tool_choice": "auto",
            },
            image_urls=urls,
        )
        return ids, [ImageData(url=url) for url in urls]

    def _process(self, prompt, images):
        return asyncio.run(
            self.processor.process_mm_data_async(images, prompt, request_obj=None)
        )

    def _expanded(self, name):
        width, height = SIZES[name]
        llm_h, llm_w, best_height, best_width = plan_image_grid(
            width, height, self.config
        )
        return len(image_token_types(llm_h, llm_w)), best_height // 14, best_width // 14

    def test_tokenized_prompt_counts_placeholders_by_id(self):
        """Text that spells the image token is not an image placeholder: the prompt
        keeps it, and each image expands at its own placeholder in order."""
        literal = f"literal {IMAGE_TOKEN}"
        tool = {"function": {"name": "create_ticket", "parameters": {"type": "object"}}}
        for names, content, tools in (
            (["wide"], [_text("see"), _image("wide"), _text(literal)], [tool]),
            (["wide"], [_text("see"), _image("wide"), _text(literal)], None),
            (
                ["tall", "wide", "square"],
                [
                    _image("tall"),
                    _text("one"),
                    _image("wide"),
                    _text("two"),
                    _image("square"),
                    _text(f"{literal} {literal}"),
                ],
                [tool],
            ),
        ):
            with self.subTest(names=names, tools=bool(tools)):
                ids, images = self._prompt(content, tools)
                self.assertEqual(ids.count(self.image_token_id), len(names))
                output = self._process(ids, images)
                expected_ids, expected_offsets, expected_grids = [], [], []
                expanded = iter(self._expanded(name) for name in names)
                for token_id in ids:
                    if token_id != self.image_token_id:
                        expected_ids.append(token_id)
                        continue
                    count, vit_h, vit_w = next(expanded)
                    start = len(expected_ids)
                    expected_ids += [self.image_token_id] * count
                    expected_offsets.append([(start, start + count - 1)])
                    expected_grids.append({"n_vit_h": vit_h, "n_vit_w": vit_w})
                self.assertEqual(output.input_ids, expected_ids)
                self.assertEqual(
                    [item.offsets for item in output.mm_items], expected_offsets
                )
                self.assertEqual(
                    [item.model_specific_data for item in output.mm_items],
                    expected_grids,
                )
        ids, images = self._prompt([_image("wide"), _text(literal)])
        with self.assertRaisesRegex(ValueError, "placeholders and images must match"):
            self._process(ids, images * 2)

    def test_text_prompt_counts_placeholders_by_spelling(self):
        """A text prompt has only spellings, so each one needs an image."""
        output = self._process(f"see {IMAGE_TOKEN}", [ImageData(url=_png("wide"))])
        self.assertEqual(len(output.mm_items), 1)
        with self.assertRaisesRegex(ValueError, "2 image placeholders but 1 image"):
            self._process(
                f"see {IMAGE_TOKEN} literal {IMAGE_TOKEN}",
                [ImageData(url=_png("wide"))],
            )


if __name__ == "__main__":
    unittest.main()
