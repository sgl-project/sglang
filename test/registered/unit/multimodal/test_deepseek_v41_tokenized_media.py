"""CPU parity of the DeepSeek-V4.1 token-space strategy with the legacy processor; no weights.

serving   served output == legacy output: expanded ids, one item per image with its offsets, BF16 patches and
          n_vit_h / n_vit_w, for a square PNG, a wide JPEG and a tall PNG in one prompt
          -> test_serving_output_matches_legacy_processor
layout    two images with equal patches and transposed grids hash apart
          -> test_transposed_grids_hash_apart
towers    a checkpoint without a vision tower rejects images
          -> test_text_only_checkpoint_rejects_images
"""

import asyncio
import io
import unittest

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from sglang.srt.configs.deepseek_v41 import (
    DeepseekV41Config,
    normalize_deepseek_v41_config,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.multimodal.processors.deepseek_v41 import DeepseekV41ImageProcessor
from sglang.srt.multimodal.token_space_mm_process.models.deepseek_v41 import (
    DeepseekV41TokenSpaceProcessStrategy,
    DeepseekV41TokenSpaceServingSpec,
)
from sglang.srt.multimodal.token_space_mm_process.sglang import (
    TokenSpaceSGLangProcessor,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

IMAGE = 129264
VOCAB = {"[UNK]": 0, "hi": 1, "there": 2, "<image>": IMAGE}

DeepseekV41TokenSpaceProcessor = (
    TokenSpaceSGLangProcessor.for_strategy_and_serving_spec(
        DeepseekV41TokenSpaceProcessStrategy, DeepseekV41TokenSpaceServingSpec
    )
)


def _tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]")),
        unk_token="[UNK]",
    )


def _config(*, vision=True):
    # The release's vision config, smaller token budget so the test images stay small.
    values = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "image_token_id": IMAGE,
    }
    if vision:
        values["vision_config"] = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "patch_size": 14,
            "rope_theta": 10000,
            "downsample_ratio": 3,
            "max_image_tokens": 128,
            "min_pixels": 20000,
            "max_wh_ratio": None,
        }
    return DeepseekV41Config(**normalize_deepseek_v41_config(values))


def _image(width, height, seed, format):
    pixels = (np.random.RandomState(seed).rand(height, width, 3) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, format=format)
    return buffer.getvalue()


class TestDeepseekV41TokenSpace(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.server_args = ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            mm_process_config={},
            mm_feature_transport="cpu",
            mm_preprocess_cache_size_mb=0,
            disable_fast_image_processor=True,
            mm_processor_worker_num=1,
            enable_token_space_processor=True,
        )
        publish(self.server_args, role="test")
        self.tokenizer = _tokenizer()

    def _served(self, config=None):
        served = DeepseekV41TokenSpaceProcessor(
            config or _config(), self.server_args, self.tokenizer, None
        )
        self.addCleanup(served.shutdown)
        return served

    def test_serving_output_matches_legacy_processor(self):
        images = [
            _image(200, 200, 1, "PNG"),
            _image(400, 120, 2, "JPEG"),
            _image(90, 300, 3, "PNG"),
        ]
        input_ids = [1, IMAGE, 2, IMAGE, IMAGE, 1]
        config = _config()
        legacy = DeepseekV41ImageProcessor(
            config, self.server_args, self.tokenizer, None
        )
        self.addCleanup(legacy.shutdown)
        legacy_output = asyncio.run(
            legacy.process_mm_data_async(
                image_data=images,
                input_text=input_ids,
                request_obj=GenerateReqInput(input_ids=input_ids),
            )
        )
        served_output = asyncio.run(
            self._served(config).process_mm_data_async(
                image_data=images, input_text=input_ids
            )
        )

        self.assertEqual(served_output.input_ids, legacy_output.input_ids)
        self.assertEqual(served_output.im_token_id, legacy_output.im_token_id)
        self.assertEqual(len(served_output.mm_items), len(images))
        self.assertEqual(len(served_output.mm_items), len(legacy_output.mm_items))
        for item, reference in zip(served_output.mm_items, legacy_output.mm_items):
            self.assertEqual(item.offsets, reference.offsets)
            self.assertEqual(item.feature.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(item.feature, reference.feature))
            self.assertEqual(
                (item.n_vit_h, item.n_vit_w), (reference.n_vit_h, reference.n_vit_w)
            )

    def test_transposed_grids_hash_apart(self):
        """A solid image and its transpose give equal patches; only the grid tells their spans apart."""
        solid = Image.new("RGB", (280, 140), (10, 20, 30))
        buffers = []
        for image in (solid, solid.transpose(Image.Transpose.TRANSPOSE)):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffers.append(buffer.getvalue())
        output = asyncio.run(
            self._served().process_mm_data_async(
                image_data=buffers, input_text=[IMAGE, 1, IMAGE]
            )
        )
        wide, tall = output.mm_items
        self.assertEqual(wide.feature.shape, tall.feature.shape)
        self.assertTrue(torch.equal(wide.feature, tall.feature))
        self.assertNotEqual((wide.n_vit_h, wide.n_vit_w), (tall.n_vit_h, tall.n_vit_w))
        self.assertNotEqual(wide.hash, tall.hash)

    def test_text_only_checkpoint_rejects_images(self):
        served = self._served(_config(vision=False))
        with self.assertRaisesRegex(Exception, "no vision tower"):
            asyncio.run(
                served.process_mm_data_async(
                    image_data=[_image(32, 32, 1, "PNG")], input_text=[IMAGE]
                )
            )


if __name__ == "__main__":
    unittest.main()
