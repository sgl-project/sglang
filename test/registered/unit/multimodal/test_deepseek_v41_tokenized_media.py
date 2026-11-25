"""CPU parity of the DeepSeek-V4.1 token-space strategy with the legacy processor; no weights.

serving   the flag selects this strategy for the checkpoint's architecture; served output
          == legacy PIL backend: every output field, and items, offsets, patches, grids,
          per-item storage, mm_tokens; PNG, JPEG and RGBA, an exact, padded, upscaled,
          stretched and token-capped resize, and two adjacent images; the trainer's output
          holds model inputs and the legacy spans
          -> test_serving_output_matches_legacy_processor
hashes    transposed solid images have equal patches, but their grids and spans differ
          -> test_transposed_images_get_different_hashes
turns     a later turn adds one image behind the expanded prefix
          -> test_partial_expansion_adds_the_new_turns_image
mrope     the scheduler's M-RoPE fallback returns none for two images
          -> test_scheduler_mrope_fallback_takes_two_images
media     video and audio are request errors; a GPU image decoder is rejected
          -> test_video_and_audio_are_rejected, test_gpu_image_decoder_is_rejected
trainer   the strategy module imports without serving modules
          -> test_strategy_imports_without_serving_modules
"""

import asyncio
import io
import subprocess
import sys
import unittest
from unittest.mock import patch

import msgspec
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor
from sglang.srt.multimodal.processors.deepseek_v41 import DeepseekV41ImageProcessor
from sglang.srt.multimodal.token_space_mm_process.models.deepseek_v41 import (
    DeepseekV41TokenSpaceProcessStrategy,
    DeepseekV41TokenSpaceServingSpec,
)
from sglang.srt.multimodal.token_space_mm_process.registry import (
    TOKEN_SPACE_SERVING_SPEC_MAPPING,
    TOKEN_SPACE_STRATEGY_MAPPING,
    register_token_space_models,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# The checkpoint's spelling; DeepseekV41Config normalizes it to DeepseekV4ForCausalLM.
ARCH = "DeepseekV41ForCausalLM"
IMAGE = 129264
VOCAB = {"[UNK]": 0, "hi": 1, "there": 2, "<image>": IMAGE}


def _tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]")),
        unk_token="[UNK]",
    )


def _config():
    # 4 px patches in 8 px token cells keep the grids small; each resize depends on ratios only.
    return DeepseekV41Config(
        architectures=[ARCH],
        vision_n_layers=1,
        image_token_id=IMAGE,
        vision_patch_size=4,
        vision_downsample_ratio=2,
        vision_max_n_token=40,
        vision_min_pixels=256,
        vision_max_wh_ratio=4,
    )


def _image(width, height, seed, format, mode="RGB"):
    pixels = np.random.RandomState(seed).rand(height, width, len(mode)) * 255
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype("uint8"), mode).save(buffer, format=format)
    return buffer.getvalue()


def _solid_png(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 90)).save(buffer, format="PNG")
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
        self.images = [
            _image(16, 16, 1, "PNG"),
            _image(24, 10, 2, "JPEG"),
            _image(20, 20, 3, "PNG", mode="RGBA"),
            _image(8, 8, 4, "PNG"),
            _image(64, 8, 5, "PNG"),
            _image(80, 60, 6, "PNG"),
        ]

    def _processors(self):
        # disable_fast_image_processor pins legacy to PIL, the backend the strategy uses.
        legacy = DeepseekV41ImageProcessor(
            _config(), self.server_args, self.tokenizer, None
        )
        self.addCleanup(legacy.shutdown)
        # Serving's own discovery, undone afterwards so later tests see the registries as before.
        for mapping in (TOKEN_SPACE_STRATEGY_MAPPING, TOKEN_SPACE_SERVING_SPEC_MAPPING):
            registry = patch.dict(mapping)
            registry.start()
            self.addCleanup(registry.stop)
        register_token_space_models()
        served = get_mm_processor(_config(), self.server_args, self.tokenizer, None)
        self.addCleanup(served.shutdown)
        return legacy, served

    def _legacy_output(self, legacy, input_ids, images):
        return asyncio.run(
            legacy.process_mm_data_async(
                image_data=list(images),
                input_text=input_ids,
                request_obj=GenerateReqInput(input_ids=input_ids),
            )
        )

    def assert_same_media(self, served, legacy):
        fields, legacy_fields = (msgspec.structs.asdict(o) for o in (served, legacy))
        # Items are compared one by one below.
        del fields["mm_items"], legacy_fields["mm_items"]
        self.assertEqual(fields, legacy_fields)
        self.assertEqual(len(served.mm_items), len(legacy.mm_items))
        for item, reference in zip(served.mm_items, legacy.mm_items):
            self.assertEqual(item.modality, reference.modality)
            self.assertEqual(item.offsets, reference.offsets)
            self.assertEqual(item.feature.dtype, reference.feature.dtype)
            self.assertTrue(torch.equal(item.feature, reference.feature))
            # Inline IPC pickles a tensor's whole storage, so each item owns its own.
            self.assertEqual(
                item.feature.untyped_storage().nbytes(), item.feature.nbytes
            )
            item, reference = (msgspec.structs.asdict(i) for i in (item, reference))
            self.assertEqual(
                item["model_specific_data"], reference["model_specific_data"]
            )

    def test_serving_output_matches_legacy_processor(self):
        """Exact, padded, RGBA, upscaled, stretched and token-capped images; the last two adjacent."""
        legacy, served = self._processors()
        self.assertIs(type(served._core.strategy), DeepseekV41TokenSpaceProcessStrategy)
        self.assertIs(type(served._serving_spec), DeepseekV41TokenSpaceServingSpec)
        # /v1/decisions validates image markers through mm_tokens.
        self.assertEqual(vars(served.mm_tokens), vars(legacy.mm_tokens))
        input_ids = [1, IMAGE, 2, IMAGE, 1, IMAGE, 2, IMAGE, 1, IMAGE, IMAGE]

        served_output = asyncio.run(
            served.process_mm_data_async(image_data=self.images, input_text=input_ids)
        )
        legacy_output = self._legacy_output(legacy, input_ids, self.images)
        trainer_output = asyncio.run(
            served._core.process(self.images, input_ids=input_ids)
        )

        self.assert_same_media(served_output, legacy_output)
        # A trainer gets model inputs only, and each image's legacy span, half-open.
        self.assertNotIn("num_image_tokens", trainer_output)
        self.assertEqual(
            trainer_output["mm_spans"],
            [
                [(start, end + 1) for start, end in item.offsets]
                for item in legacy_output.mm_items
            ],
        )

    def test_transposed_images_get_different_hashes(self):
        """A solid 32x16 and 16x32 image have equal patches but spans of 12 and 14 tokens."""
        _, served = self._processors()
        output = asyncio.run(
            served.process_mm_data_async(
                image_data=[_solid_png(32, 16), _solid_png(16, 32)],
                input_text=[IMAGE, 1, IMAGE],
            )
        )
        wide, tall = output.mm_items
        self.assertTrue(torch.equal(wide.feature, tall.feature))
        self.assertNotEqual(wide.hash, tall.hash)

    def test_partial_expansion_adds_the_new_turns_image(self):
        """The second turn resends every image; only the placeholder after the prefix expands."""
        legacy, served = self._processors()
        first_turn = asyncio.run(
            served.process_mm_data_async(
                image_data=self.images[:1], input_text=[1, IMAGE, 2]
            )
        )
        second_turn_ids = first_turn.input_ids + [1, IMAGE]
        second_turn = asyncio.run(
            served.process_mm_data_async(
                image_data=self.images[:2],
                input_text=second_turn_ids,
                request_obj=GenerateReqInput(
                    input_ids=second_turn_ids,
                    mm_token_expansion_start_len=len(first_turn.input_ids),
                ),
            )
        )
        legacy_output = self._legacy_output(
            legacy, [1, IMAGE, 2, 1, IMAGE], self.images[:2]
        )
        self.assert_same_media(second_turn, legacy_output)

    def test_scheduler_mrope_fallback_takes_two_images(self):
        """The scheduler asks every request without M-RoPE positions for them;
        a model without M-RoPE must answer none for two images, not raise."""
        _, served = self._processors()
        output = asyncio.run(
            served.process_mm_data_async(
                image_data=self.images[:2], input_text=[IMAGE, 1, IMAGE]
            )
        )
        self.assertEqual(
            served.compute_mrope_positions(output.input_ids, output.mm_items),
            (None, None),
        )

    def test_video_and_audio_are_rejected(self):
        """Legacy reads images only; other media must be a request error, not a server error."""
        _, served = self._processors()
        for media in ({"video_data": [b"\x00"]}, {"audio_data": [b"\x00"]}):
            with (
                self.subTest(media=list(media)),
                self.assertRaisesRegex(ValueError, "image input only"),
            ):
                asyncio.run(
                    served.process_mm_data_async(
                        image_data=self.images[:1], input_text=[1, IMAGE], **media
                    )
                )

    def test_gpu_image_decoder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "image_decoder must be 'pil'"):
            DeepseekV41TokenSpaceProcessStrategy(
                _config(), self.tokenizer, mm_process_config={"image_decoder": "nvjpeg"}
            )

    def test_strategy_imports_without_serving_modules(self):
        """Trainers import the strategy; the serving spec defers its serving imports."""
        loaded = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import sglang.srt.multimodal.token_space_mm_process.models.deepseek_v41\n"
                "print(sorted(m for m in sys.modules if m.startswith(("
                "'sglang.srt.managers', 'sglang.srt.models', 'sglang.srt.server_args', "
                "'sglang.srt.multimodal.processors.base_processor'))))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(loaded, "[]")


if __name__ == "__main__":
    unittest.main()
