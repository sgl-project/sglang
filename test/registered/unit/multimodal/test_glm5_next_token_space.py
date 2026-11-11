"""CPU parity tests for the GLM-5.3 token-space strategy; real processor files, no weights.

One row per test: what it pins, then the test.

    startup  the registry and the serving factory pick the GLM-5.3 strategy and spec
             -> test_registry_and_factory_select_the_glm5_next_strategy
    image    BatchFeature == Glm5NextProcessor(text, images), 1 and 3 images,
             torchvision and PIL backends; M-RoPE == legacy get_rope_index_glm4v
             -> test_batch_feature_matches_hf_processor
    serving  served output == the legacy Glm4vImageProcessor output, from image
             files (one RGBA), 1 and 3 images
             -> test_serving_output_matches_legacy_processor
    chat     turn 2 with an expanded turn-1 prefix (mm_token_expansion_start_len)
             == legacy on the whole conversation text
             -> test_second_turn_partial_expansion_matches_legacy_processor
"""

import asyncio
import functools
import os
import tempfile
import unittest

import msgspec
import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor
from sglang.srt.multimodal.mrope_rope_index import get_rope_index_glm4v
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor
from sglang.srt.multimodal.token_space_mm_process.models.glm5_next import (
    Glm5NextTokenSpaceProcessStrategy,
    Glm5NextTokenSpaceServingSpec,
)
from sglang.srt.multimodal.token_space_mm_process.registry import (
    register_token_space_models,
    serving_spec_for,
    strategy_for,
)
from sglang.srt.multimodal.token_space_mm_process.sglang import (
    TokenSpaceSGLangProcessor,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers import get_config, get_processor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# The processor, tokenizer and config files of GLM-5.3-Flash; no weights.
_REPO = "zai-org/GLM-5.3-Flash"
_REVISION = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
_PROCESSOR_FILES = [
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]
# Cache identity: legacy keys it by artifact or by feature bytes, token space by
# feature bytes scoped by the item's grids, so the routes never compare it.
_CACHE_IDENTITY_ITEM_FIELDS = ("hash", "pad_value", "pad_values", "identity")
_IMAGE = "<|begin_of_image|><|image|><|end_of_image|>"

Glm5NextTokenSpaceProcessor = TokenSpaceSGLangProcessor.for_strategy_and_serving_spec(
    Glm5NextTokenSpaceProcessStrategy, Glm5NextTokenSpaceServingSpec
)


@functools.cache
def _checkpoint():
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            _REPO, revision=_REVISION, allow_patterns=_PROCESSOR_FILES
        )
    except Exception as error:
        raise unittest.SkipTest(f"{_REPO} processor files unavailable: {error}")


def _images(directory):
    """Three image files of different sizes; the last one is RGBA with transparent edges."""
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, (720, 1280, 4), dtype=np.uint8)
    rgba[:40, :, 3] = 0
    arrays = {
        "small.png": rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
        "odd.png": rng.integers(0, 256, (217, 333, 3), dtype=np.uint8),
        "screen.png": rgba,
    }
    paths = []
    for name, array in arrays.items():
        path = os.path.join(directory, name)
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


class TestGlm5NextTokenSpace(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = _checkpoint()
        cls.hf_config = get_config(checkpoint, trust_remote_code=True)
        cls.processors = {
            backend: get_processor(
                checkpoint, trust_remote_code=True, image_processor_backend=backend
            )
            for backend in ("torchvision", "pil")
        }
        if type(cls.processors["torchvision"]).__name__ != "Glm5NextProcessor":
            raise unittest.SkipTest("transformers has no Glm5NextProcessor")
        cls._directory = tempfile.TemporaryDirectory()
        cls.image_paths = _images(cls._directory.name)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

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

    def _routes(self, processor):
        legacy = Glm4vImageProcessor(self.hf_config, self.server_args, processor, None)
        self.addCleanup(legacy.shutdown)
        native = Glm5NextTokenSpaceProcessor(
            self.hf_config, self.server_args, processor, None
        )
        self.addCleanup(native.shutdown)
        return legacy, native

    def _encode(self, processor, text):
        return processor.tokenizer.encode(text, add_special_tokens=False)

    def test_registry_and_factory_select_the_glm5_next_strategy(self):
        register_token_space_models()
        self.assertIs(strategy_for(self.hf_config), Glm5NextTokenSpaceProcessStrategy)
        self.assertIs(serving_spec_for(self.hf_config), Glm5NextTokenSpaceServingSpec)
        processor = self.processors["torchvision"]
        native = get_mm_processor(self.hf_config, self.server_args, processor, None)
        self.addCleanup(native.shutdown)
        self.assertIs(type(native), Glm5NextTokenSpaceProcessor)
        legacy, _ = self._routes(processor)
        # /v1/decisions validates image markers through mm_tokens.
        self.assertEqual(vars(native.mm_tokens), vars(legacy.mm_tokens))

    def test_batch_feature_matches_hf_processor(self):
        images = [Image.open(path) for path in self.image_paths]
        images = [image.convert("RGB") for image in images]
        for backend, processor in self.processors.items():
            for count in (1, 3):
                with self.subTest(backend=backend, images=count):
                    _, native = self._routes(processor)
                    text = f"<|user|>{_IMAGE * count}Describe.<|assistant|>"
                    reference = processor(
                        text=[text],
                        images=images[:count],
                        padding=True,
                        return_tensors="pt",
                    )
                    output = asyncio.run(
                        native._core.process(
                            images[:count],
                            input_ids=self._encode(processor, text),
                            compute_mrope_positions=True,
                        )
                    )
                    for name in (
                        "input_ids",
                        "mm_token_type_ids",
                        "pixel_values",
                        "image_grid_thw",
                    ):
                        self.assert_tensor_bytes_equal(output[name], reference[name])
                    positions, delta = get_rope_index_glm4v(
                        input_ids=reference["input_ids"],
                        hf_config=self.hf_config,
                        image_grid_thw=reference["image_grid_thw"],
                        video_grid_thw=None,
                        attention_mask=reference["attention_mask"],
                    )
                    self.assert_tensor_bytes_equal(
                        output["mrope_positions"], positions.squeeze(1)
                    )
                    self.assert_tensor_bytes_equal(
                        output["mrope_position_delta"], delta
                    )

    def test_serving_output_matches_legacy_processor(self):
        processor = self.processors["torchvision"]
        legacy_processor, native = self._routes(processor)
        for count in (1, 3):
            with self.subTest(images=count):
                text = f"<|user|>{_IMAGE * count}Describe.<|assistant|>"
                paths = self.image_paths[:count]
                legacy = asyncio.run(
                    legacy_processor.process_mm_data_async(
                        image_data=paths,
                        input_text=text,
                        request_obj=GenerateReqInput(text=text, image_data=paths),
                    )
                )
                served = asyncio.run(
                    native.process_mm_data_async(
                        image_data=paths, input_text=self._encode(processor, text)
                    )
                )
                self.assertEqual(len(legacy.mm_items), count)
                self.assert_same_output(served, legacy)

    def test_second_turn_partial_expansion_matches_legacy_processor(self):
        processor = self.processors["torchvision"]
        legacy_processor, native = self._routes(processor)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "What?"}],
            },
            {"role": "assistant", "content": "Noise."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Compare them."},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Turn 1 ends where the second user turn starts.
        split = text.index("<|user|>", text.index("<|user|>") + 1)
        first_ids = self._encode(processor, text[:split])
        second_ids = self._encode(processor, text[split:])
        self.assertEqual(first_ids + second_ids, self._encode(processor, text))
        paths = self.image_paths
        first_turn = asyncio.run(
            native.process_mm_data_async(image_data=paths[:1], input_text=first_ids)
        )
        prefix = first_turn.input_ids
        input_ids = prefix + second_ids
        served = asyncio.run(
            native.process_mm_data_async(
                image_data=paths,
                input_text=input_ids,
                request_obj=GenerateReqInput(
                    input_ids=input_ids,
                    image_data=paths,
                    mm_token_expansion_start_len=len(prefix),
                ),
            )
        )
        legacy = asyncio.run(
            legacy_processor.process_mm_data_async(
                image_data=paths,
                input_text=text,
                request_obj=GenerateReqInput(text=text, image_data=paths),
            )
        )
        self.assertEqual(served.input_ids[: len(prefix)], prefix)
        self.assert_same_output(served, legacy)

    def assert_same_output(self, actual, expected):
        actual, expected = (msgspec.structs.asdict(o) for o in (actual, expected))
        actual_items, expected_items = actual.pop("mm_items"), expected.pop("mm_items")
        # Padded IDs are the final IDs with the cache identity's pad values.
        actual.pop("padded_input_ids"), expected.pop("padded_input_ids")
        self.assert_same_fields(actual, expected, "output")
        self.assertEqual(len(actual_items), len(expected_items))
        for index, (item, reference) in enumerate(zip(actual_items, expected_items)):
            item, reference = (msgspec.structs.asdict(i) for i in (item, reference))
            for name in _CACHE_IDENTITY_ITEM_FIELDS:
                item.pop(name, None), reference.pop(name, None)
            self.assert_same_fields(item, reference, f"mm_items[{index}]")

    def assert_same_fields(self, actual, expected, where):
        self.assertEqual(actual.keys(), expected.keys(), where)
        for name in actual:
            self.assert_same_value(actual[name], expected[name], f"{where}.{name}")

    def assert_same_value(self, actual, expected, where):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor, where)
            self.assert_tensor_bytes_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assert_same_fields(actual, expected, where)
        elif isinstance(expected, (list, tuple)) and any(
            isinstance(v, torch.Tensor) for v in expected
        ):
            self.assertEqual(len(actual), len(expected), where)
            for index, (a, e) in enumerate(zip(actual, expected)):
                self.assert_same_value(a, e, f"{where}[{index}]")
        else:
            self.assertEqual(actual, expected, where)

    def assert_tensor_bytes_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            actual.contiguous().view(torch.uint8).numpy().tobytes()
            == expected.contiguous().view(torch.uint8).numpy().tobytes(),
            "Tensor bytes differ",
        )


if __name__ == "__main__":
    unittest.main()
