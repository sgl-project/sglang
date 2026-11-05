"""Literal HTML in a Qwen prompt must not consume image attachments."""

import asyncio
import io
import tempfile
import unittest
from types import SimpleNamespace

import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    Qwen2TokenizerFast,
    Qwen2VLConfig,
    Qwen2VLImageProcessorPil,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NATIVE = "<|vision_start|><|image_pad|><|vision_end|>"


def _local_hf_processor():
    # Real tokenizer and image processor, built locally without model downloads.
    tokens = [
        "<|endoftext|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Explain the literal HTML tag <image>. Actual attachment: "],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=tokens,
        ),
    )
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend, additional_special_tokens=tokens[1:]
    )
    config = Qwen2VLConfig(
        architectures=["Qwen2VLForConditionalGeneration"],
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    processor = Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessorPil(
            size={"shortest_edge": 56 * 56, "longest_edge": 112 * 112}
        ),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
    )
    return config, processor


class TestQwenImagePlaceholders(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.wrapped = _local_hf_processor()
        stream = io.BytesIO()
        Image.new("RGB", (56, 56), "red").save(stream, format="PNG")
        cls.image = stream.getvalue()

    def _check_prompt(self, prompt, image_count):
        async def run():
            reference = None
            for mode in ("off", "memory", "shared"):
                with (
                    self.subTest(mode=mode),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    model_path = directory + "/model"
                    self.config.save_pretrained(model_path)
                    args = ServerArgs(
                        model_path=model_path,
                        disable_fast_image_processor=True,
                        mm_feature_transport="cpu",
                        mm_io_worker_num=1,
                        mm_processor_worker_num=1,
                        mm_preprocess_cache_size_mb=0 if mode == "off" else 8,
                        mm_preprocess_cache_dir=(
                            directory + "/cache" if mode == "shared" else None
                        ),
                    )
                    publish(args, role="tokenizer")
                    processor = QwenVLImageProcessor(
                        self.config, args, self.wrapped, None, skip_mm_pool=True
                    )
                    try:
                        request = SimpleNamespace(
                            video_data=None, audio_data=None, mm_content_hashes=None
                        )
                        for attempt in ("cold", "repeat"):
                            with self.subTest(attempt=attempt):
                                output = await processor.process_mm_data_async(
                                    [self.image] * image_count, prompt, request
                                )
                                self.assertEqual(len(output.mm_items), image_count)
                                text = self.wrapped.tokenizer.decode(output.input_ids)
                                # Native image pads expand; ordinary prose is exact.
                                self.assertEqual(
                                    text.split("Actual attachment:")[0],
                                    prompt.split("Actual attachment:")[0],
                                )
                                if reference is None:
                                    reference = output
                                else:
                                    self.assertEqual(
                                        output.input_ids, reference.input_ids
                                    )
                                    self.assertTrue(
                                        torch.equal(
                                            output.mrope_positions,
                                            reference.mrope_positions,
                                        )
                                    )
                                    self.assertTrue(
                                        torch.equal(
                                            output.mrope_position_delta,
                                            reference.mrope_position_delta,
                                        )
                                    )
                                    for item, expected in zip(
                                        output.mm_items, reference.mm_items
                                    ):
                                        self.assertEqual(item.offsets, expected.offsets)
                                        self.assertTrue(
                                            torch.equal(item.feature, expected.feature)
                                        )
                                        self.assertTrue(
                                            torch.equal(
                                                item.image_grid_thw,
                                                expected.image_grid_thw,
                                            )
                                        )
                    finally:
                        processor.shutdown()

        asyncio.run(run())

    def test_literal_html_with_one_attachment(self):
        self._check_prompt(
            "Explain the literal HTML tag <image>. Actual attachment: " + NATIVE, 1
        )

    def test_multiple_literal_tags_with_one_attachment(self):
        self._check_prompt(
            "Explain <image> and <image>. Actual attachment: " + NATIVE, 1
        )

    def test_literal_html_with_two_attachments(self):
        self._check_prompt("Explain <image>. Actual attachment: " + NATIVE + NATIVE, 2)

    def test_native_attachment_control(self):
        self._check_prompt("Actual attachment: " + NATIVE, 1)


if __name__ == "__main__":
    unittest.main()
