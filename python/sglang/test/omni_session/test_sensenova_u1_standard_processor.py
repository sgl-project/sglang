# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest

import torch
from PIL import Image

from sglang.srt.multimodal.processors import sensenova_u1 as processor_module
from sglang.srt.multimodal.processors.sensenova_u1 import (
    SenseNovaU1MultimodalProcessor,
)
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format


class FakeTokenizer:
    token_ids = {
        "<img>": 101,
        "</img>": 102,
        "<IMG_CONTEXT>": 103,
    }

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": torch.tensor([[ord(ch) for ch in text]])}

    def convert_tokens_to_ids(self, token):
        return self.token_ids[token]

    def decode(self, input_ids):
        return "".join(chr(token_id) for token_id in input_ids)


class TestSenseNovaU1Processor(unittest.TestCase):
    def test_image_input_builds_standard_vlm_mm_output(self):
        captured = {}
        original_load = processor_module.load_u1_native_image
        original_build = processor_module.build_u1_vlm_input_ids_and_offsets

        def fake_load(image):
            captured["image_mode"] = image.mode
            return torch.zeros((4, 768), dtype=torch.float32), torch.tensor([[4, 4]])

        def fake_build(*, tokenizer, grid_hw, question):
            captured["tokenizer"] = tokenizer
            captured["grid_hw"] = grid_hw
            captured["question"] = question
            return [10, 103, 103, 11], [(1, 2)], "prompt"

        processor_module.load_u1_native_image = fake_load
        processor_module.build_u1_vlm_input_ids_and_offsets = fake_build
        try:
            processor = SenseNovaU1MultimodalProcessor.__new__(
                SenseNovaU1MultimodalProcessor
            )
            processor._tokenizer = FakeTokenizer()

            output = asyncio.run(
                processor.process_mm_data_async(
                    image_data=[Image.new("RGBA", (2, 2), (0, 0, 0, 0))],
                    input_text=(
                        "<|im_start|>user\n<image>\nWhat color is it?"
                        "<|im_end|>\n<|im_start|>assistant\n"
                    ),
                )
            )
        finally:
            processor_module.load_u1_native_image = original_load
            processor_module.build_u1_vlm_input_ids_and_offsets = original_build

        self.assertEqual([10, 103, 103, 11], output.input_ids)
        self.assertEqual("RGB", captured["image_mode"])
        self.assertEqual("What color is it?", captured["question"])
        self.assertEqual(103, output.im_token_id)
        self.assertEqual([(1, 2)], output.mm_items[0].offsets)
        self.assertTrue(torch.equal(torch.tensor([[4, 4]]), captured["grid_hw"]))

    def test_string_chat_template_keeps_multimodal_payload_for_vlm(self):
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        processed = process_content_for_template_format(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                    {"type": "text", "text": "Describe it"},
                ],
            },
            "string",
            image_data,
            video_data,
            audio_data,
            modalities,
            extract_multimodal_for_string=True,
        )

        self.assertEqual("Describe it", processed["content"])
        self.assertEqual("data:image/png;base64,abc", image_data[0].url)


if __name__ == "__main__":
    unittest.main()
