import unittest
from asyncio import run
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.kimi_k25 import KimiK2_5VLImageProcessor
from sglang.srt.multimodal.processors.kimi_token_ids import (
    process_kimi_token_ids_mm_data,
)


class FakeMediaProcessor:
    def media_tokens_calculator(self, media):
        image = media["image"]
        if isinstance(image, dict) and "tokens" in image:
            return image["tokens"]
        return 3


class TestKimiVLMTokenIds(unittest.TestCase):
    def _processor(self):
        processor = KimiK2_5VLImageProcessor.__new__(KimiK2_5VLImageProcessor)
        processor.mm_tokens = SimpleNamespace(
            image_token="<|media_pad|>",
            image_token_id=163605,
        )
        processor._processor = SimpleNamespace(media_processor=FakeMediaProcessor())
        async def fast_load_mm_data(**kwargs):
            return SimpleNamespace(images=kwargs["image_data"])

        processor.fast_load_mm_data = fast_load_mm_data
        processor.process_mm_data_calls = []

        def process_mm_data(**kwargs):
            processor.process_mm_data_calls.append(kwargs)
            return {"input_ids": torch.tensor([[999]])}

        processor.process_mm_data = process_mm_data
        return processor

    def test_expands_single_image_placeholder_in_token_space(self):
        processor = self._processor()
        processor.collect_mm_items_from_processor_output = lambda _: [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.empty((3, 2)),
            )
        ]

        output = run(
            process_kimi_token_ids_mm_data(
                processor, [11, 163605, 22], [{"tokens": 3}]
            )
        )

        self.assertEqual(output.input_ids, [11, 163605, 163605, 163605, 22])
        self.assertEqual(output.mm_items[0].offsets, [(1, 3)])
        self.assertEqual(
            processor.process_mm_data_calls[0]["input_text"],
            "<|media_pad|><|media_pad|><|media_pad|>",
        )
        self.assertIsNone(processor.process_mm_data_calls[0]["images"])

    def test_adjacent_image_placeholders_keep_distinct_offsets(self):
        processor = self._processor()
        processor.collect_mm_items_from_processor_output = lambda _: [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=[torch.empty((2, 2)), torch.empty((3, 2))],
            )
        ]

        output = run(
            process_kimi_token_ids_mm_data(
                processor, [163605, 163605], [{"tokens": 2}, {"tokens": 3}]
            )
        )

        self.assertEqual(output.input_ids, [163605] * 5)
        self.assertEqual(len(output.mm_items), 2)
        self.assertEqual(output.mm_items[0].offsets, [(0, 1)])
        self.assertEqual(output.mm_items[1].offsets, [(2, 4)])

    def test_raises_on_placeholder_image_count_mismatch(self):
        processor = self._processor()

        with self.assertRaisesRegex(
            ValueError, "mismatched media placeholders and images"
        ):
            run(
                process_kimi_token_ids_mm_data(
                    processor, [11, 163605, 22], [{"tokens": 3}, {"tokens": 3}]
                )
            )


if __name__ == "__main__":
    unittest.main()
