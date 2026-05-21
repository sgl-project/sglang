import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FailingTokenizer:
    def __call__(self, *args, **kwargs):
        raise AssertionError("tokenizer should not run for preprocessed inputs")

    def decode(self, *args, **kwargs):
        raise AssertionError("decode should not run for preprocessed inputs")


class DummyProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def make_processor():
    processor = DummyProcessor.__new__(DummyProcessor)
    processor._tokenizer = FailingTokenizer()
    processor.ATTR_NAME_TO_MODALITY = {
        "pixel_values": Modality.IMAGE,
        "image_grid_thw": Modality.IMAGE,
    }
    processor.FEATURE_NAMES = ["pixel_values"]
    return processor


class TestPreprocessedInputFastPath(unittest.TestCase):
    def test_load_mm_data_skips_decode_for_preprocessed_input_ids(self):
        processor = make_processor()
        input_ids = [1, 42, 42, 2]
        processor_output = {
            "format": "processor_output",
            "input_ids": torch.tensor([input_ids]),
            "pixel_values": torch.arange(8).reshape(2, 4),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }

        base_output = processor.load_mm_data(
            prompt=input_ids,
            multimodal_tokens=MultimodalSpecialTokens(image_token_id=42),
            image_data=[processor_output],
        )

        self.assertEqual(base_output.input_ids, input_ids)
        self.assertEqual(base_output.input_text, "")
        self.assertIs(base_output.images[0], processor_output)

    def test_processor_output_uses_original_input_ids_without_tokenizing(self):
        processor = make_processor()
        input_ids = [1, 42, 42, 2]
        processor_output = {
            "format": "processor_output",
            "input_ids": torch.tensor([input_ids]),
            "pixel_values": torch.arange(8).reshape(2, 4),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }
        base_output = BaseMultiModalProcessorOutput(
            input_text="",
            input_ids=input_ids,
            images=[processor_output],
        )

        mm_items, output_ids, ret = processor.process_and_combine_mm_data(
            base_output,
            MultimodalSpecialTokens(image_token_id=42),
        )

        self.assertEqual(output_ids.tolist(), input_ids)
        self.assertIs(ret, processor_output)
        self.assertEqual(len(mm_items), 1)
        self.assertEqual(mm_items[0].offsets, [(1, 2)])
        self.assertIsNotNone(mm_items[0].pad_value)
        self.assertTrue(
            torch.equal(mm_items[0].feature, processor_output["pixel_values"])
        )

    def test_processor_output_multi_image_split_uses_grid_lengths(self):
        processor = make_processor()
        input_ids = [1, 42, 42, 2, 42, 42, 42, 3]
        processor_output = {
            "format": "processor_output",
            "input_ids": torch.tensor([input_ids]),
            "pixel_values": torch.arange(20).reshape(5, 4),
            "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
        }
        base_output = BaseMultiModalProcessorOutput(
            input_text="",
            input_ids=input_ids,
            images=[processor_output],
        )

        mm_items, output_ids, _ = processor.process_and_combine_mm_data(
            base_output,
            MultimodalSpecialTokens(image_token_id=42),
        )

        self.assertEqual(output_ids.tolist(), input_ids)
        self.assertEqual(len(mm_items), 2)
        self.assertEqual(mm_items[0].offsets, [(1, 2)])
        self.assertEqual(mm_items[1].offsets, [(4, 6)])
        self.assertIsNotNone(mm_items[0].pad_value)
        self.assertIsNotNone(mm_items[1].pad_value)
        self.assertEqual(tuple(mm_items[0].feature.shape), (2, 4))
        self.assertEqual(tuple(mm_items[1].feature.shape), (3, 4))
        self.assertEqual(mm_items[0].image_grid_thw.tolist(), [[1, 1, 2]])
        self.assertEqual(mm_items[1].image_grid_thw.tolist(), [[1, 1, 3]])

    def test_precomputed_embedding_input_dict_is_not_mutated(self):
        processor = make_processor()
        input_ids = [1, 42, 42, 2]
        feature = torch.arange(8).reshape(2, 4)
        precomputed_input = {
            "format": "precomputed_embedding",
            "feature": feature,
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }
        base_output = BaseMultiModalProcessorOutput(
            input_text="",
            input_ids=input_ids,
            images=[precomputed_input],
        )

        mm_items, output_ids, _ = processor.process_and_combine_mm_data(
            base_output,
            MultimodalSpecialTokens(image_token_id=42),
        )

        self.assertEqual(output_ids.tolist(), input_ids)
        self.assertIn("feature", precomputed_input)
        self.assertIs(precomputed_input["feature"], feature)
        self.assertEqual(len(mm_items), 1)
        self.assertIs(mm_items[0].feature, feature)
        self.assertIsNotNone(mm_items[0].pad_value)

    def test_qwen_mrope_collects_all_split_image_grids(self):
        processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        processor.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=1, tokens_per_second=None)
        )
        processor.mm_tokens = SimpleNamespace(image_token_id=42, video_token_id=43)
        processor.vision_start_token_id = 11
        processor.model_type = "qwen2_vl"

        mm_items = []
        for offset, grid in (
            ((1, 4), torch.tensor([[1, 2, 2]])),
            ((7, 15), torch.tensor([[1, 3, 3]])),
        ):
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[offset],
                    model_specific_data={"image_grid_thw": grid},
                )
            )

        input_ids = [11, 42, 42, 42, 42, 12, 11] + [42] * 9 + [12]
        mrope_positions, mrope_delta = processor.compute_mrope_positions(
            input_ids, mm_items
        )

        self.assertEqual(tuple(mrope_positions.shape), (3, len(input_ids)))
        self.assertEqual(tuple(mrope_delta.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
