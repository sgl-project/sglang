"""Geometry and visual-prefix regression coverage for Nemotron Parse."""

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.nemotron_parse import (
    NemotronParseVision,
    content_normalized_positions,
)
from sglang.srt.multimodal.processors.nemotron_parse import NemotronParseProcessor
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestParsePositions(CustomTestCase):
    def test_partial_patches_and_padding_use_content_coordinates(self):
        field = torch.tensor([[[0.0], [10.0], [100.0], [110.0]]])
        positions = content_normalized_positions(field, (32, 64), (24, 24), 16)
        expected = torch.tensor(
            [
                [
                    [55 / 3],
                    [80 / 3],
                    [80 / 3],
                    [80 / 3],
                    [305 / 3],
                    [110.0],
                    [110.0],
                    [110.0],
                ]
            ]
        )
        torch.testing.assert_close(positions, expected)

    def test_single_patch_axis_uses_area_average(self):
        field = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
        actual = content_normalized_positions(field, (16, 64), (16, 32), 16)
        expected = torch.tensor([[[6.5], [8.5], [8.5], [8.5]]])
        torch.testing.assert_close(actual, expected)

    def test_complete_grid_keeps_axis_order_and_dtype(self):
        field = torch.arange(8, dtype=torch.bfloat16).reshape(1, 4, 2)
        actual = content_normalized_positions(field, (32, 32), (32, 32), 16)
        torch.testing.assert_close(actual, field)


class TestParseProcessor(CustomTestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.pool.shutdown)
        self.processor = object.__new__(NemotronParseProcessor)
        self.processor.io_executor = self.pool
        self.processor.hf_config = SimpleNamespace(
            image_token_index=1, encoder=SimpleNamespace(patch_size=16)
        )
        self.processor._tokenizer = Mock()
        self.processor._tokenizer.encode.return_value = [2, 0, 123]
        self.processor._processor = Mock()
        self.processor._processor.tokenizer = self.processor._tokenizer
        self.processor._processor.image_processor.return_value = {
            "pixel_values": torch.zeros(8, 768),
            "image_sizes": torch.tensor([[32, 64]]),
            "content_sizes": torch.tensor([[30, 60]]),
        }
        self.processor.load_mm_data = AsyncMock(
            return_value=SimpleNamespace(images=[object()])
        )

    def process(self, prompt):
        return asyncio.run(
            self.processor.process_mm_data_async(
                image_data=["image"], input_text=prompt
            )
        )

    def test_visual_prefix_precedes_unchanged_task_tokens(self):
        result = self.process([2, 0, 123])
        self.assertEqual(result.input_ids, [1, 1, 1, 2, 0, 123])
        item = result.mm_items[0]
        self.assertEqual(item.offsets, [(0, 2)])
        self.assertEqual(item.model_specific_data["content_size"], [30, 60])
        self.processor._tokenizer.encode.assert_not_called()

    def test_string_prompt_does_not_add_special_tokens(self):
        self.process("</s><s><predict_bbox>")
        self.processor._tokenizer.encode.assert_called_once_with(
            "</s><s><predict_bbox>", add_special_tokens=False
        )

    def test_geometry_is_part_of_cache_identity(self):
        first = self.process([2, 0]).mm_items[0]
        self.processor._processor.image_processor.return_value["content_sizes"] = (
            torch.tensor([[24, 64]])
        )
        second = self.process([2, 0]).mm_items[0]
        self.assertNotEqual(first.hash, second.hash)

    def test_rejects_multiple_pages_in_one_request(self):
        with self.assertRaisesRegex(ValueError, "one document image"):
            asyncio.run(
                self.processor.process_mm_data_async(
                    image_data=["a", "b"], input_text=[2, 0]
                )
            )
        self.processor.load_mm_data.assert_not_called()


class TestParseVisionBatch(CustomTestCase):
    def test_sdpa_variable_size_pages_match_separate_encoding(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", mm_attention_backend="sdpa"),
            role="test",
            ranks=SpawnRanks(world_rank=0, gpu_id=0),
        )
        config = SimpleNamespace(
            encoder=SimpleNamespace(
                args={"model": "parse_test", "register_multiple": 2},
                patch_size=2,
                max_resolution=16,
            ),
            decoder=SimpleNamespace(d_model=4),
            vision_summary_dim=16,
        )
        with (
            patch.dict(
                "sglang.srt.configs.radio.VIT_TIMM_DIM_BY_NAME",
                {"parse_test": (8, 1, 2, 16)},
            ),
            get_parallel().override(tp_group=None, attn_tp_group=None),
        ):
            vision = NemotronParseVision(config).eval()
            generator = torch.Generator().manual_seed(123)
            for parameter in vision.parameters():
                with torch.no_grad():
                    parameter.copy_(
                        torch.randn(parameter.shape, generator=generator) * 0.1
                    )
            items = [
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=torch.randn(height * width // 4, 12, generator=generator),
                    model_specific_data={
                        "image_size": [height, width],
                        "content_size": [height - 1, width - 1],
                    },
                )
                for height, width in ((8, 8), (16, 8))
            ]
            with torch.inference_mode():
                actual = vision(items)
                expected = torch.cat([vision([item]) for item in items])
            self.assertEqual(actual.shape, (14, 4))
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
