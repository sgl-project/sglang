"""CPU regression tests for the GLM-V multimodal processor."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch

from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
)
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGlm4vImageProcessor(CustomTestCase):
    def test_mrope_uses_final_ids_when_processor_mask_is_stale(self):
        self._assert_mrope_uses_final_ids(torch.ones((1, 3), dtype=torch.long))

    def test_mrope_uses_final_ids_without_processor_mask(self):
        self._assert_mrope_uses_final_ids(None)

    def test_mrope_uses_final_ids_when_processor_mask_is_longer(self):
        self._assert_mrope_uses_final_ids(torch.ones((1, 5), dtype=torch.long))

    def _assert_mrope_uses_final_ids(self, attention_mask):
        processor = object.__new__(Glm4vImageProcessor)
        processor._processor = SimpleNamespace(video_processor=None)
        processor.video_config = {}
        processor.hf_config = SimpleNamespace(
            image_token_id=101,
            video_start_token_id=102,
            video_end_token_id=103,
            vision_config=SimpleNamespace(spatial_merge_size=1),
        )
        processor.mm_tokens = SimpleNamespace(image_token_id=101, video_token_id=None)

        processor.load_mm_data = AsyncMock(
            return_value=BaseMultiModalProcessorOutput(input_text="")
        )
        final_input_ids = torch.tensor([7, 101, 101, 8], dtype=torch.long)
        processor.process_and_combine_mm_data_async = AsyncMock(
            return_value=(
                [],
                final_input_ids,
                SimpleNamespace(
                    image_grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.long),
                    video_grid_thw=None,
                    attention_mask=attention_mask,
                ),
            )
        )

        output = asyncio.run(
            processor.process_mm_data_async(
                image_data=[],
                input_text="",
                request_obj=SimpleNamespace(video_data=[]),
            )
        )

        self.assertEqual(output.input_ids, final_input_ids.tolist())
        self.assertEqual(output.mrope_positions.shape, (3, 4))
        self.assertEqual(output.mrope_position_delta.shape, (1, 1))
        expected_positions = torch.tensor(
            [[0, 1, 1, 3], [0, 1, 1, 3], [0, 1, 2, 3]], dtype=torch.long
        )
        torch.testing.assert_close(output.mrope_positions, expected_positions)


if __name__ == "__main__":
    unittest.main()
