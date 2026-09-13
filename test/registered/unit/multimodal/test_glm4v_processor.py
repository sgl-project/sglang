"""CPU regression tests for the GLM-V multimodal processor."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch

from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
)
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGlm4vImageProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_mrope_uses_final_ids_when_processor_mask_is_stale(self):
        processor = object.__new__(Glm4vImageProcessor)
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
                    attention_mask=torch.ones((1, 3), dtype=torch.long),
                ),
            )
        )

        output = await processor.process_mm_data_async(
            image_data=[],
            input_text="",
            request_obj=SimpleNamespace(video_data=[]),
        )

        self.assertEqual(output.input_ids, final_input_ids.tolist())
        self.assertEqual(output.mrope_positions.shape, (3, 4))
        self.assertEqual(output.mrope_position_delta.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
