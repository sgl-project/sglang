"""CPU tests for InternS1-Pro multimodal processor behavior."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.interns1pro import InternS1_1ImageProcessor


def test_epd_stores_the_image_tensor_in_the_mm_item():
    processor = object.__new__(InternS1_1ImageProcessor)
    processor.build_input_ids = Mock(return_value=([1, 2, 3], [(1, 2)]))
    processor.IM_START_TOKEN_ID = 10
    processor.IM_END_TOKEN_ID = 11
    processor.mm_tokens = SimpleNamespace(
        image_token_id=12,
        video_token_id=13,
        audio_token_id=14,
    )
    image_embedding = torch.zeros(2, 4)

    output = processor.get_validated_mm_data(
        [1, 2, 3],
        {Modality.IMAGE: image_embedding},
        img_grid_thw=torch.tensor([[1, 2, 2]]),
    )

    assert output.mm_items[0].precomputed_embeddings is image_embedding


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
