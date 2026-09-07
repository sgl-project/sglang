import pytest
import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _processor():
    processor = Glm4vImageProcessor.__new__(Glm4vImageProcessor)
    processor.IM_TOKEN_ID = 99
    processor.VIDEO_START_TOKEN_ID = 101
    processor.VIDEO_END_TOKEN_ID = 102
    return processor


def test_glm4v_partitions_shared_image_and_video_token_offsets():
    processor = _processor()
    mm_tokens = MultimodalSpecialTokens(image_token_id=99, video_token_id=99)
    input_ids = torch.tensor(
        [
            1,
            99,
            99,  # image 1
            2,
            99,
            99,
            99,  # image 2
            3,
            101,  # begin video
            4,
            99,
            99,  # frame 1
            5,
            99,
            99,
            99,  # frame 2
            6,
            102,  # end video
            7,
        ]
    )

    assert processor.get_mm_item_offsets(input_ids, mm_tokens, Modality.IMAGE) == [
        (1, 2),
        (4, 6),
    ]
    assert processor.get_mm_item_offsets(input_ids, mm_tokens, Modality.VIDEO) == [
        (10, 11),
        (13, 15),
    ]


def test_glm4v_keeps_interleaved_media_offsets_in_their_modalities():
    processor = _processor()
    mm_tokens = MultimodalSpecialTokens(image_token_id=99, video_token_id=99)
    input_ids = torch.tensor(
        [
            101,
            99,
            99,
            102,  # video 1
            8,
            99,  # image
            9,
            101,
            99,
            10,
            99,
            102,  # video 2
        ]
    )

    assert processor.get_mm_item_offsets(input_ids, mm_tokens, Modality.IMAGE) == [
        (5, 5)
    ]
    assert processor.get_mm_item_offsets(input_ids, mm_tokens, Modality.VIDEO) == [
        (1, 2),
        (8, 8),
        (10, 10),
    ]


def test_glm4v_uses_default_offsets_when_token_ids_are_distinct():
    processor = _processor()
    mm_tokens = MultimodalSpecialTokens(image_token_id=99, video_token_id=100)
    input_ids = torch.tensor([99, 99, 1, 100, 100])

    assert processor.get_mm_item_offsets(input_ids, mm_tokens, Modality.IMAGE) == [
        (0, 1)
    ]
    assert processor.get_mm_item_offsets(input_ids, mm_tokens, Modality.VIDEO) == [
        (3, 4)
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
