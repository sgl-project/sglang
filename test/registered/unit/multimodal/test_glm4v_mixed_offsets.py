import pytest
import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.glm4v import (
    Glm4vImageProcessor,
    _merge_glm_video_configs,
    glm_budget_kwargs,
    glm_max_image_tokens_from_configs,
    glm_sample_frame_indices,
)
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


def test_glm_video_merge_leaves_omitted_knobs_unset():
    defaults = {"max_image_tokens": 16384, "_presize_budget": {"max_pixels": 1}}
    merged = _merge_glm_video_configs(defaults, [{}])[0]
    assert merged == defaults
    assert "fps" not in merged
    assert "max_frames" not in merged


def test_glm_video_merge_clamps_attack_sample_count():
    defaults = {"max_image_tokens": 16384}
    attack = _merge_glm_video_configs(
        defaults, [{"fps": 60, "max_frames": 1_000_000}]
    )[0]
    baseline = _merge_glm_video_configs(
        defaults, [{"fps": 2, "max_frames": 2048}]
    )[0]
    assert attack["fps"] == 2.0
    assert attack["max_frames"] == 2048
    attack_indices = glm_sample_frame_indices(
        7200,
        fps=60.0,
        duration=120.0,
        target_fps=attack["fps"],
        max_frame_count=attack["max_frames"],
    )
    baseline_indices = glm_sample_frame_indices(
        7200,
        fps=60.0,
        duration=120.0,
        target_fps=baseline["fps"],
        max_frame_count=baseline["max_frames"],
    )
    assert len(attack_indices) == len(baseline_indices)
    assert len(attack_indices) <= 240


def test_glm_video_merge_keeps_tighter_request():
    defaults = {"max_image_tokens": 16384}
    request = {
        "fps": 1,
        "max_frames": 8,
        "max_image_tokens": 1000,
        "max_tokens_per_frame": 100,
    }
    merged = _merge_glm_video_configs(defaults, [request])[0]
    assert merged["fps"] == 1.0
    assert merged["max_frames"] == 8
    assert merged["max_image_tokens"] == 1000
    assert merged["max_tokens_per_frame"] == 100


def test_glm_video_merge_drops_huge_max_tokens_per_frame():
    defaults = {"max_image_tokens": 16384}
    merged = _merge_glm_video_configs(
        defaults, [{"max_tokens_per_frame": 1_000_000_000}]
    )[0]
    assert "max_tokens_per_frame" not in merged

    defaults_with_cap = {
        "max_image_tokens": 16384,
        "max_tokens_per_frame": 512,
    }
    restored = _merge_glm_video_configs(
        defaults_with_cap, [{"max_tokens_per_frame": 1_000_000_000}]
    )[0]
    assert restored["max_tokens_per_frame"] == 512


def test_glm_video_merge_clamps_max_image_tokens_for_budget():
    class FakeProcessor:
        max_image_tokens = 16384

    defaults = {"max_image_tokens": 16384}
    merged = _merge_glm_video_configs(
        defaults, [{"max_image_tokens": 1_000_000_000}]
    )[0]
    assert merged["max_image_tokens"] == 16384
    assert glm_max_image_tokens_from_configs([merged]) == 16384
    assert glm_budget_kwargs(
        FakeProcessor(),
        user_max_image_tokens=glm_max_image_tokens_from_configs([merged]),
        count=1,
        split=True,
    ) == {"max_image_tokens": 16384}


def test_glm_video_merge_uses_server_max_frames_ceiling():
    lowered = _merge_glm_video_configs(
        {"max_frames": 512}, [{"max_frames": 1_000_000}]
    )[0]
    assert lowered["max_frames"] == 512

    raised = _merge_glm_video_configs(
        {"max_frames": 4096}, [{"max_frames": 1_000_000}]
    )[0]
    assert raised["max_frames"] == 4096


def test_glm_video_merge_rejects_non_positive_knobs():
    with pytest.raises(ValueError, match="fps"):
        _merge_glm_video_configs({}, [{"fps": 0}])
    with pytest.raises(ValueError, match="max_frames"):
        _merge_glm_video_configs({}, [{"max_frames": -1}])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
