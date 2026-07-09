import pytest
import torch

from sglang.srt.layers.rotary_embedding.mrope_rope_index import get_rope_index
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


VISION_START = 10
VISION_END = 11
IMAGE = 12
VIDEO = 20
AUDIO_START = 30
AUDIO_END = 31
AUDIO = 32


def _get_rope_index(input_ids, **kwargs):
    return get_rope_index(
        spatial_merge_size=2,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=VISION_START,
        model_type="qwen2_5_omni",
        input_ids=torch.tensor([input_ids], dtype=torch.long),
        position_id_per_seconds=25,
        audio_token_id=AUDIO,
        audio_start_token_id=AUDIO_START,
        use_audio_in_video=False,
        **kwargs,
    )


def test_qwen25_omni_mrope_supports_interleaved_video_and_two_audios():
    positions, deltas = _get_rope_index(
        [
            100,
            VISION_START,
            VIDEO,
            VIDEO,
            VIDEO,
            VIDEO,
            VISION_END,
            101,
            AUDIO_START,
            AUDIO,
            AUDIO,
            AUDIO_END,
            102,
            AUDIO_START,
            AUDIO,
            AUDIO,
            AUDIO,
            AUDIO_END,
            103,
        ],
        video_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
        second_per_grid_ts=torch.tensor([0.5]),
        audio_seqlens=torch.tensor([7, 11], dtype=torch.long),
    )

    assert positions.shape == (3, 1, 19)
    assert deltas.shape == (1, 1)


def test_qwen25_omni_mrope_supports_audio_only():
    positions, deltas = _get_rope_index(
        [
            100,
            AUDIO_START,
            AUDIO,
            AUDIO,
            AUDIO_END,
            101,
            AUDIO_START,
            AUDIO,
            AUDIO,
            AUDIO,
            AUDIO_END,
            102,
        ],
        audio_seqlens=torch.tensor([7, 11], dtype=torch.long),
    )

    assert positions.shape == (3, 1, 12)
    assert deltas.shape == (1, 1)


def test_qwen25_omni_mrope_reports_audio_count_mismatch():
    with pytest.raises(ValueError, match="audio count mismatch"):
        _get_rope_index(
            [
                AUDIO_START,
                AUDIO,
                AUDIO,
                AUDIO_END,
                1,
                AUDIO_START,
                AUDIO,
                AUDIO,
                AUDIO_END,
            ],
            audio_seqlens=torch.tensor([7], dtype=torch.long),
        )


def test_qwen25_omni_mrope_rejects_audio_in_video_mode():
    with pytest.raises(NotImplementedError, match="separate video/audio"):
        get_rope_index(
            spatial_merge_size=2,
            image_token_id=IMAGE,
            video_token_id=VIDEO,
            vision_start_token_id=VISION_START,
            model_type="qwen2_5_omni",
            input_ids=torch.tensor([[AUDIO_START, AUDIO, AUDIO_END]], dtype=torch.long),
            audio_seqlens=torch.tensor([3], dtype=torch.long),
            audio_token_id=AUDIO,
            audio_start_token_id=AUDIO_START,
            use_audio_in_video=True,
            position_id_per_seconds=25,
        )
