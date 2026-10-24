import numpy as np
import pytest
import torch
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionMessageContentVideoURL
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors import glm4v as glm4v_module
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.glm4v import (
    Glm4vImageProcessor,
    _merge_glm_video_configs,
    glm_decode_frames_at,
    glm_processor_video_config,
    glm_sample_and_decode_sync,
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


class _Glm5NextLikeVideoProcessor:
    fps = 2
    max_frames = 2048
    max_duration = 0
    max_image_tokens = 16384
    patch_size = 14
    merge_size = 2
    temporal_patch_size = 2


class _Glm4vLikeVideoProcessor:
    """GLM-4.1V/4.6V shape: no max_frames, HF sampler stops at max_duration * fps."""

    fps = 2
    max_duration = 300

    def sample_frames(self, metadata, fps=None):
        duration = min(metadata.duration, self.max_duration)
        count = int(duration * (fps or self.fps))
        return np.linspace(0, metadata.total_num_frames - 1, count, dtype=int)


class _FakeVideoReader:
    avg_fps = 30.0

    def __init__(self, seconds):
        self._num_frames = int(seconds * self.avg_fps)

    def __len__(self):
        return self._num_frames

    def get_frames_at(self, indices):
        return np.zeros((len(indices), 2, 2, 3), dtype=np.uint8)


def _video_defaults(processor, **server_video_config):
    return {**glm_processor_video_config(processor), **server_video_config}


def _merge_one(defaults, request):
    return _merge_glm_video_configs(defaults, [request])[0]


def test_glm_video_merge_does_not_inject_omitted_max_frames():
    """Any max_frames bypasses the HF sampler, so an fps-only request must not gain one."""
    defaults = _video_defaults(_Glm4vLikeVideoProcessor())
    assert _merge_one(defaults, {}) == defaults
    assert "max_frames" not in _merge_one(defaults, {"fps": 1})


@pytest.mark.parametrize(
    "server, request_config, key, expected",
    [
        ({}, {"fps": 60}, "fps", 2.0),
        ({}, {"max_frames": 1_000_000}, "max_frames", 2048),
        ({}, {"max_frames": 10**400}, "max_frames", 2048),
        ({}, {"max_image_tokens": 10**9}, "max_image_tokens", 16384),
        ({"max_frames": 4096}, {"max_frames": 10**6}, "max_frames", 4096),
        (
            {"max_tokens_per_frame": 512},
            {"max_tokens_per_frame": 10**6},
            "max_tokens_per_frame",
            512,
        ),
    ],
)
def test_glm_video_merge_clamps_request_to_deployment_limit(
    server, request_config, key, expected
):
    defaults = _video_defaults(_Glm5NextLikeVideoProcessor(), **server)
    assert _merge_one(defaults, request_config)[key] == expected


def test_glm_video_merge_keeps_tighter_request():
    defaults = _video_defaults(_Glm5NextLikeVideoProcessor())
    request = {
        "fps": 1,
        "max_frames": 8,
        "max_image_tokens": 1000,
        "max_tokens_per_frame": 100,
    }
    merged = _merge_one(defaults, request)
    assert {key: merged[key] for key in request} == request


@pytest.mark.parametrize("server", [{}, {"fps": 3}])
def test_glm4v_request_cannot_sample_more_frames_than_default(server):
    """A request max_frames switches to the fallback sampler; it must not exceed the HF default count."""
    processor = _Glm4vLikeVideoProcessor()
    defaults = _video_defaults(processor, **server)
    video = _FakeVideoReader(seconds=3600)

    def frame_count(request):
        frames, _ = glm_sample_and_decode_sync(
            video, _merge_one(defaults, request), processor
        )
        return len(frames)

    default_count = frame_count({})
    assert frame_count({"fps": 60, "max_frames": 1_000_000}) <= default_count


def _patch_glm_resizers(monkeypatch):
    monkeypatch.setattr(
        glm4v_module,
        "_resize_frames_to_max_tokens",
        lambda frames, max_tokens_per_frame: ("per_frame", max_tokens_per_frame),
    )
    monkeypatch.setattr(
        glm4v_module,
        "_pre_resize_frames_for_processor",
        lambda frames, **budget: ("presize",),
    )


def test_glm_decode_drops_request_per_frame_cap_over_pixel_budget(monkeypatch):
    """A request per-frame cap is kept only while its total pixels fit the shared budget.

    Per-frame resize counts 784 px per token and presize counts 1568, so 16384 tokens
    x 2 frames fits a 16384-token video budget exactly.
    """
    _patch_glm_resizers(monkeypatch)
    defaults = _video_defaults(_Glm5NextLikeVideoProcessor())
    config = _merge_one(defaults, {"max_tokens_per_frame": 16384})
    video = _FakeVideoReader(seconds=1)

    assert glm_decode_frames_at(video, [0, 1], config) == ("per_frame", 16384)
    assert glm_decode_frames_at(video, [0, 1, 2], config) == ("presize",)
    # A DP shard decides on the full clip length, not its own slice.
    assert glm_decode_frames_at(video, [0], config, budget_frame_count=3) == (
        "presize",
    )


def test_glm_decode_keeps_server_per_frame_mode(monkeypatch):
    _patch_glm_resizers(monkeypatch)
    defaults = _video_defaults(_Glm5NextLikeVideoProcessor(), max_tokens_per_frame=512)
    config = _merge_one(defaults, {"max_tokens_per_frame": 100})
    frames = glm_decode_frames_at(_FakeVideoReader(seconds=1), range(200), config)
    assert frames == ("per_frame", 100)


@pytest.mark.parametrize(
    "request_config",
    [
        {"fps": 0},
        {"fps": "fast"},
        {"fps": float("inf")},
        {"fps": float("nan")},
        {"max_frames": -1},
        {"max_frames": float("inf")},
    ],
)
def test_glm_video_merge_rejects_invalid_knobs(request_config):
    (key,) = request_config
    with pytest.raises(ValueError, match=key):
        _merge_glm_video_configs({}, [request_config])


@pytest.mark.parametrize(
    "field, value",
    [("fps", 0), ("fps", float("inf")), ("max_frames", 0), ("max_image_tokens", -1)],
)
def test_video_url_rejects_non_positive_knobs(field, value):
    with pytest.raises(ValidationError):
        ChatCompletionMessageContentVideoURL(url="video.mp4", **{field: value})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
