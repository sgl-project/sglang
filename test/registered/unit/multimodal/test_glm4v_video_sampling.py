import sys

import pytest

from sglang.srt.multimodal.processors.glm4v import (
    _passthrough_video_metadata,
    glm_sample_and_decode_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeVideoReader:
    def __init__(self, total=100, fps=10.0):
        self.avg_fps = fps
        self._total = total

    def __len__(self):
        return self._total

    def get_frames_at(self, indices):
        return [f"frame-{i}" for i in indices]


class _FakeHFVideoProcessor:
    def __init__(self, indices):
        self._indices = indices
        self.calls = []

    def sample_frames(self, metadata, fps=None):
        self.calls.append((metadata, fps))
        return self._indices


def test_sampling_delegates_to_hf_processor():
    processor = _FakeHFVideoProcessor(indices=[0, 5, 10, 15])
    frames, metadata = glm_sample_and_decode_sync(
        _FakeVideoReader(total=100, fps=10.0),
        {"fps": 2.0},
        processor,
    )

    assert frames == ["frame-0", "frame-5", "frame-10", "frame-15"]
    assert metadata["frames_indices"] == [0, 5, 10, 15]
    assert len(processor.calls) == 1
    metadata_arg, fps_arg = processor.calls[0]
    assert metadata_arg.total_num_frames == 100
    assert metadata_arg.fps == 10.0
    assert metadata_arg.duration == 10.0
    assert fps_arg == 2.0


def test_sampling_falls_back_without_hf_sampler():
    frames, metadata = glm_sample_and_decode_sync(
        _FakeVideoReader(total=100, fps=10.0), {"fps": 2.0}, None
    )

    assert frames
    assert metadata["frames_indices"]


def test_sampling_falls_back_when_max_frames_set():
    processor = _FakeHFVideoProcessor(indices=[0])
    glm_sample_and_decode_sync(
        _FakeVideoReader(total=100, fps=10.0), {"max_frames": 4}, processor
    )

    assert processor.calls == []


def test_sampling_rejects_missing_fps():
    with pytest.raises(ValueError, match="fps"):
        glm_sample_and_decode_sync(_FakeVideoReader(fps=None), {}, None)

    with pytest.raises(ValueError, match="fps"):
        glm_sample_and_decode_sync(_FakeVideoReader(fps=0.0), {}, None)


def test_passthrough_metadata_covers_all_frames():
    metadata = _passthrough_video_metadata([0] * 6, {"fps": 3.0})

    assert metadata["total_num_frames"] == 6
    assert metadata["fps"] == 3.0
    assert metadata["duration"] == 2.0
    assert metadata["frames_indices"] == list(range(6))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
