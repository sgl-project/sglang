from dataclasses import dataclass

import pytest

from sglang.srt.utils import sample_video_frames


class DummyVideo:
    def __init__(self, total_frames: int, avg_fps: float):
        self._frames = total_frames
        self._fps = avg_fps

    def __len__(self):
        return self._frames

    def get_avg_fps(self):
        return self._fps


@dataclass(kw_only=True)
class Case:
    frames: int
    avg_fps: float
    desired_fps: int
    max_frames: int
    expected_frames: list[int]
    description: str


# fmt: off
@pytest.mark.parametrize("case", [
    Case(
        frames=100, avg_fps=25.0, desired_fps=5, max_frames=200,
        expected_frames=[0, 5, 10, 15, 20, 26, 31, 36, 41, 46, 52, 57, 62, 67, 72, 78, 83, 88, 93, 99],
        description="capped by desired_fps"
    ),
    Case(
        frames=10, avg_fps=10.0, desired_fps=100, max_frames=5,
        expected_frames=[0, 2, 4, 6, 9],
        description="capped by max_frames"
    ),
    Case(
        frames=50, avg_fps=25.0, desired_fps=50, max_frames=200,
        expected_frames=list(range(50)),
        description="capped by total_frames"
    ),
    Case(
        frames=1, avg_fps=30.0, desired_fps=0, max_frames=0,
        expected_frames=[0],
        description="always sample at least 1 frame"
    )
],     ids=lambda c: c.description)
def test_sample_video_frames_lengths(case: Case):
    video = DummyVideo(case.frames, case.avg_fps)
    result = sample_video_frames(video, desired_fps=case.desired_fps, max_frames=case.max_frames)
    assert result == case.expected_frames

if __name__ == "__main__":
    pytest.main([__file__])
