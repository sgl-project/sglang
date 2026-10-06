# SPDX-License-Identifier: Apache-2.0
"""Streaming input / output helpers of Kandinsky 6 video SR (fps and frame budget).

The reference decodes every frame and then applies ``resample_to_target_fps`` and
``clip_to_aligned_frames``; the port streams.  These tests pin that both give the same
frames, and that the output tensor survives the framework's uint8 conversion.
"""

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import _sample_to_uint8_frames
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.input_stage import (
    Kandinsky6SRInputStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_TILING_SCALE_KEY,
    SR_VIDEO_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.video_io import (
    FrameSelector,
    decode_clip,
    extract_source_audio,
    to_output_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.video_utils import (
    clip_to_aligned_frames,
    resample_to_target_fps,
)

FRAME_COUNTS = [1, 2, 9, 24, 50, 121, 125, 200, 301, 302, 400, 1000]
SOURCE_FPS = [12, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120]


def _stream_select(total: int, fps: float) -> tuple[list[int], int]:
    """Run the streaming selector over frames ``0 .. total - 1`` (early stop included)."""
    selector = FrameSelector(fps)
    kept: list[int] = []
    seen = 0
    for index in range(total):
        if selector.keep(index):
            kept.append(index)
            selector.mark_kept()
        seen += 1
        if selector.enough(seen):
            break
    return kept[: selector.limit(seen, len(kept))], selector.effective_fps


def _reference_select(total: int, fps: float) -> tuple[list[int], int] | None:
    ids = torch.arange(total).view(total, 1, 1, 1)
    resampled, effective_fps = resample_to_target_fps(ids, fps)
    try:
        clipped = clip_to_aligned_frames(resampled)
    except ValueError:
        return None
    return clipped.flatten().tolist(), effective_fps


@pytest.mark.parametrize("fps", SOURCE_FPS)
def test_streaming_selection_equals_reference_resample_and_clip(fps):
    """The stream may stop early once 121 frames are kept, but must return exactly the
    frames (and the effective fps) of decode-all -> resample_to_target_fps ->
    clip_to_aligned_frames, including sources too short for the 121-frame budget and
    ``int(total / step)`` truncation of the last selectable frame."""
    for total in FRAME_COUNTS:
        expected = _reference_select(total, fps)
        kept, effective_fps = _stream_select(total, fps)
        if expected is None:
            # the reference raises "no readable frames"; nothing usable was selected
            assert not kept, (total, fps)
            continue
        ids = torch.tensor(kept).view(-1, 1, 1, 1)
        actual = clip_to_aligned_frames(ids).flatten().tolist()
        assert (actual, effective_fps) == expected, (total, fps)


def test_effective_fps_is_an_int_by_the_reference_rule():
    """Downsampled sources play at exactly 24, in-tolerance / slow sources keep
    ``round(source fps)`` (never a float: SamplingParams.fps must be a positive int)."""
    assert FrameSelector(60.0).effective_fps == 24
    assert FrameSelector(25.0).effective_fps == 25
    assert FrameSelector(23.976).effective_fps == 24
    assert FrameSelector(12.0).effective_fps == 12
    assert all(isinstance(FrameSelector(fps).effective_fps, int) for fps in SOURCE_FPS)


# --------------------------------------------------------------------------- #
# Real files (PyAV)
# --------------------------------------------------------------------------- #
av = pytest.importorskip("av")


def write_test_video(path, *, frames, fps, size=(48, 64), audio_seconds=None):
    """Tiny mp4 whose frame ``i`` is a flat colour ``i`` (so index errors are visible)."""
    from fractions import Fraction

    height, width = size
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        audio = None
        if audio_seconds is not None:
            audio = container.add_stream("aac", rate=22050)
            audio.layout = "mono"
        for index in range(frames):
            gray = (index * 5) % 250
            array = np.full((height, width, 3), gray, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        if audio is not None:
            t = np.arange(int(22050 * audio_seconds)) / 22050
            samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None]
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            frame.sample_rate, frame.pts = 22050, 0
            frame.time_base = Fraction(1, 22050)
            for packet in audio.encode(frame):
                container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)


def _decode_everything(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = [
            torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1)
            for f in container.decode(stream)
        ]
    return torch.stack(frames), fps


@pytest.mark.parametrize("fps,frames", [(30, 40), (24, 26), (12, 20), (60, 80)])
def test_decode_clip_equals_reference_cli_pipeline(tmp_path, fps, frames):
    """decode_clip == read_video_tchw_uint8 -> resample_to_target_fps ->
    clip_to_aligned_frames: same pixels, same frame count (1 + 8k), same output fps."""
    path = tmp_path / "clip.mp4"
    write_test_video(path, frames=frames, fps=fps)
    full, source_fps = _decode_everything(path)
    expected, expected_fps = resample_to_target_fps(full, source_fps)
    expected = clip_to_aligned_frames(expected)

    clip = decode_clip(str(path))

    assert clip.frames.dtype == torch.uint8 and clip.frames.shape[1] == 3
    assert clip.frames.shape[0] % 8 == 1
    assert torch.equal(clip.frames, expected)
    assert clip.fps == expected_fps and isinstance(clip.fps, int)


def test_extract_source_audio_is_mono_float_trimmed_and_optional(tmp_path):
    """Audio is decoded to mono float32 at 44.1 kHz, trimmed to the processed duration
    (the reference muxes the whole track, leaving audio longer than the video), and a
    file without an audio stream yields None instead of a silent track."""
    with_audio = tmp_path / "with_audio.mp4"
    write_test_video(with_audio, frames=10, fps=10, audio_seconds=1.0)
    full = extract_source_audio(str(with_audio))
    trimmed = extract_source_audio(str(with_audio), max_seconds=0.25)
    assert full.dtype == np.float32 and full.ndim == 1
    assert abs(len(full) - 44100) < 2048  # aac priming / resampler delay
    assert len(trimmed) == 11025
    assert np.abs(full).max() <= 1.0 and np.abs(full).max() > 0.05

    silent = tmp_path / "silent.mp4"
    write_test_video(silent, frames=10, fps=10)
    assert extract_source_audio(str(silent)) is None


def test_output_video_round_trips_through_uint8_conversions():
    """Every uint8 value must survive the framework writer's ``(x * 255)`` truncation,
    also when a consumer upcasts the fp16 tensor before multiplying (a plain ``k / 255``
    stored in fp16 fails that order for over half of the values)."""
    values = torch.arange(256, dtype=torch.uint8)
    video = values.view(1, 1, 8, 32).expand(3, 2, 8, 32).contiguous()
    output = to_output_video(video)
    assert output.dtype == torch.float16 and output.shape == (1, 3, 2, 8, 32)

    frames = _sample_to_uint8_frames(output[0])  # framework writer: list of [H, W, 3]
    restored = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)
    assert torch.equal(restored, video)
    assert torch.equal((output.float() * 255).to(torch.uint8)[0], video)

    naive = (video.float() / 255.0).to(torch.float16)
    assert not torch.equal((naive.float() * 255).to(torch.uint8), video)


@pytest.mark.parametrize(
    "scale,size,expected_frame,tiling_scale",
    [
        (2, (48, 64), (48, 64), 2),
        (2.25, (64, 96), (64, 112), 2),
        (4, (48, 64), (48, 64), 4),
    ],
)
def test_input_stage_fills_the_request_from_a_video_file(
    tmp_path, scale, size, expected_frame, tiling_scale
):
    """A 30 fps source is thinned to 24 fps (an int), floored to 1 + 8k frames, and the
    request's output geometry follows the clip: x1.125 pre-upscale aligned to 16 for the
    2.25 scale, the tile scale for the rest.  The audio is mono float32 at 44.1 kHz and
    trimmed to the processed duration."""
    path = tmp_path / "clip.mp4"
    write_test_video(path, frames=40, fps=30, size=size, audio_seconds=1.5)
    params = Kandinsky6SRSamplingParams(video_path=str(path), sr_resolution_scale=scale)
    batch = Req(sampling_params=params)

    Kandinsky6SRInputStage().forward(batch, None)

    video = batch.extra[SR_VIDEO_KEY]
    assert video.dtype == torch.uint8 and video.shape[1] == 3
    assert video.shape[0] == 25 and batch.num_frames == 25  # int(40 / 1.25) = 32 -> 25
    assert tuple(video.shape[2:]) == expected_frame
    assert batch.extra[SR_TILING_SCALE_KEY] == tiling_scale
    assert batch.fps == 24 and isinstance(batch.fps, int)
    assert (batch.height, batch.width) == (
        expected_frame[0] * tiling_scale,
        expected_frame[1] * tiling_scale,
    )
    assert batch.audio.dtype == torch.float32 and batch.audio.ndim == 1
    assert batch.audio.shape[0] == round(25 / 24 * 44100)
    assert batch.audio_sample_rate == 44100


def test_input_stage_produces_no_audio_for_a_silent_source(tmp_path):
    path = tmp_path / "silent.mp4"
    write_test_video(path, frames=9, fps=24)
    batch = Req(sampling_params=Kandinsky6SRSamplingParams(video_path=str(path)))
    Kandinsky6SRInputStage().forward(batch, None)
    assert batch.audio is None and batch.audio_sample_rate is None
    assert batch.fps == 24 and batch.num_frames == 9
