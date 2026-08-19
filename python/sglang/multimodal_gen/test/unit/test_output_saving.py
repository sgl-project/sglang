import numpy as np
import pytest
import torch
from PIL import Image

import sglang.multimodal_gen.runtime.entrypoints.utils as output_utils
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    MaterializedOutput,
    post_process_sample,
    save_materialized_output,
)


def _rgb_frame() -> np.ndarray:
    return np.array(
        [
            [[0, 32, 255], [64, 128, 192], [255, 224, 16]],
            [[9, 17, 33], [127, 128, 129], [240, 12, 88]],
        ],
        dtype=np.uint8,
    )


@pytest.mark.parametrize("output_compression", [None, 0, 75])
def test_png_output_saving_preserves_pixels(tmp_path, output_compression):
    frame = _rgb_frame()
    output_path = tmp_path / f"sample_{output_compression}.png"

    frames = post_process_sample(
        frame,
        DataType.IMAGE,
        fps=1,
        save_file_path=str(output_path),
        output_compression=output_compression,
    )

    assert output_path.exists()
    np.testing.assert_array_equal(frames[0], frame)
    np.testing.assert_array_equal(np.array(Image.open(output_path)), frame)


@pytest.mark.parametrize(
    ("output_compression", "expected_compress_level"), [(None, 1), (0, 0), (75, 1)]
)
def test_png_output_saving_uses_fast_pillow_path(
    tmp_path, monkeypatch, output_compression, expected_compress_level
):
    frame = _rgb_frame()
    output_path = tmp_path / f"sample_{output_compression}.png"

    def fail_imageio_imwrite(*args, **kwargs):
        raise AssertionError("PNG output should use Pillow's PNG fast path")

    original_save = Image.Image.save
    save_calls = []

    def save_spy(self, fp, format=None, **params):
        save_calls.append((format, params.get("compress_level")))
        return original_save(self, fp, format=format, **params)

    monkeypatch.setattr(output_utils.imageio, "imwrite", fail_imageio_imwrite)
    monkeypatch.setattr(Image.Image, "save", save_spy)

    post_process_sample(
        frame,
        DataType.IMAGE,
        fps=1,
        save_file_path=str(output_path),
        output_compression=output_compression,
    )

    assert save_calls == [("PNG", expected_compress_level)]


def test_video_with_audio_uses_single_pass_encoder(tmp_path, monkeypatch):
    output_path = tmp_path / "sample.mp4"
    calls = []

    class FakeWavFile:
        @staticmethod
        def write(*_args, **_kwargs):
            pass

    def mimsave_spy(path, frames, **kwargs):
        calls.append((path, frames, kwargs))
        assert kwargs["audio_path"].endswith(".wav")
        assert kwargs["audio_codec"] == "aac"

    def fail_legacy_mux(**_kwargs):
        raise AssertionError("the two-pass mux path should not run")

    monkeypatch.setattr(output_utils.imageio, "mimsave", mimsave_spy)
    monkeypatch.setattr(output_utils, "scipy_wavfile", FakeWavFile)
    monkeypatch.setattr(output_utils, "_maybe_mux_audio_into_mp4", fail_legacy_mux)

    materialized = MaterializedOutput(
        sample=None,
        frames=[_rgb_frame()],
        audio=np.zeros((320, 2), dtype=np.float32),
        fps=24,
    )
    save_materialized_output(
        materialized,
        DataType.VIDEO,
        str(output_path),
        audio_sample_rate=32000,
    )

    assert len(calls) == 1


def test_video_audio_single_pass_failure_falls_back(tmp_path, monkeypatch):
    output_path = tmp_path / "sample.mp4"
    calls = []
    mux_calls = []

    class FakeWavFile:
        @staticmethod
        def write(*_args, **_kwargs):
            pass

    def mimsave_spy(path, frames, **kwargs):
        calls.append((path, frames, kwargs))
        if "audio_path" in kwargs:
            raise RuntimeError("unsupported audio input")

    monkeypatch.setattr(output_utils.imageio, "mimsave", mimsave_spy)
    monkeypatch.setattr(output_utils, "scipy_wavfile", FakeWavFile)
    monkeypatch.setattr(
        output_utils,
        "_maybe_mux_audio_into_mp4",
        lambda **kwargs: mux_calls.append(kwargs),
    )

    materialized = MaterializedOutput(
        sample=None,
        frames=[_rgb_frame()],
        audio=np.zeros((320, 2), dtype=np.float32),
        fps=24,
    )
    save_materialized_output(
        materialized,
        DataType.VIDEO,
        str(output_path),
        audio_sample_rate=32000,
    )

    assert len(calls) == 2
    assert "audio_path" in calls[0][2]
    assert "audio_path" not in calls[1][2]
    assert len(mux_calls) == 1


@pytest.mark.parametrize(
    ("height", "available_cpus", "expected_threads"),
    [
        (768, 256, 24),
        (720, 256, 22),
        (2160, 16, 24),
        (4320, 256, 128),
        (16, 1, 1),
    ],
)
def test_x264_auto_thread_count(monkeypatch, height, available_cpus, expected_threads):
    monkeypatch.setattr(
        output_utils.os,
        "sched_getaffinity",
        lambda _pid: set(range(available_cpus)),
    )

    assert output_utils._x264_auto_thread_count(height) == expected_threads


def test_video_direct_save_short_circuits_materialization(tmp_path, monkeypatch):
    output_path = tmp_path / "sample.mp4"
    direct_calls = []

    monkeypatch.setattr(
        output_utils,
        "_try_save_cuda_video_direct",
        lambda **kwargs: direct_calls.append(kwargs) or True,
    )
    monkeypatch.setattr(
        output_utils,
        "post_process_sample",
        lambda *_args, **_kwargs: pytest.fail(
            "successful direct save should skip frame materialization"
        ),
    )

    paths = output_utils.save_outputs(
        [torch.zeros((3, 1, 2, 3))],
        DataType.VIDEO,
        fps=24,
        save_output=True,
        build_output_path=lambda _idx: str(output_path),
    )

    assert paths == [str(output_path)]
    assert len(direct_calls) == 1


def test_multiple_videos_use_parallel_direct_save_with_serial_fallback(
    tmp_path, monkeypatch
):
    outputs = [torch.zeros((3, 1, 2, 3)), torch.ones((3, 1, 2, 3))]
    direct_calls = []

    def parallel_save(samples, paths, **kwargs):
        direct_calls.append((samples, paths, kwargs))
        return [True, False]

    serial_calls = []

    monkeypatch.setattr(output_utils, "_try_save_cuda_videos_direct", parallel_save)
    monkeypatch.setattr(
        output_utils,
        "_try_save_cuda_video_direct",
        lambda **kwargs: serial_calls.append(kwargs) or True,
    )
    monkeypatch.setattr(
        output_utils,
        "post_process_sample",
        lambda *_args, **_kwargs: pytest.fail(
            "successful parallel direct saves should skip frame materialization"
        ),
    )

    paths = output_utils.save_outputs(
        outputs,
        DataType.VIDEO,
        fps=24,
        save_output=True,
        build_output_path=lambda idx: str(tmp_path / f"sample_{idx}.mp4"),
    )

    assert paths == [str(tmp_path / "sample_0.mp4"), str(tmp_path / "sample_1.mp4")]
    assert len(direct_calls) == 1
    samples, save_paths, kwargs = direct_calls[0]
    assert all(actual is expected for actual, expected in zip(samples, outputs))
    assert save_paths == paths
    assert kwargs["fps"] == 24
    assert len(serial_calls) == 1
    assert serial_calls[0]["save_file_path"] == paths[1]
