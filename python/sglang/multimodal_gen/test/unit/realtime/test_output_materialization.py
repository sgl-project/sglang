# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import (
    MAX_FRAME_INTERPOLATION_EXP,
    DataType,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    materialize_output_sample,
    save_outputs,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.postprocess import FrameInterpolator
from sglang.multimodal_gen.runtime.postprocess.rife_interpolator import Model as RIFEModel
from sglang.multimodal_gen.runtime.realtime.video import build_raw_rgb_frame_batches


def test_materialize_output_sample_converts_tensor_to_uint8_frames():
    sample = torch.zeros(3, 1, 2, 2)
    sample[0] = 1.0
    sample[1] = 0.5

    materialized = materialize_output_sample(sample, DataType.VIDEO, fps=24)

    assert materialized.fps == 24
    assert materialized.audio is None
    assert len(materialized.frames) == 1
    frame = materialized.frames[0]
    assert frame.shape == (2, 2, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame[..., 0] == 255)
    assert np.all(frame[..., 1] == 127)
    assert np.all(frame[..., 2] == 0)


def test_save_outputs_can_materialize_without_saving(tmp_path):
    sample = np.full((2, 2, 3), 0.25, dtype=np.float32)
    output_path = tmp_path / "image.png"
    samples_out = []
    frames_out = []

    paths = save_outputs(
        [sample],
        DataType.IMAGE,
        fps=1,
        save_output=False,
        build_output_path=lambda _idx: str(output_path),
        samples_out=samples_out,
        frames_out=frames_out,
    )

    assert paths == [str(output_path)]
    assert not output_path.exists()
    assert samples_out[0] is sample
    assert len(frames_out) == 1
    assert len(frames_out[0]) == 1
    assert frames_out[0][0].dtype == np.uint8
    assert np.all(frames_out[0][0] == 63)


def test_file_path_transport_clears_in_memory_outputs():
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = True
    output_batch = OutputBatch(
        output=[object()],
        audio=torch.zeros(1),
        audio_sample_rate=16000,
    )

    def save_output_paths(batch):
        batch.output_file_paths = ["/tmp/output.png"]

    worker._materialize_file_path_transport(output_batch, save_output_paths)

    assert output_batch.output_file_paths == ["/tmp/output.png"]
    assert output_batch.output is None
    assert output_batch.audio is None
    assert output_batch.audio_sample_rate is None


def test_raw_rgb_frame_batches_convert_batched_video_tensor_to_thwc_bytes():
    output = torch.zeros(1, 3, 2, 2, 2)
    output[0, 0] = 1.0
    output[0, 1] = 0.5
    req = type(
        "Req",
        (),
        {
            "enable_frame_interpolation": False,
            "enable_upscaling": False,
            "request_id": "req",
            "block_idx": 0,
        },
    )()
    output_batch = OutputBatch(audio_sample_rate=None)

    frame_batches, metadata = build_raw_rgb_frame_batches(
        output,
        req,
        output_batch,
        post_process_sample_fn=lambda *args, **kwargs: None,
    )

    assert metadata == {
        "format": "rgb24",
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }
    assert len(frame_batches) == 1
    assert len(frame_batches[0]) == 2
    first = np.frombuffer(frame_batches[0][0], dtype=np.uint8).reshape(2, 2, 3)
    assert np.all(first[..., 0] == 255)
    assert np.all(first[..., 1] == 127)
    assert np.all(first[..., 2] == 0)


def test_raw_rgb_frame_batches_apply_realtime_upscaling(monkeypatch):
    calls = []

    def fake_batch_upscale_frames(frames, *, model_path, scale):
        calls.append((model_path, scale, [frame.shape for frame in frames]))
        return [
            np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
            for frame in frames
        ]

    from sglang.multimodal_gen.runtime import postprocess

    monkeypatch.setattr(postprocess, "batch_upscale_frames", fake_batch_upscale_frames)

    req = type(
        "Req",
        (),
        {
            "data_type": DataType.VIDEO,
            "fps": 24,
            "output_compression": None,
            "enable_frame_interpolation": False,
            "frame_interpolation_exp": 1,
            "frame_interpolation_scale": 1.0,
            "frame_interpolation_model_path": None,
            "enable_upscaling": True,
            "upscaling_model_path": "mock-sr",
            "upscaling_scale": 2,
            "request_id": "req",
            "block_idx": 0,
        },
    )()
    output_batch = OutputBatch(audio_sample_rate=None)

    def post_process_sample(_sample, *_args, **kwargs):
        assert kwargs["enable_upscaling"] is False
        return [np.array([[[1, 2, 3]]], dtype=np.uint8)]

    frame_batches, metadata = build_raw_rgb_frame_batches(
        torch.zeros(1, 3, 1, 1, 1),
        req,
        output_batch,
        post_process_sample,
    )

    assert calls == [("mock-sr", 2, [(1, 1, 3)])]
    assert metadata == {
        "format": "rgb24",
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }
    assert len(frame_batches) == 1
    assert frame_batches[0][0] == bytes([1, 2, 3] * 4)


@pytest.mark.parametrize("exp", [0, MAX_FRAME_INTERPOLATION_EXP + 1, 1.5, 2.0, True])
def test_materialize_output_sample_rejects_out_of_range_interpolation_exp(
    monkeypatch, exp
):
    """Out-of-range or non-int exp must be rejected before loading weights."""

    def fail_load(_self):
        raise AssertionError("RIFE weights must not load for an invalid exp")

    monkeypatch.setattr(FrameInterpolator, "_ensure_model_loaded", fail_load)

    with pytest.raises(ValueError, match="frame_interpolation_exp must be an int in"):
        materialize_output_sample(
            torch.zeros(3, 2, 2, 2),
            DataType.VIDEO,
            fps=24,
            enable_frame_interpolation=True,
            frame_interpolation_exp=exp,
        )


@pytest.mark.parametrize(
    "h, w",
    [
        (480, 832),   # 480p: old pad=480 not divisible by 64 at scale=0.5
        (720, 1280),  # 720p: old pad=736 not divisible by 64 at scale=0.5
    ],
)
def test_rife_inference_scale_half_pads_to_64_boundary(h, w):
    """scale=0.5 requires padding to multiples of 64, not 32.

    Regresses the bug where Model.inference always padded to 32, causing a
    shape mismatch inside IFBlock at 480p and 720p with scale=0.5.
    """
    model = RIFEModel().eval()
    img0 = torch.zeros(1, 3, h, w)
    img1 = torch.zeros(1, 3, h, w)
    out = model.inference(img0, img1, scale=0.5)
    assert out.shape == (1, 3, h, w), f"expected (1,3,{h},{w}), got {out.shape}"
