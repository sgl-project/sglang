# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    materialize_output_sample,
    save_outputs,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    build_raw_rgb_frame_batches,
)


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
    worker.rank = 0
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
