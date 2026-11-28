import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.multimodal.processors.moss_vl import MossVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _vision_info(frame_count: int):
    return [
        {
            "medias": [
                {
                    "num_frames": frame_count,
                    "grid_h": 4,
                    "grid_w": 4,
                    "start": 0,
                    "vision_tokens_per_frame": 4,
                }
            ]
        }
    ]


def _processor():
    processor = object.__new__(MossVLImageProcessor)
    processor.image_token_id = 99
    processor.hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    return processor


@pytest.mark.parametrize(
    ("input_ids", "frame_count", "expected_counts"),
    [
        ([[1, 99, 2]], 2, "2 frame(s) and 1 token(s)"),
        ([[99, 99]], 1, "1 frame(s) and 2 token(s)"),
        ([[1, 2]], 1, "1 frame(s) and 0 token(s)"),
        ([[1, 99, 2]], 0, "0 frame(s) and 1 token(s)"),
    ],
)
def test_moss_vl_rejects_vision_metadata_token_mismatch(
    input_ids, frame_count, expected_counts
):
    processor = _processor()
    input_ids = torch.tensor(input_ids)
    position_ids = processor._compute_position_ids(input_ids)

    with pytest.raises(ValueError, match=re.escape(expected_counts)):
        processor._compute_vision_position_ids(
            input_ids=input_ids,
            position_ids=position_ids,
            vision_token_info=_vision_info(frame_count),
            max_vision_seq_len=16,
            attention_mask=None,
        )


def test_moss_vl_accepts_matching_vision_metadata_and_tokens():
    processor = _processor()
    input_ids = torch.tensor([[1, 99, 2]])
    position_ids = processor._compute_position_ids(input_ids)

    vision_positions, updated_positions, rope_deltas = (
        processor._compute_vision_position_ids(
            input_ids=input_ids,
            position_ids=position_ids,
            vision_token_info=_vision_info(1),
            max_vision_seq_len=16,
            attention_mask=None,
        )
    )

    assert vision_positions.shape == (3, 1, 16)
    assert updated_positions.shape == position_ids.shape
    assert rope_deltas.shape == (1,)


def test_video_normalization_cleans_sibling_temp_file_on_failure(tmp_path):
    processor = _processor()
    processor.io_executor = ThreadPoolExecutor(max_workers=2)
    temp_path = tmp_path / "normalized.mp4"
    created = threading.Event()

    def normalize(value):
        if value == "good":
            temp_path.write_bytes(b"video")
            created.set()
            return str(temp_path), [str(temp_path)]
        assert created.wait(timeout=5)
        raise ValueError("invalid video")

    processor._normalize_single_video_input = normalize
    try:
        with pytest.raises(ValueError, match="invalid video"):
            asyncio.run(processor._normalize_video_inputs_async(["good", "bad"]))
    finally:
        processor.io_executor.shutdown()

    assert not temp_path.exists()


def test_video_normalization_waits_for_worker_cleanup_when_cancelled(tmp_path):
    processor = _processor()
    processor.io_executor = ThreadPoolExecutor(max_workers=1)
    temp_path = tmp_path / "cancelled.mp4"
    created = threading.Event()
    finish = threading.Event()

    def normalize(_value):
        temp_path.write_bytes(b"video")
        created.set()
        assert finish.wait(timeout=5)
        return str(temp_path), [str(temp_path)]

    processor._normalize_single_video_input = normalize

    async def run():
        task = asyncio.create_task(
            processor._normalize_video_inputs_async(["cancelled"])
        )
        assert await asyncio.to_thread(created.wait, 5)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(run())
    finally:
        finish.set()
        processor.io_executor.shutdown()

    assert not temp_path.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
