import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.multimodal.processors.moss_vl import (
    MossVLImageProcessor,
    _patch_moss_vl_image_processor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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
    processor.vision_seq_pad_multiple = 1
    processor.hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        vision_seq_pad_multiple=1,
    )
    processor.server_args = SimpleNamespace(enable_streaming_session=True)
    return processor


def _mm_item(grid_thw):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        model_specific_data={"grid_thw": torch.tensor(grid_thw)},
    )


def test_moss_vl_image_processor_maps_transformers_v5_resample_keyword():
    class ReleasedMossVLImageProcessor:
        def _preprocess(self, images, interpolation, **kwargs):
            return images, interpolation, kwargs

        def resize(self, image, size, resample=None, **kwargs):
            return image, size, resample, kwargs

    processor = ReleasedMossVLImageProcessor()
    _patch_moss_vl_image_processor(processor)
    _patch_moss_vl_image_processor(processor)

    assert processor._preprocess(["image"], resample="bilinear") == (
        ["image"],
        "bilinear",
        {},
    )
    assert processor.resize("image", "size", interpolation="bilinear") == (
        "image",
        "size",
        "bilinear",
        {},
    )


def test_moss_vl_image_processor_keeps_current_resample_signature():
    class CurrentMossVLImageProcessor:
        def _preprocess(self, images, resample=None, **kwargs):
            return images, resample, kwargs

        def resize(self, image, size, resample=None, **kwargs):
            return image, size, resample, kwargs

    original_preprocess = CurrentMossVLImageProcessor._preprocess
    processor = CurrentMossVLImageProcessor()

    _patch_moss_vl_image_processor(processor)

    assert CurrentMossVLImageProcessor._preprocess is original_preprocess
    assert processor._preprocess(["image"], resample="bilinear") == (
        ["image"],
        "bilinear",
        {},
    )


def test_moss_vl_realtime_metadata_matches_pinned_hf_oracle():
    processor = _processor()

    (
        positions,
        rope_delta,
        full_vision_positions,
        new_vision_positions,
        visible_frame_counts,
    ) = processor.compute_realtime_metadata(
        input_ids=[7, 99, 8, 9, 99, 10],
        historical_mm_items=[_mm_item([1, 4, 4])],
        new_mm_items=[_mm_item([1, 4, 6])],
    )

    expected_positions = torch.tensor(
        [
            [0, 3, 4, 5, 9, 10],
            [0, 3, 4, 5, 9, 10],
            [0, 3, 4, 5, 9, 10],
        ]
    )
    expected_full_vision_positions = torch.tensor(
        [
            [1, 1, 1, 1, 3, 6, 6, 6, 6, 6, 6, 9],
            [1, 1, 2, 2, 3, 6, 6, 6, 7, 7, 7, 9],
            [1, 2, 1, 2, 3, 6, 7, 8, 6, 7, 8, 9],
        ]
    )

    assert torch.equal(positions, expected_positions)
    assert torch.equal(rope_delta, torch.tensor([[5]]))
    assert torch.equal(full_vision_positions, expected_full_vision_positions)
    assert torch.equal(new_vision_positions, expected_full_vision_positions[:, 5:])
    assert torch.equal(
        visible_frame_counts, torch.tensor([0, 1, 1, 1, 2, 2], dtype=torch.int32)
    )


def test_moss_vl_realtime_text_only_metadata_has_zero_visibility():
    processor = _processor()

    (
        positions,
        rope_delta,
        full_vision_positions,
        new_vision_positions,
        visible_frame_counts,
    ) = processor.compute_realtime_metadata(
        input_ids=[7, 8, 9],
        historical_mm_items=[],
        new_mm_items=[],
    )

    assert torch.equal(positions, torch.tensor([[0, 1, 2]]).expand(3, -1))
    assert torch.equal(rope_delta, torch.tensor([[0]]))
    assert full_vision_positions is None
    assert new_vision_positions is None
    assert torch.equal(visible_frame_counts, torch.tensor([0, 0, 0], dtype=torch.int32))


def test_moss_vl_realtime_text_only_turn_reuses_historical_frame_metadata():
    processor = _processor()

    (
        positions,
        rope_delta,
        full_vision_positions,
        new_vision_positions,
        visible_frame_counts,
    ) = processor.compute_realtime_metadata(
        input_ids=[7, 99, 8, 9, 10],
        historical_mm_items=[_mm_item([1, 4, 4])],
        new_mm_items=[],
    )

    assert torch.equal(positions, torch.tensor([[0, 3, 4, 5, 6]]).expand(3, -1))
    assert torch.equal(rope_delta, torch.tensor([[2]]))
    assert full_vision_positions.shape == (3, 5)
    assert new_vision_positions is None
    assert torch.equal(
        visible_frame_counts, torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    (
        "image_data",
        "video_data",
        "input_ids",
        "grid_thw",
        "item_count",
        "expected",
    ),
    [
        (None, None, [7, 8], [[1, 4, 4]], 1, True),
        (["image"], None, [7, 8], [[1, 4, 4]], 1, False),
        (None, ["video"], [7, 8], [[1, 4, 4]], 1, False),
        (None, None, [7, 99, 8], [[1, 4, 4]], 1, False),
        (None, None, [7, 8], [[2, 4, 4]], 1, False),
        (None, None, [7, 8], [[1, 4]], 1, False),
        (None, None, [7, 8], None, 1, False),
        (None, None, [7, 8], [[1, 4, 4]], 0, False),
        (None, None, [7, 8], [[1, 4, 4]], 2, False),
    ],
)
def test_moss_vl_text_only_placeholder_detection(
    image_data,
    video_data,
    input_ids,
    grid_thw,
    item_count,
    expected,
):
    processor = _processor()
    mm_items = [_mm_item([1, 4, 4]) for _ in range(item_count)]
    request = SimpleNamespace(session_params={"id": "session-a"})

    assert (
        processor._is_text_only_placeholder(
            request,
            image_data,
            video_data,
            torch.tensor([input_ids]),
            grid_thw,
            [1],
            torch.zeros(len(input_ids), dtype=torch.int32),
            mm_items,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("vision_seq_pad_multiple", "enable_streaming_session", "session_params"),
    [
        (8, True, {"id": "session-a"}),
        (1, False, {"id": "session-a"}),
        (1, True, None),
        (1, True, {}),
    ],
)
def test_moss_vl_text_only_placeholder_requires_realtime_session_candidate(
    vision_seq_pad_multiple,
    enable_streaming_session,
    session_params,
):
    processor = _processor()
    processor.hf_config.vision_seq_pad_multiple = vision_seq_pad_multiple
    processor.server_args.enable_streaming_session = enable_streaming_session

    assert not processor._is_text_only_placeholder(
        SimpleNamespace(session_params=session_params),
        None,
        None,
        torch.tensor([[7, 8]]),
        [[1, 4, 4]],
        [1],
        torch.zeros(2, dtype=torch.int32),
        [_mm_item([1, 4, 4])],
    )


@pytest.mark.parametrize(
    ("media_nums_per_sample", "visible_frame_counts"),
    [
        (None, torch.zeros(2, dtype=torch.int32)),
        ([2], torch.zeros(2, dtype=torch.int32)),
        ([1], torch.ones(2, dtype=torch.int32)),
        ([1], torch.empty(0, dtype=torch.int32)),
    ],
)
def test_moss_vl_text_only_placeholder_requires_synthetic_output_signature(
    media_nums_per_sample,
    visible_frame_counts,
):
    processor = _processor()

    assert not processor._is_text_only_placeholder(
        SimpleNamespace(session_params={"id": "session-a"}),
        None,
        None,
        torch.tensor([[7, 8]]),
        [[1, 4, 4]],
        media_nums_per_sample,
        visible_frame_counts,
        [_mm_item([1, 4, 4])],
    )


def test_moss_vl_text_only_placeholder_reaches_scheduler_without_vision_positions():
    processor = _processor()
    item = _mm_item([1, 4, 4])
    item.feature = torch.zeros((1, 3, 4, 4))
    processor.image_only_mm_tokens = SimpleNamespace()
    processor._normalize_video_inputs_async = AsyncMock(return_value=([], []))
    processor.load_mm_data = AsyncMock(
        return_value=SimpleNamespace(input_text="prompt-only turn", images=None)
    )
    processor.process_mm_data = MagicMock(
        return_value={
            "input_ids": torch.tensor([[7, 8]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "grid_thw": torch.tensor([[1, 4, 4]]),
            "media_nums_per_sample": [1],
        }
    )
    processor._compute_visible_frame_counts = MagicMock(
        return_value=torch.zeros(2, dtype=torch.int32)
    )
    processor._compute_position_metadata = MagicMock(
        return_value=(
            torch.zeros((3, 1, 2), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
            None,
            [],
        )
    )
    processor._build_mm_items = MagicMock(return_value=[item])
    processor._prepare_mm_items_for_transport = MagicMock(
        side_effect=lambda items: items
    )
    processor._remove_temp_video_paths = MagicMock()

    result = asyncio.run(
        processor.process_mm_data_async(
            image_data=None,
            audio_data=None,
            input_text="prompt-only turn",
            request_obj=SimpleNamespace(
                video_data=None,
                session_params={"id": "session-a"},
            ),
        )
    )

    assert result.input_ids == [7, 8]
    assert result.mm_items[0].model_specific_data["moss_vl_text_only_placeholder"]
    assert processor._compute_position_metadata.call_args.kwargs["grid_thw"] is None


def test_moss_vl_realtime_merge_is_copy_on_write():
    processor = _processor()
    historical_item = _mm_item([1, 4, 4])
    new_item = _mm_item([1, 4, 6])
    previous = MultimodalInputs(
        mm_items=[historical_item],
        im_token_id=99,
        im_start_id=97,
        mrope_positions=torch.tensor([[-1], [-1], [-1]]),
    )
    current = MultimodalInputs(
        mm_items=[new_item],
        mrope_positions=torch.tensor([[-2], [-2], [-2]]),
    )

    merged = processor.merge_realtime_inputs(
        input_ids=[7, 99, 8, 9, 99, 10],
        previous=previous,
        current=current,
    )

    assert merged is not previous
    assert merged is not current
    assert len(merged.mm_items) == 2
    assert merged.mm_items[0] is historical_item
    assert merged.mm_items[1] is new_item
    assert previous.mm_items[0] is historical_item
    assert current.mm_items[0] is new_item
    assert torch.equal(previous.mrope_positions, torch.tensor([[-1], [-1], [-1]]))
    assert torch.equal(current.mrope_positions, torch.tensor([[-2], [-2], [-2]]))
    assert merged.im_token_id == 99
    assert merged.im_start_id == 97
    assert merged.incremental_encoder_cache is True
    assert merged.encoder_cached_len == 5
    assert merged.encoder_append_len == 7
    assert merged.num_image_tokens == 12
    assert merged.media_nums_per_sample == [2]
    assert merged.vision_position_ids.shape == (3, 7)
    assert torch.equal(
        merged.visible_frame_counts,
        torch.tensor([0, 1, 1, 1, 2, 2], dtype=torch.int32),
    )


@pytest.mark.parametrize("item_group", ["historical", "new"])
def test_moss_vl_realtime_rejects_multi_frame_grids(item_group):
    processor = _processor()
    item = _mm_item([2, 4, 4])

    with pytest.raises(ValueError, match="t == 1"):
        processor.compute_realtime_metadata(
            input_ids=[7, 99, 8],
            historical_mm_items=[item] if item_group == "historical" else [],
            new_mm_items=[item] if item_group == "new" else [],
        )


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
