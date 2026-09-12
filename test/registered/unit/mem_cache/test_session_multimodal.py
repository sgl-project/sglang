"""Session appends preserve precomputed media positions and parent metadata."""

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.rotary_embedding.mrope_rope_index import get_rope_index
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

IMAGE_IDS = [1, 10, 11, 11, 11, 11, 13, 2]
VIDEO_IDS = [1, 10, 12, 12, 13, 2]


def positions(ids, model, media):
    images = [[1, 4, 4] for kind in media if kind == "image"]
    videos = [[2, 2, 2] for kind in media if kind == "video"]
    pos, delta = get_rope_index(
        spatial_merge_size=2,
        image_token_id=11,
        video_token_id=12,
        vision_start_token_id=10,
        model_type=model,
        tokens_per_second=2,
        input_ids=torch.tensor([ids]),
        image_grid_thw=torch.tensor(images) if images else None,
        video_grid_thw=torch.tensor(videos) if videos else None,
        second_per_grid_ts=(
            torch.tensor([0.5] * len(videos)) if model != "qwen3_vl" else None
        ),
        audio_token_id=21,
        audio_start_token_id=20,
        position_id_per_seconds=2,
    )
    return pos.squeeze(1), delta


def multimodal(ids, model, kind):
    pos, delta = positions(ids, model, [kind])
    return MultimodalInputs(
        mm_items=[
            MultimodalDataItem(
                modality=Modality.IMAGE if kind == "image" else Modality.VIDEO
            )
        ],
        mrope_positions=pos,
        mrope_position_delta=delta,
    )


@pytest.mark.parametrize(
    "model", ["qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_omni_moe"]
)
@pytest.mark.parametrize("first", ["text", "image", "video"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("reply", [[], [3, 4]])
def test_appends_match_full_history(model, first, streaming, reply):
    history = {"text": [1, 2], "image": IMAGE_IDS, "video": VIDEO_IDS}[first]
    if first == "video" and model == "qwen3_vl":
        history = [1, 10, 12, 13, 10, 12, 13, 2]
    media = [] if first == "text" else [first]
    saved = multimodal(history, model, first) if media else None
    for turn in range(2):
        history = history + reply + IMAGE_IDS
        media = media + ["image"]
        req = Req(
            str(turn),
            None,
            array("q", history),
            SamplingParams(),
            session=SimpleNamespace(streaming=streaming),
        )
        req.multimodal_inputs = saved
        if streaming:
            # The carried buffer still contains raw markers when padding keeps its length.
            req.full_untruncated_fill_ids = array("q", history)
        req.origin_input_ids = array(
            "q", [1000 if token in (11, 12) else token for token in history]
        )
        old_positions = saved.mrope_positions.clone() if saved else None
        old_items = list(saved.mm_items) if saved else []
        if saved:
            saved.mrope_position_delta_repeated_cache = torch.ones(3, 1)

        req.extend_image_inputs(multimodal(IMAGE_IDS, model, "image"))
        req._refresh_fill_ids()

        expected, delta = positions(history, model, media)
        torch.testing.assert_close(req.multimodal_inputs.mrope_positions, expected)
        torch.testing.assert_close(req.multimodal_inputs.mrope_position_delta, delta)
        assert req.multimodal_inputs.mrope_position_delta_repeated_cache is None
        assert req.full_untruncated_fill_ids == req.origin_input_ids
        if saved:
            # A sibling branch or an aborted streaming turn must retain this parent.
            torch.testing.assert_close(saved.mrope_positions, old_positions)
            assert saved.mm_items == old_items
            assert req.multimodal_inputs is not saved
        saved = req.multimodal_inputs


def test_image_reply_image_positions():
    """Eight image-prompt tokens + two reply tokens + eight new prompt tokens."""
    req = Req(
        "image-reply-image",
        None,
        array("q", IMAGE_IDS + [3, 4] + IMAGE_IDS),
        SamplingParams(),
        session=SimpleNamespace(streaming=False),
    )
    req.multimodal_inputs = multimodal(IMAGE_IDS, "qwen2_5_vl", "image")
    req.extend_image_inputs(multimodal(IMAGE_IDS, "qwen2_5_vl", "image"))

    # The reply occupies positions 6 and 7; the next prompt starts at 8.
    expected = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 12, 13],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 11, 12, 13],
            [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 11, 12, 13],
        ]
    )
    torch.testing.assert_close(req.multimodal_inputs.mrope_positions, expected)
    assert req.multimodal_inputs.mrope_position_delta.item() == -4


def test_missing_positions_allow_scheduler_recompute():
    req = Req(
        "missing-positions",
        None,
        array("q", IMAGE_IDS + IMAGE_IDS),
        SamplingParams(),
        session=SimpleNamespace(streaming=False),
    )
    parent = multimodal(IMAGE_IDS, "qwen2_5_vl", "image")
    parent.mrope_position_delta_repeated_cache = torch.ones(3, 1)
    req.multimodal_inputs = parent
    incoming = multimodal(IMAGE_IDS, "qwen2_5_vl", "image")
    incoming.mrope_positions = incoming.mrope_position_delta = None
    req.extend_image_inputs(incoming)

    # A stale table would make _maybe_compute_mrope_positions skip recomputation.
    assert req.multimodal_inputs.mrope_positions is None
    assert req.multimodal_inputs.mrope_position_delta is None
    assert req.multimodal_inputs.mrope_position_delta_repeated_cache is None
    assert len(req.multimodal_inputs.mm_items) == 2
    assert parent.mrope_positions is not None
    assert parent.mrope_position_delta_repeated_cache is not None
    assert len(parent.mm_items) == 1


def test_non_session_keeps_precomputed_input():
    req = Req("normal", None, array("q", IMAGE_IDS), SamplingParams())
    mm = multimodal(IMAGE_IDS, "qwen2_5_vl", "image")
    original_positions = mm.mrope_positions
    original_fill = req.full_untruncated_fill_ids
    req.extend_image_inputs(mm)
    assert req.multimodal_inputs is mm
    assert mm.mrope_positions is original_positions
    assert req.full_untruncated_fill_ids is original_fill


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
