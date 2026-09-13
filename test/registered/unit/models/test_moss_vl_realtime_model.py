from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.models.moss_vl import MossVLForConditionalGeneration
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _model():
    model = MossVLForConditionalGeneration.__new__(MossVLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.spatial_merge_size = 2
    model.vision_seq_pad_multiple = 1
    return model


def _item(grid_thw, *, feature=None, pad_value=99):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        pad_value=pad_value,
        feature=feature,
        model_specific_data={"grid_thw": torch.tensor(grid_thw)},
    )


def _incremental_inputs(*, feature=None):
    old_item = _item([1, 4, 4])
    new_item = _item([1, 4, 6], feature=feature)
    return MultimodalInputs(
        mm_items=[old_item, new_item],
        num_image_tokens=12,
        incremental_encoder_cache=True,
        encoder_cached_len=5,
        encoder_append_len=7,
        vision_position_ids=torch.arange(21).reshape(3, 7),
        visible_frame_counts=torch.tensor([0, 1, 1, 1, 2, 2], dtype=torch.int32),
    )


def test_realtime_encoder_length_includes_historical_items():
    model = _model()
    mm_inputs = _incremental_inputs()
    mm_inputs.mm_items[1].pad_value = 101

    assert model._get_encoder_len(mm_inputs) == 12
    assert (
        model._build_encoder_prefix_pad_ids(mm_inputs).tolist() == [99] * 5 + [101] * 7
    )


def test_non_realtime_multi_item_input_preserves_first_item_behavior():
    model = _model()
    first_pixels = torch.arange(16).reshape(4, 4)
    second_pixels = torch.arange(24).reshape(6, 4)
    mm_inputs = MultimodalInputs(
        mm_items=[
            _item([1, 4, 4], feature=first_pixels, pad_value=99),
            _item([1, 4, 6], feature=second_pixels, pad_value=101),
        ],
        incremental_encoder_cache=False,
        vision_position_ids=torch.arange(15).reshape(3, 5),
        visible_frame_counts=torch.tensor([0, 1, 1], dtype=torch.int32),
    )

    assert model._get_encoder_len(mm_inputs) == 5
    assert model._build_encoder_prefix_pad_ids(mm_inputs).tolist() == [99] * 5

    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        encoder_cached=[False],
        encoder_lens_cpu=[5],
        mm_inputs=[mm_inputs],
    )
    pixel_values, grid_thw, vision_positions = model._collect_mm_data(forward_batch)

    assert torch.equal(pixel_values, first_pixels)
    assert torch.equal(grid_thw, torch.tensor([[1, 4, 4]]))
    assert torch.equal(vision_positions, mm_inputs.vision_position_ids)


def test_realtime_collects_only_new_vision_features():
    model = _model()
    new_pixels = torch.arange(24).reshape(6, 4)
    mm_inputs = _incremental_inputs(feature=new_pixels)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        encoder_cached=[False],
        encoder_lens_cpu=[12],
        mm_inputs=[mm_inputs],
    )

    pixel_values, grid_thw, vision_positions = model._collect_mm_data(forward_batch)

    assert torch.equal(pixel_values, new_pixels)
    assert torch.equal(grid_thw, torch.tensor([[1, 4, 6]]))
    assert torch.equal(vision_positions, mm_inputs.vision_position_ids)


def test_realtime_mask_uses_all_frames_and_keeps_visibility_history():
    model = _model()
    mm_inputs = _incremental_inputs()
    forward_mode = SimpleNamespace(is_decode=lambda: False)
    forward_batch = SimpleNamespace(
        batch_size=1,
        forward_mode=forward_mode,
        encoder_lens=torch.tensor([12]),
        encoder_lens_cpu=[12],
        seq_lens=torch.tensor([6]),
        extend_seq_lens=torch.tensor([6]),
        extend_seq_lens_cpu=[6],
        extend_prefix_lens_cpu=[0],
        mm_inputs=[mm_inputs],
    )

    mask = model._build_cross_attention_custom_mask(forward_batch).reshape(6, 12)
    expected = torch.zeros((6, 12), dtype=torch.uint8)
    expected[1:4, :5] = 1
    expected[4:, :] = 1
    assert torch.equal(mask, expected)

    text_row_mask = model.get_full_text_row_masked_out_mask(forward_batch)
    assert torch.equal(
        text_row_mask.flatten(),
        torch.tensor([False, True, True, True, True, True]),
    )
    assert mm_inputs.visible_frame_counts.shape == (6,)


def test_retracted_realtime_request_extends_metadata_once():
    req = Req(
        rid="realtime",
        origin_input_text=None,
        origin_input_ids=array("q", [90, 91, 92, 93, 94, 7, 99, 8]),
        origin_input_ids_unpadded=array("q", [7, 99, 8]),
        sampling_params=SamplingParams(),
    )
    req.output_ids.extend([10, 11])
    req.is_retracted = True
    req.multimodal_inputs = MultimodalInputs(
        mm_items=[_item([1, 4, 4])],
        num_image_tokens=5,
        incremental_encoder_cache=True,
        encoder_cached_len=0,
        encoder_append_len=5,
        mrope_positions=torch.tensor([[0, 3, 4]]).expand(3, -1).clone(),
        visible_frame_counts=torch.tensor([0, 1, 1], dtype=torch.int32),
    )

    req.init_next_round_input()
    req.init_next_round_input()

    assert req.multimodal_inputs.mrope_positions.shape == (3, 5)
    assert torch.equal(
        req.multimodal_inputs.visible_frame_counts,
        torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
    )


def test_realtime_validation_abort_releases_current_features():
    item = _item([1, 4, 4], feature=torch.ones(4, 4))
    req = Req(
        rid="realtime",
        origin_input_text=None,
        origin_input_ids=array("q", [7, 99, 8]),
        sampling_params=SamplingParams(),
        session=SimpleNamespace(streaming=True),
    )
    req.multimodal_inputs = MultimodalInputs(
        mm_items=[item],
        incremental_encoder_cache=True,
    )

    with get_parallel().override(tp_rank=0):
        req.set_finish_with_abort("invalid request")

    assert item.feature is None
    assert req.multimodal_inputs is None
