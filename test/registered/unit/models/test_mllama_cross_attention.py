from array import array
from types import SimpleNamespace

import pytest
import torch
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)

from sglang.srt.layers.attention.cross_attention_mask import (
    filter_cross_attention_kv_indices,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.models.mllama_utils import build_mllama_cross_attention_mask
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.mlama import MllamaImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _item(positions, tile_counts, value=1):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.full((1, len(positions), 2, 3, 2, 2), float(value)),
        pad_value=value,
        offsets=[(pos, pos) for pos in positions],
        model_specific_data={
            "aspect_ratio_mask": torch.stack(
                [torch.arange(2) < count for count in tile_counts]
            )[None],
            "aspect_ratio_ids": torch.tensor([tile_counts]),
        },
    )


def _model():
    model = object.__new__(MllamaForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.vision_model = SimpleNamespace(num_patches=2)
    model.image_size = 2
    return model


def test_image_visibility_matches_transformers():
    input_ids = [0, 99, 99, 0, 0, 99, 0]
    tile_counts = [1, 2, 1]
    reference = convert_sparse_cross_attention_mask_to_dense(
        [get_cross_attention_token_mask(input_ids, 99)],
        [tile_counts],
        max_num_tiles=2,
        length=len(input_ids),
    )[0]
    expected = torch.from_numpy(reference).flatten(1).repeat_interleave(2, dim=1).bool()

    # Full prefill, cached prefix, and generation share the image visibility rules.
    for start, rows in ((0, expected), (3, expected[3:6]), (7, expected[-1:])):
        actual = build_mllama_cross_attention_mask(
            [1, 2, 5],
            [torch.arange(2) < count for count in tile_counts],
            num_patches=2,
            query_start=start,
            query_length=len(rows),
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(actual, rows)


def test_processor_preserves_adjacent_image_markers():
    processor = object.__new__(MllamaImageProcessor)
    input_ids = [99, 99, 1, 2, 99]
    offsets = processor.get_mm_item_offsets(
        torch.tensor(input_ids),
        MultimodalSpecialTokens(image_token_id=99),
        Modality.IMAGE,
    )
    assert offsets == [(0, 0), (1, 1), (4, 4)]
    assert (
        processor._expand_input_ids(
            input_ids,
            processor.resolve_image_token_counts([None] * 3),
            placeholder_token_id=99,
        )
        == input_ids
    )


def test_multiple_items_contribute_all_encoder_tokens():
    model = _model()
    mm_input = MultimodalInputs(
        mm_items=[_item([1], [1], value=10), _item([2], [2], value=20)]
    )
    text = array("q", [1, 99, 99, 2])
    padded = model.pad_input_ids(text, mm_input)
    assert mm_input.num_image_tokens == 8
    assert padded[8:] == text
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mm_inputs=[mm_input],
        encoder_cached=[False],
        encoder_lens_cpu=[8],
        out_cache_loc=torch.empty(0, dtype=torch.long),
    )
    images, ids, masks, lengths = model._batch_image_inputs(batch)
    assert images[0, :, 0, 0, 0, 0].tolist() == [10, 20]
    assert ids.tolist() == [[1, 2]]
    assert masks.sum(dim=-1).tolist() == [[1, 2]]
    assert lengths == [8]


def test_mixed_batch_masks_with_cached_encoder():
    model = _model()
    mm_inputs = [
        MultimodalInputs(mm_items=[_item([2], [1])]),
        None,
        MultimodalInputs(mm_items=[_item([1, 4], [2, 1])]),
    ]
    for mm_input in (mm_inputs[0], mm_inputs[2]):
        mm_input.mm_items[0].feature = None
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mm_inputs=mm_inputs,
        seq_lens=torch.tensor([4, 5, 7]),
        extend_seq_lens_cpu=[3, 2, 4],
        extend_prefix_lens_cpu=[1, 3, 3],
        encoder_lens_cpu=[4, 0, 8],
        encoder_lens=torch.tensor([4, 0, 8]),
    )
    model.prepare_forward_batch(batch)
    assert model.get_full_text_row_masked_out_mask(batch).flatten().tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert batch.cross_attention_custom_mask[:12].view(3, 4).sum(1).tolist() == [
        4,
        2,
        2,
    ]
    assert batch.cross_attention_custom_mask[12:].view(4, 8).sum(1).tolist() == [
        4,
        2,
        2,
        2,
    ]

    batch.forward_mode = ForwardMode.DECODE
    model.prepare_forward_batch(batch)
    indices, indptr = filter_cross_attention_kv_indices(
        torch.arange(12, dtype=torch.int32),
        torch.tensor([0, 4, 4, 12], dtype=torch.int32),
        batch.cross_attention_custom_mask,
    )
    assert indices.tolist() == [0, 1, 8, 9]
    assert indptr.tolist() == [0, 2, 2, 4]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
