from array import array
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention import torch_native_backend
from sglang.srt.layers.attention.cross_attention_mask import (
    filter_cross_attention_kv_indices,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.mllama import (
    MllamaForConditionalGeneration,
    build_mllama_cross_attention_mask,
)
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
            "mllama_prompt_length": 7,
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
    model.max_num_tiles = 2
    model.image_size = 2
    return model


def test_image_visibility():
    # Meta create_vision_mask/_pad_masks: markers at 1, 2, 5, two tiles per image.
    expected = torch.tensor(
        [
            [False, False, False, False, False, False],
            [True, False, False, False, False, False],
            [True, False, True, True, False, False],
            [True, False, True, True, False, False],
            [True, False, True, True, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, True, False],
        ]
    ).repeat_interleave(2, dim=1)
    for start, rows in (
        (0, expected),
        (3, expected[3:6]),
        (5, torch.cat([expected[5:], torch.zeros_like(expected[:2])])),
    ):
        actual = build_mllama_cross_attention_mask(
            image_positions=[1, 2, 5],
            tile_masks=[torch.arange(2) < count for count in [1, 2, 1]],
            num_patches=2,
            query_start=start,
            query_length=len(rows),
            prompt_length=7,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(actual, rows)

    for positions in ([1], [1, 2], [1, 4]):
        actual = build_mllama_cross_attention_mask(
            image_positions=positions,
            tile_masks=[torch.tensor([True, False])] * len(positions),
            num_patches=2,
            query_start=7,
            query_length=1,
            prompt_length=7,
            device=torch.device("cpu"),
        )
        expected_decode = torch.zeros_like(actual)
        if len(positions) == 1:
            expected_decode[:, :2] = True
        torch.testing.assert_close(actual, expected_decode)


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
    assert mm_input.mm_items[0].model_specific_data["mllama_prompt_length"] == len(text)
    assert padded[8:] == text
    longer = model.pad_input_ids(text + array("q", [3]), mm_input)
    assert longer[0] != padded[0]
    assert model.pad_input_ids(text, mm_input) == padded
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
    assert model.get_full_text_row_masked_out_mask(batch).flatten().tolist() == [
        True,
        False,
        False,
    ]
    indices, indptr = filter_cross_attention_kv_indices(
        torch.arange(12, dtype=torch.int32),
        torch.tensor([0, 4, 4, 12], dtype=torch.int32),
        batch.cross_attention_custom_mask,
    )
    # The inactive multi-image row uses finite attention before residual gating.
    assert indices.tolist() == [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
    assert indptr.tolist() == [0, 2, 2, 10]

    # Recomputed generated tokens retain the original prompt boundary.
    batch.forward_mode = ForwardMode.EXTEND
    batch.extend_prefix_lens_cpu = [7, 7, 7]
    batch.extend_seq_lens_cpu = [2, 2, 2]
    model.prepare_forward_batch(batch)
    assert model.get_full_text_row_masked_out_mask(batch).flatten().tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("masked", [False, True])
def test_torch_native_cross_attention_with_cached_prefix(masked):
    query = torch.ones(2, 1, 2)
    keys = torch.ones(4, 1, 2)
    values = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    mask = torch.tensor([[True, False, False, False], [False, False, False, True]])
    backend = object.__new__(torch_native_backend.TorchNativeAttnBackend)

    with patch.object(
        torch_native_backend,
        "scaled_dot_product_attention",
        wraps=torch.nn.functional.scaled_dot_product_attention,
    ) as sdpa:
        actual = backend._run_sdpa_forward_extend(
            query=query,
            output=torch.empty_like(query),
            k_cache=keys,
            v_cache=values,
            req_to_token=torch.arange(4).view(1, 4),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([5]),
            extend_prefix_lens=torch.tensor([3]),
            extend_seq_lens=torch.tensor([2]),
            encoder_lens=torch.tensor([4]),
            is_cross_attn=True,
            cross_attention_custom_mask=mask.flatten() if masked else None,
        )

    # Preserve the full query layout for callers using the existing unmasked path.
    assert sdpa.call_args.args[0].shape[-2] == (2 if masked else 5)
    expected = values[[0, 3]] if masked else values.mean(0).expand_as(query)
    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
