from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_schedule as mm_utils
from sglang.srt.managers.mm_utils import embed_mm_inputs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    (
        "prefix_length",
        "extend_length",
        "items_offset_list",
        "expected",
    ),
    [
        ([8], [16], [[(2, 5), (9, 14), (20, 24)]], 10),
        ([30], [0], [[(2, 5), (9, 14), (20, 24)]], 0),
        (
            [4, 0, 10],
            [4, 10, 10],
            [[(2, 5)], [], [(5, 12), (18, 25)]],
            7,
        ),
    ],
)
def test_count_mm_tokens_in_extend(
    prefix_length, extend_length, items_offset_list, expected
):
    input_ids = []
    for prefix, extend, item_offsets in zip(
        prefix_length, extend_length, items_offset_list
    ):
        seq_len = max(
            prefix + extend,
            max((item_end + 1 for _, item_end in item_offsets), default=0),
        )
        req_input_ids = torch.zeros(seq_len, dtype=torch.long)
        for item_start, item_end in item_offsets:
            req_input_ids[item_start : item_end + 1] = 1
        input_ids.append(req_input_ids[prefix : prefix + extend])

    actual = torch.isin(torch.cat(input_ids), torch.tensor([1])).sum().item()
    derived = mm_utils._count_mm_tokens_in_extend(
        prefix_length=prefix_length,
        extend_length=extend_length,
        items_offset_list=items_offset_list,
    )
    assert actual == derived == expected


def test_get_embedding_and_mask_uses_offset_count_without_readback():
    input_ids = torch.zeros(8, dtype=torch.long)
    input_ids[2:5] = 1
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = Mock()
    mask.sum.side_effect = AssertionError("mask count must stay on device")

    with (
        envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False),
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=embedding),
        patch.object(mm_utils, "_get_multimodal_mask", return_value=mask),
    ):
        result, result_mask, result_input_ids = mm_utils.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    mask.sum.assert_not_called()
    assert result is embedding
    assert result_mask is mask
    assert result_input_ids is input_ids


def test_get_embedding_and_mask_async_asserts_offset_count():
    input_ids = torch.zeros(8, dtype=torch.long)
    input_ids[2:5] = 1
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    with (
        envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True),
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=embedding),
        patch.object(mm_utils.torch, "_assert_async") as assert_async,
    ):
        mm_utils.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    assert_async.assert_called_once()
    condition, message = assert_async.call_args.args
    assert condition.item()
    assert "derived from offsets" in message


@pytest.mark.parametrize("shape", [(6, 4), (2, 3, 4), (1, 2, 3, 4)])
def test_adjust_embedding_length_preserves_exact_flattened_rows(shape):
    """Leading encoder batch axes count as tokens, not embedding width."""
    embedding = torch.arange(24, dtype=torch.float32).reshape(shape)

    result = mm_utils._adjust_embedding_length(embedding, 6, Mock())

    assert result is embedding


@pytest.mark.parametrize("shape", [(6, 4), (2, 3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("placeholder_count", [0, 5, 7])
@pytest.mark.parametrize("chunked_prefill_size", [-1, 4])
def test_adjust_embedding_length_rejects_mismatched_flattened_rows(
    shape, placeholder_count, chunked_prefill_size
):
    """Never silently discard encoder rows or accept a shortage at placement."""
    embedding = torch.arange(24, dtype=torch.float32).reshape(shape)
    original = embedding.clone()

    with (
        patch.object(
            mm_utils,
            "get_schedule",
            return_value=Mock(chunked_prefill_size=chunked_prefill_size),
        ),
        pytest.raises(RuntimeError, match="Multimodal embedding length") as error,
    ):
        mm_utils._adjust_embedding_length(embedding, placeholder_count, Mock())

    assert f"num_mm_tokens_in_input_ids={placeholder_count}" in str(error.value)
    assert "num_mm_tokens_in_embedding=6" in str(error.value)
    assert ("Chunked prefill is enabled" in str(error.value)) == (
        chunked_prefill_size != -1
    )
    torch.testing.assert_close(embedding, original, rtol=0, atol=0)


def test_get_embedding_and_mask_falls_back_after_input_ids_rewrite():
    input_ids = torch.zeros(8, dtype=torch.long)
    rewritten_input_ids = input_ids.clone()
    embedding = torch.zeros(2, 4)
    mask_sum = Mock()
    mask_sum.item.return_value = 2
    mask = Mock()
    mask.sum.return_value = mask_sum

    with (
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=None),
        patch.object(
            mm_utils,
            "_get_chunked_prefill_embedding",
            return_value=(embedding, rewritten_input_ids),
        ),
        patch.object(mm_utils, "_get_multimodal_mask", return_value=mask),
    ):
        result, result_mask, result_input_ids = mm_utils.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    mask.sum.assert_called_once_with()
    mask_sum.item.assert_called_once_with()
    assert result is embedding
    assert result_mask is mask
    assert result_input_ids is rewritten_input_ids


@pytest.mark.parametrize(
    ("route", "per_item"),
    [
        ("batched", False),
        ("batched", True),
        ("by_item", False),
        ("by_item", True),
        ("full", False),
        ("full", True),
        ("precomputed", False),
    ],
)
def test_encoder_rows_keep_their_positions_across_prefill_chunks(route, per_item):
    """Flatten batch axes before chunking; cache reuse must retain row order."""
    rows = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    offsets = [(2, 5), (8, 11)]
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=100 + i,
            pad_value=20 + i,
            feature=rows[i * 4 : (i + 1) * 4].reshape(2, 2, 4),
            offsets=[offset],
        )
        for i, offset in enumerate(offsets)
    ]
    if route == "full":
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=100,
                pad_value=20,
                feature=rows.reshape(2, 4, 4),
                offsets=offsets,
            )
        ]
    if route == "precomputed":
        for item in items:
            item.precomputed_embeddings = item.feature

    def encode(encode_items):
        if route == "precomputed":
            raise AssertionError("Precomputed embeddings must skip encoding")
        features = [item.feature for item in encode_items]
        return features if per_item else torch.cat(features)

    input_ids = torch.zeros(14, dtype=torch.long)
    for item in items:
        for start, end in item.offsets:
            input_ids[start : end + 1] = item.pad_value
    text_embedding = torch.nn.Embedding(2, 4)
    expected = text_embedding(torch.zeros_like(input_ids))
    expected[2:6] = rows[:4]
    expected[8:12] = rows[4:]
    override = get_context().override_server_args(tp_size=1, chunked_prefill_size=4)
    override.install()
    try:
        with (
            get_parallel().override(attn_tp_rank=0, attn_tp_size=1, tp_size=1),
            patch.object(mm_utils, "_is_hip", route == "by_item"),
            patch.object(
                mm_utils, "embedding_cache", mm_utils.MultiModalStaticCache(4096)
            ),
        ):
            chunks = []
            for prefix, length in [(0, 4), (4, 4), (8, 6)]:
                chunk, _ = embed_mm_inputs(
                    mm_inputs_list=[MultimodalInputs(mm_items=items)],
                    extend_prefix_lens=[prefix],
                    extend_seq_lens=[length],
                    input_ids=input_ids[prefix : prefix + length].clone(),
                    input_embedding=text_embedding,
                    data_embedding_func_mapping={Modality.IMAGE: encode},
                )
                chunks.append(chunk)
    finally:
        override.restore()

    torch.testing.assert_close(torch.cat(chunks), expected, rtol=0, atol=0)


def test_evs_rewritten_spans_determine_chunk_row_counts():
    """Frame redistribution changes a chunk's row count without changing length."""
    rows = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    original_ids = [0, 20, 20, 0, 20, 20, 0]
    item = MultimodalDataItem(
        modality=Modality.VIDEO,
        hash=100,
        pad_value=20,
        feature=torch.zeros(1),
        offsets=[(1, 2), (4, 5)],
        model_specific_data={"pre_chunked_input_ids": original_ids},
    )
    encoder = Mock(
        return_value=mm_utils.EVSEmbeddingResult(
            embedding=rows, num_tokens_per_frame=[3, 1]
        )
    )
    override = get_context().override_server_args(tp_size=1, chunked_prefill_size=4)
    override.install()
    try:
        with (
            get_parallel().override(attn_tp_rank=0, attn_tp_size=1, tp_size=1),
            patch.object(
                mm_utils, "embedding_cache", mm_utils.MultiModalStaticCache(4096)
            ),
        ):
            for prefix, length, expected_rows, expected_mask in [
                (0, 4, rows[:3], [False, True, True, True]),
                (4, 3, rows[3:], [False, True, False]),
            ]:
                result, mask, _ = mm_utils.get_embedding_and_mask(
                    data_embedding_func=encoder,
                    embedding_items=[item],
                    placeholder_tensor=torch.tensor([20]),
                    input_ids=torch.tensor(original_ids[prefix : prefix + length]),
                    items_size=[0, 1],
                    prefix_length=[prefix],
                    extend_length=[length],
                    items_offset_list=[item.offsets],
                )
                torch.testing.assert_close(result, expected_rows, rtol=0, atol=0)
                assert mask.flatten().tolist() == expected_mask
    finally:
        override.restore()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
