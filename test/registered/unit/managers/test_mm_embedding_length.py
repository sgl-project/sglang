from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_schedule as mm_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu-intel")


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


def test_adjust_embedding_length_crops_overlong_embedding():
    embedding = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    server_args = Mock(chunked_prefill_size=-1)

    with patch.object(mm_utils, "get_schedule", return_value=server_args):
        result = mm_utils._adjust_embedding_length(embedding, 3, Mock())

    torch.testing.assert_close(result, embedding[-3:], rtol=0, atol=0)


def test_adjust_embedding_length_rejects_short_embedding():
    embedding = torch.zeros(2, 4)

    with pytest.raises(RuntimeError, match="Insufficient multimodal embedding length"):
        mm_utils._adjust_embedding_length(embedding, 3, Mock())


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
