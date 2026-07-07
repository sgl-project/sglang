import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)


def _reference_embedding(
    input_ids,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    org_vocab_mask = (input_ids >= org_vocab_start_index) & (
        input_ids < org_vocab_end_index
    )
    added_vocab_mask = (input_ids >= added_vocab_start_index) & (
        input_ids < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    masked_input = vocab_mask * (input_ids - valid_offset)
    output = F.embedding(masked_input.long(), weight)
    output.masked_fill_((~vocab_mask).unsqueeze(-1), 0)
    return output


def _run_case(input_ids, weight, cfg):
    from sglang.srt.layers.triton_ops.vocab_parallel_embedding import (
        vocab_parallel_embedding,
    )

    expected = _reference_embedding(input_ids, weight, **cfg)
    actual = vocab_parallel_embedding(input_ids, weight, **cfg)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hidden_dim", [7, 128, 6144])
def test_vocab_parallel_embedding_no_added_vocab(dtype, input_dtype, hidden_dim):
    cfg = dict(
        org_vocab_start_index=16,
        org_vocab_end_index=32,
        num_org_vocab_padding=0,
        added_vocab_start_index=64,
        added_vocab_end_index=64,
    )
    weight = torch.randn((16, hidden_dim), dtype=dtype, device="cuda")
    input_ids = torch.tensor([0, 16, 17, 31, 32, 63], dtype=input_dtype, device="cuda")
    _run_case(input_ids, weight, cfg)


def test_vocab_parallel_embedding_added_vocab_with_padding():
    cfg = dict(
        org_vocab_start_index=10,
        org_vocab_end_index=18,
        num_org_vocab_padding=4,
        added_vocab_start_index=100,
        added_vocab_end_index=103,
    )
    weight = torch.randn((16, 257), dtype=torch.bfloat16, device="cuda")
    input_ids = torch.tensor(
        [[9, 10, 17], [18, 100, 102]], dtype=torch.int64, device="cuda"
    )
    _run_case(input_ids, weight, cfg)


def test_vocab_parallel_embedding_empty_input():
    cfg = dict(
        org_vocab_start_index=0,
        org_vocab_end_index=8,
        num_org_vocab_padding=0,
        added_vocab_start_index=8,
        added_vocab_end_index=8,
    )
    weight = torch.randn((8, 64), dtype=torch.bfloat16, device="cuda")
    input_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    _run_case(input_ids, weight, cfg)
