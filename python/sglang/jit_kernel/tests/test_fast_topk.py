import pytest
import torch

from sglang.jit_kernel.fast_topk import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)

TOPK = 2048


def _reference_topk(score, lengths, row_starts=None):
    B = score.size(0)
    indices = torch.full((B, TOPK), -1, dtype=torch.int32, device=score.device)
    for i in range(B):
        length = lengths[i].item()
        row_start = 0 if row_starts is None else row_starts[i].item()
        if length <= TOPK:
            indices[i, :length] = torch.arange(length, dtype=torch.int32)
        else:
            vals = score[i, row_start : row_start + length]
            _, top_idx = torch.topk(vals, TOPK)
            indices[i] = top_idx.int()
    return indices


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_fast_topk_short(batch_size):
    device = "cuda"
    input_stride = 4096
    score = torch.randn(batch_size, input_stride, dtype=torch.float32, device=device)
    lengths = torch.full((batch_size,), 1024, dtype=torch.int32, device=device)

    result = fast_topk_v2(score, lengths, TOPK)
    ref = _reference_topk(score, lengths)

    for i in range(batch_size):
        length = lengths[i].item()
        result_set = set(result[i, :length].cpu().tolist())
        ref_set = set(ref[i, :length].cpu().tolist())
        assert result_set == ref_set, f"Row {i}: mismatch"
        assert torch.all(result[i, length:] == -1)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_fast_topk_long(batch_size):
    device = "cuda"
    input_stride = 8192
    score = torch.randn(batch_size, input_stride, dtype=torch.float32, device=device)
    lengths = torch.full((batch_size,), input_stride, dtype=torch.int32, device=device)

    result = fast_topk_v2(score, lengths, TOPK)

    for i in range(batch_size):
        _, ref_idx = torch.topk(score[i], TOPK)
        result_set = set(result[i].cpu().tolist())
        ref_set = set(ref_idx.cpu().tolist())
        assert result_set == ref_set, f"Row {i}: mismatch"


@pytest.mark.parametrize("batch_size", [1, 4])
def test_fast_topk_with_row_starts(batch_size):
    device = "cuda"
    input_stride = 8192
    score = torch.randn(batch_size, input_stride, dtype=torch.float32, device=device)
    lengths = torch.full((batch_size,), 4096, dtype=torch.int32, device=device)
    row_starts = torch.zeros(batch_size, dtype=torch.int32, device=device)

    result = fast_topk_v2(score, lengths, TOPK, row_starts=row_starts)

    for i in range(batch_size):
        length = lengths[i].item()
        _, ref_idx = torch.topk(score[i, :length], TOPK)
        result_set = set(result[i].cpu().tolist())
        ref_set = set(ref_idx.cpu().tolist())
        assert result_set == ref_set, f"Row {i}: mismatch"


@pytest.mark.parametrize("num_requests", [1, 4])
def test_fast_topk_transform_decode(num_requests):
    device = "cuda"
    input_stride = 8192
    B = num_requests  # decode: 1 query per request
    max_pages = input_stride

    score = torch.randn(B, input_stride, dtype=torch.float32, device=device)
    lengths = torch.full((B,), input_stride, dtype=torch.int32, device=device)
    # page table: identity mapping for simplicity
    src_page_table = torch.arange(max_pages, dtype=torch.int32, device=device)
    src_page_table = src_page_table.unsqueeze(0).expand(B, -1).contiguous()
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=device)

    dst = fast_topk_transform_fused(
        score, lengths, src_page_table, cu_seqlens_q, TOPK
    )

    # With identity page table, dst should equal raw topk indices
    ref_indices = fast_topk_v2(score, lengths, TOPK)
    for i in range(B):
        dst_set = set(dst[i].cpu().tolist())
        ref_set = set(ref_indices[i].cpu().tolist())
        assert dst_set == ref_set, f"Row {i}: decode transform mismatch"


@pytest.mark.parametrize("num_requests", [1, 2])
def test_fast_topk_transform_prefill(num_requests):
    device = "cuda"
    input_stride = 8192
    queries_per_request = 3
    B = num_requests * queries_per_request
    max_pages = input_stride

    score = torch.randn(B, input_stride, dtype=torch.float32, device=device)
    lengths = torch.full((B,), input_stride, dtype=torch.int32, device=device)
    src_page_table = torch.arange(max_pages, dtype=torch.int32, device=device)
    src_page_table = src_page_table.unsqueeze(0).expand(num_requests, -1).contiguous()
    cu_seqlens_q = torch.arange(
        0, B + 1, queries_per_request, dtype=torch.int32, device=device
    )
    # Ensure last element is B
    cu_seqlens_q = torch.cat([
        cu_seqlens_q[:-1],
        torch.tensor([B], dtype=torch.int32, device=device),
    ])

    dst = fast_topk_transform_fused(
        score, lengths, src_page_table, cu_seqlens_q, TOPK
    )

    ref_indices = fast_topk_v2(score, lengths, TOPK)
    for i in range(B):
        dst_set = set(dst[i].cpu().tolist())
        ref_set = set(ref_indices[i].cpu().tolist())
        assert dst_set == ref_set, f"Row {i}: prefill transform mismatch"


@pytest.mark.parametrize("batch_size", [1, 4])
def test_fast_topk_transform_ragged(batch_size):
    device = "cuda"
    input_stride = 8192

    score = torch.randn(batch_size, input_stride, dtype=torch.float32, device=device)
    lengths = torch.full((batch_size,), input_stride, dtype=torch.int32, device=device)
    # Each row has offset = row_index * input_stride
    topk_indices_offset = torch.arange(
        0, batch_size * input_stride, input_stride,
        dtype=torch.int32, device=device,
    )

    dst = fast_topk_transform_ragged_fused(
        score, lengths, topk_indices_offset, TOPK
    )

    ref_indices = fast_topk_v2(score, lengths, TOPK)
    for i in range(batch_size):
        offset = topk_indices_offset[i].item()
        dst_set = set(dst[i].cpu().tolist())
        ref_set = set((ref_indices[i] + offset).cpu().tolist())
        assert dst_set == ref_set, f"Row {i}: ragged transform mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
