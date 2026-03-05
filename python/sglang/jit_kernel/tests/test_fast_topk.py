import pytest
import torch

from sglang.jit_kernel.fast_topk import fast_topk_v2

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
