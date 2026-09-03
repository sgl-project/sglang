import sys
from typing import Optional

import pytest
import torch
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)


def _ref_torch_impl(
    score: torch.Tensor,
    seq_len: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        ks = row_starts.cpu().tolist()
        ke = (row_starts + seq_len).tolist()
        scores = []
        for i, (start, end) in enumerate(zip(ks, ke)):
            scores.append(score[i, start:end].unsqueeze(0))
        score = torch.cat(scores, dim=0)
        return torch.topk(score, topk, dim=-1, sorted=False).indices


def _ref_torch_transform_decode_impl(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)
    topk_indices = torch.empty(
        (batch_size, topk), dtype=torch.int32, device=score.device
    )
    for i in range(batch_size):
        topk_indices[i] = src_page_table[i, indices[i]]
    return topk_indices


def _ref_torch_transform_ragged_impl(
    score: torch.Tensor,
    seq_len: int,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    assert score.shape[0] == topk_indices_offset.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)

    mask = indices != -1
    topk_indices_offset = topk_indices_offset.unsqueeze(1)
    return torch.where(mask, indices + topk_indices_offset, indices)


MAX_SEQ_LEN = 131072
SMEM_CANDIDATE_CAPACITY = 4096


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0
        if len(more) > 0 or len(less) > 0:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(score[i, idx - offset].item() for idx in more)
            less_values = sorted(score[i, idx - offset].item() for idx in less)
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"
                )
        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"


def _assert_exact_selected_values(
    score: torch.Tensor, indices: torch.Tensor, topk: int
) -> None:
    selected = score.gather(1, indices.long()).sort(dim=1, descending=True).values
    reference = torch.topk(score, topk, dim=1, sorted=True).values
    torch.testing.assert_close(selected, reference, rtol=0, atol=0)


def _assert_valid_indices(
    indices: torch.Tensor, lengths: torch.Tensor, topk: int
) -> None:
    """Strict structural checks: shape, range, no duplicates, exact K valid."""
    B = indices.shape[0]
    assert indices.shape == (B, topk), f"shape mismatch: {indices.shape}"
    assert (indices >= 0).all(), "negative index found"
    for i in range(B):
        L = lengths[i].item()
        assert (indices[i] < L).all(), f"row {i}: index >= length ({L})"
        sorted_row = indices[i].sort().values
        dups = (sorted_row[1:] == sorted_row[:-1]).sum().item()
        assert dups == 0, f"row {i}: {dups} duplicate indices"


def _coarse_threshold_bucket_population(score: torch.Tensor, topk: int) -> torch.Tensor:
    half_bits = score.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    keys = torch.where(half_bits & 0x8000 != 0, half_bits ^ 0xFFFF, half_bits | 0x8000)
    bins = (keys >> 8) & 0xFF
    kth_indices = torch.topk(score, topk, dim=1).indices[:, -1:]
    kth_bins = bins.gather(1, kth_indices)
    return (bins == kth_bins).sum(dim=1)


@pytest.mark.parametrize("entrypoint", ["plain", "decode", "prefill", "ragged"])
@torch.inference_mode()
def test_topk_candidate_cache_overflow(entrypoint: str) -> None:
    """All AOT callers must recover candidates beyond the 4096-entry cache."""
    torch.manual_seed(0)
    batch_size, seq_len, topk = 2, 256 * 1024, 2048
    score = torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")
    score_input = score
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    assert bool(
        (_coarse_threshold_bucket_population(score, topk) > SMEM_CANDIDATE_CAPACITY)
        .all()
        .item()
    )

    if entrypoint == "plain":
        indices = fast_topk_v2(score, lengths, topk)
    elif entrypoint in ("decode", "prefill"):
        page_table = torch.arange(seq_len, dtype=torch.int32, device="cuda")
        page_table = page_table.unsqueeze(0).expand(batch_size, -1)
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
        # A non-null row_starts selects the prefill kernel. Use nonzero starts
        # so overflow recovery also exercises per-row input windows; returned
        # indices remain relative and therefore match the identity page table.
        row_starts = None
        if entrypoint == "prefill":
            row_starts = torch.tensor([17, 31], dtype=torch.int32, device="cuda")
            score_input = torch.empty(
                batch_size, seq_len + 64, dtype=torch.float32, device="cuda"
            )
            for row, start in enumerate(row_starts.tolist()):
                score_input[row, start : start + seq_len] = score[row]
        indices = fast_topk_transform_fused(
            score_input,
            lengths,
            page_table,
            cu_seqlens_q,
            topk,
            row_starts=row_starts,
        )
    else:
        offsets = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        indices = fast_topk_transform_ragged_fused(score, lengths, offsets, topk)

    _assert_valid_indices(indices, lengths, topk)
    _assert_exact_selected_values(score, indices, topk)


@torch.inference_mode()
def test_topk_repeated_candidate_cache_overflow() -> None:
    """A raw tail remains complete when several consecutive radix bins overflow."""
    torch.manual_seed(1)
    batch_size, seq_len, topk = 2, 16 * 1024, 2048
    values = 1.0 + torch.linspace(0, 1e-4, seq_len, dtype=torch.float32, device="cuda")
    score = torch.stack(
        [values[torch.randperm(seq_len, device="cuda")] for _ in range(batch_size)]
    )
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    # This narrow interval shares both the coarse bin and the first two full-key
    # bytes, forcing the resume tail to survive more than one overflow round.
    assert bool(
        (_coarse_threshold_bucket_population(score, topk) > SMEM_CANDIDATE_CAPACITY)
        .all()
        .item()
    )
    bits = score.view(torch.int32)
    full_keys = torch.where(bits < 0, ~bits, bits | torch.iinfo(torch.int32).min)
    kth_indices = torch.topk(score, topk, dim=1).indices[:, -1:]
    kth_prefix = (full_keys.gather(1, kth_indices) >> 16) & 0xFFFF
    prefix_population = (((full_keys >> 16) & 0xFFFF) == kth_prefix).sum(dim=1)
    assert bool((prefix_population > SMEM_CANDIDATE_CAPACITY).all().item())

    indices = fast_topk_v2(score, lengths, topk)
    _assert_valid_indices(indices, lengths, topk)
    _assert_exact_selected_values(score, indices, topk)


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_kernel(bs: int, k: int, seq_len: int, has_row_starts: bool) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    indices_ref = _ref_torch_impl(score, seq_len, k, row_starts=row_starts)
    indices_our = fast_topk_v2(score, lengths, k, row_starts=row_starts)

    # sort and compare
    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    # Tests can pass with max_permit_error=3, set to 5 for safety
    assert_equal(score, indices_ref, indices_our, bs, k, seq_len, max_permit_error=5)


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("mode", ["extend", "decode", "target_verify"])
@torch.inference_mode()
def test_topk_transform_kernel(bs: int, k: int, seq_len: int, mode: str) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    # NOTE: for decode, cumulative seqlens_q is just 0..=bs
    # NOTE: since page table is arange, they equal topk indices
    if mode == "decode":
        step = 1
    else:
        step = 4 if bs % 4 == 0 else 1
    num_tokens = bs
    bs = bs // step

    if mode == "extend":
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(
        0, num_tokens + 1, step=step, dtype=torch.int32, device="cuda"
    )
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)

    dst_page_table_ref = _ref_torch_transform_decode_impl(
        score=score,
        seq_len=seq_len,
        src_page_table=src_page_table,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        row_starts=row_starts,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        seq_len,
        max_permit_error=5,
    )


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_transform_ragged_kernel(
    bs: int, k: int, seq_len: int, has_row_starts: bool
) -> None:
    # Used in prefill only
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    # bs: # of q tokens
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    # kv_len
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device="cuda"
    )

    dst_page_table_ref = _ref_torch_transform_ragged_impl(
        score=score,
        seq_len=seq_len,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_ragged_fused(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        seq_len,
        topk_indices_offset,
        max_permit_error=5,
    )


@pytest.mark.skipif(
    torch.version.hip is None,
    reason="deepseek_v4_topk_transform_512 is only built on ROCm",
)
@pytest.mark.parametrize("bs", [1, 48])
@pytest.mark.parametrize("c4_len", [2048, 8192, 32768])
@torch.inference_mode()
def test_deepseek_v4_topk_transform(bs: int, c4_len: int) -> None:
    # c4_len 32768 is the 128k-context decode shape, i.e. the longest scan the
    # kernel runs and the one most sensitive to the block size it launches with.
    from sgl_kernel import deepseek_v4_topk_transform_512

    torch.manual_seed(42)
    topk, page_size = 1024, 64

    scores = torch.randn(bs, c4_len, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((bs,), c4_len, dtype=torch.int32, device="cuda")
    # Identity page table, so emitted paged slots equal raw token positions and
    # can be compared against torch.topk indices directly.
    num_pages = (c4_len + page_size - 1) // page_size
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )
    page_indices = torch.full((bs, topk), -1, dtype=torch.int32, device="cuda")

    deepseek_v4_topk_transform_512(
        scores, seq_lens, page_table, page_indices, page_size
    )

    indices_ref = torch.topk(scores, topk, dim=-1, sorted=False).indices
    assert_equal(
        scores,
        torch.sort(indices_ref, dim=-1).values,
        torch.sort(page_indices, dim=-1).values,
        bs,
        topk,
        c4_len,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
