import sys
from typing import Any, Optional

import pytest
import torch
from sgl_kernel import (
    fast_kpool_topk_transform_fused,
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


def _ref_torch_kpool_transform_impl(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows = score.shape[0]
    group_topk = topk // pool_size
    offsets = torch.arange(pool_size, dtype=torch.int32, device=score.device)
    out_cols = topk + (pool_size - 1 if seq_lens is not None else 0)
    out = torch.full((rows, out_cols), -1, dtype=torch.int32, device=score.device)
    for i in range(rows):
        length = int(lengths[i].item())
        valid_count = min(length, group_topk)
        write_pos = 0
        if valid_count == 0:
            token_ids = torch.empty((0,), dtype=torch.int32, device=score.device)
        elif length <= group_topk:
            selected = torch.arange(length, dtype=torch.int32, device=score.device)
            token_ids = (selected.unsqueeze(1) * pool_size + offsets).reshape(-1)
        else:
            selected = torch.topk(
                score[i, :length], group_topk, dim=-1, sorted=False
            ).indices.to(torch.int32)
            token_ids = (selected.unsqueeze(1) * pool_size + offsets).reshape(-1)
        if token_ids.numel() > 0:
            if page_table is not None:
                token_ids = page_table[i, token_ids.long()].to(torch.int32)
            elif topk_indices_offset is not None:
                token_ids = token_ids + topk_indices_offset[i].to(torch.int32)
            write_pos = valid_count * pool_size
            out[i, :write_pos] = token_ids[:write_pos]
        if seq_lens is not None:
            tail_count = int(seq_lens[i].item()) % pool_size
            if tail_count > 0:
                raw_tail = length * pool_size + torch.arange(
                    tail_count, dtype=torch.int32, device=score.device
                )
                if page_table is not None:
                    tail = page_table[i, raw_tail.long()].to(torch.int32)
                elif topk_indices_offset is not None:
                    tail = raw_tail + topk_indices_offset[i].to(torch.int32)
                else:
                    tail = raw_tail
                out[i, write_pos : write_pos + tail_count] = tail
    return out


MAX_SEQ_LEN = 131072


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


@pytest.mark.parametrize(
    "pool_size,group_topk",
    [(16, 128), (16, 160), (16, 192), (16, 224), (8, 256), (4, 512)],
)
@pytest.mark.parametrize("mode", ["raw", "paged", "ragged"])
@pytest.mark.parametrize("append_tail", [False, True])
@torch.inference_mode()
def test_kpool_topk_transform_kernel(
    pool_size: int, group_topk: int, mode: str, append_tail: bool
) -> None:
    torch.manual_seed(42)
    bs = 17
    topk = pool_size * group_topk
    num_groups = 4096
    score = torch.randn(bs, num_groups, dtype=torch.float32, device="cuda")
    lengths = torch.randint(
        group_topk + 1, num_groups + 1, (bs,), dtype=torch.int32, device="cuda"
    )

    page_table = None
    topk_indices_offset = None
    seq_lens = None
    tail_counts = torch.randint(0, pool_size, (bs,), dtype=torch.int32, device="cuda")
    if append_tail:
        seq_lens = lengths * pool_size + tail_counts
    if mode == "paged":
        page_table = torch.arange(
            bs * (num_groups * pool_size + pool_size),
            dtype=torch.int32,
            device="cuda",
        ).view(bs, num_groups * pool_size + pool_size)
    elif mode == "ragged":
        topk_indices_offset = torch.randint(
            0, 2048, (bs,), dtype=torch.int32, device="cuda"
        )

    out_ref = _ref_torch_kpool_transform_impl(
        score,
        lengths,
        pool_size,
        topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        seq_lens=seq_lens,
    )
    out_our = fast_kpool_topk_transform_fused(
        score,
        lengths,
        pool_size,
        topk,
        page_table=page_table,
        topk_indices_offset=topk_indices_offset,
        seq_lens=seq_lens,
    )

    out_ref = torch.sort(out_ref, dim=-1).values
    out_our = torch.sort(out_our, dim=-1).values
    torch.testing.assert_close(out_our, out_ref, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
