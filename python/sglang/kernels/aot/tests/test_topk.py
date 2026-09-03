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


def _tight_cluster(batch: int, width: int) -> torch.Tensor:
    """Finite scores that all land in one fp16 coarse radix bin."""
    base = torch.arange(width, dtype=torch.float32, device="cuda")
    values = 1.0 + base * (1e-3 / width)
    return values.expand(batch, -1).contiguous()


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0
        row_start = row_starts[i].item() if row_starts is not None else 0
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        assert len(indices_our_cpu[i]) == k
        assert len(indices_our_set_i) == k, "top-k output indices must be distinct"
        assert all(0 <= idx - offset < seq_len for idx in indices_our_cpu[i]), (
            "top-k output index is outside the selected window"
        )
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        if len(more) > 0 or len(less) > 0:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(
                score[i, row_start + idx - offset].item() for idx in more
            )
            less_values = sorted(
                score[i, row_start + idx - offset].item() for idx in less
            )
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"
                )
        assert wrong_values == 0, f"{wrong_values=}"


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

    assert_equal(score, indices_ref, indices_our, bs, k, seq_len, row_starts=row_starts)


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
        row_starts=row_starts,
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
        row_starts,
    )


@torch.inference_mode()
def test_legacy_topk_oversized_coarse_bin() -> None:
    """The shared legacy selector must not clip an oversized threshold bin."""
    torch.manual_seed(20260903)
    batch, seq_len, topk, row_start = 2, 7001, 2048, 3
    scores = torch.full(
        (batch, seq_len + row_start), -1000.0, dtype=torch.float32, device="cuda"
    )
    scores[:, row_start:] = _tight_cluster(batch, seq_len)
    lengths = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
    row_starts = torch.full((batch,), row_start, dtype=torch.int32, device="cuda")
    reference = _ref_torch_impl(scores, seq_len, topk, row_starts=row_starts)

    direct = fast_topk_v2(scores, lengths, topk, row_starts=row_starts)
    assert_equal(scores[:, row_start:], reference, direct, batch, topk, seq_len)

    offsets = torch.tensor([17, 4099], dtype=torch.int32, device="cuda")
    ragged = fast_topk_transform_ragged_fused(
        score=scores,
        lengths=lengths,
        topk_indices_offset=offsets,
        topk=topk,
        row_starts=row_starts,
    )
    assert_equal(
        scores[:, row_start:],
        reference + offsets[:, None],
        ragged,
        batch,
        topk,
        seq_len,
        offsets,
    )

    page_table = torch.arange(seq_len, dtype=torch.int32, device="cuda").expand(
        batch, -1
    )
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    paged = fast_topk_transform_fused(
        score=scores,
        lengths=lengths,
        page_table_size_1=page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=topk,
        row_starts=row_starts,
    )
    assert_equal(scores[:, row_start:], reference, paged, batch, topk, seq_len)

    # With one query per row and no row starts, this interface selects the
    # separate decode kernel rather than the prefill kernel above.
    decode_scores = scores[:, row_start:].contiguous()
    decode = fast_topk_transform_fused(
        score=decode_scores,
        lengths=lengths,
        page_table_size_1=page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=topk,
    )
    assert_equal(decode_scores, reference, decode, batch, topk, seq_len)


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


@pytest.mark.skipif(
    torch.version.hip is None,
    reason="deepseek_v4_topk_transform_512 is only built on ROCm",
)
@torch.inference_mode()
def test_deepseek_v4_topk_oversized_coarse_bin_and_idle_row() -> None:
    """Oversized bins select exactly; a negative DP-idle length emits padding."""
    from sgl_kernel import deepseek_v4_topk_transform_512

    torch.manual_seed(20260903)
    active_batch, batch, c4_len, topk, page_size = 2, 3, 7001, 1024, 64
    scores = _tight_cluster(batch, c4_len)
    seq_lens = torch.tensor(
        [c4_len] * active_batch + [-1], dtype=torch.int32, device="cuda"
    )
    num_pages = (c4_len + page_size - 1) // page_size
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .expand(batch, -1)
        .contiguous()
    )
    page_indices = torch.full((batch, topk), -2, dtype=torch.int32, device="cuda")
    raw_indices = torch.full_like(page_indices, -2)

    deepseek_v4_topk_transform_512(
        scores,
        seq_lens,
        page_table,
        page_indices,
        page_size,
        raw_indices,
    )

    reference = torch.topk(scores[:active_batch], topk, dim=-1, sorted=False).indices
    assert_equal(
        scores[:active_batch],
        reference,
        raw_indices[:active_batch],
        active_batch,
        topk,
        c4_len,
    )
    assert_equal(
        scores[:active_batch],
        reference,
        page_indices[:active_batch],
        active_batch,
        topk,
        c4_len,
    )
    assert bool((raw_indices[-1] == -1).all())
    assert bool((page_indices[-1] == -1).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
