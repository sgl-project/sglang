import sys
from typing import Any, Optional

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


def _make_scores(kind: str, bs: int, width: int, seed: int) -> torch.Tensor:
    """Score distributions that stress the coarse stage of a histogram top-k.

    Everything above uses ``torch.randn``, which spreads over enough exponents that
    even a narrow coarse key separates it -- 127 populated buckets out of 256 on an
    8-bit key. Real DSA indexer logits are far more concentrated than that. Captured
    from a GLM-5.2 decode at 134,849 tokens of context, six consecutive indexer calls
    populate 4 to 126 buckets, with the largest holding 6% to 88% of the row.

    The three below bracket that regime, and each is calibrated to what it provokes on
    an 8-bit fp16 coarse key: `banded` populates 4 buckets with 47% in the largest,
    `narrow` collapses to a single bucket, and `subnormal` to two.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "diffuse":
        return torch.randn(bs, width, generator=g, device="cuda", dtype=torch.float32)
    if kind == "banded":
        # The value range of the worst real capture, [54, 88].
        return 54.0 + 34.0 * torch.rand(
            bs, width, generator=g, device="cuda", dtype=torch.float32
        )
    if kind == "narrow":
        # A row whose spread is small next to its magnitude, which is what makes a
        # truncating coarse key run out of buckets.
        return 70.0 + torch.randn(
            bs, width, generator=g, device="cuda", dtype=torch.float32
        )
    if kind == "subnormal":
        # Same shape as diffuse, scaled below fp16's smallest normal: a coarse key that
        # rounds through fp16 cannot separate this row at all, an fp32 one is unaffected.
        return 1e-16 * torch.randn(
            bs, width, generator=g, device="cuda", dtype=torch.float32
        )
    raise ValueError(kind)


def assert_exact(
    score: torch.Tensor, indices: torch.Tensor, seq_len: int, k: int
) -> None:
    """The selected scores must be the top-k scores, as a multiset.

    Stricter than ``assert_equal`` on purpose. Comparing index sets has to forgive
    tie-breaking, and that forgiveness is what lets a kernel selecting from a silently
    truncated candidate set pass: the indices it returns are all in range and all
    distinct, they are simply not the largest.
    """
    for i in range(score.shape[0]):
        want = torch.sort(
            torch.topk(score[i, :seq_len], k).values, descending=True
        ).values
        got = torch.sort(score[i, :seq_len][indices[i].long()], descending=True).values
        assert torch.equal(got, want), (
            f"row {i}: {int((got != want).sum())}/{k} selected scores are not the top-{k}"
        )


def assert_exact_rows(
    score: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    row_starts: Optional[torch.Tensor] = None,
) -> None:
    """Exact value-multiset check for variable windows and relative indices."""
    for i in range(score.shape[0]):
        length = int(lengths[i])
        start = 0 if row_starts is None else int(row_starts[i])
        row_indices = indices[i].long()
        assert torch.all((row_indices >= 0) & (row_indices < length))
        row = score[i, start : start + length]
        want = torch.sort(torch.topk(row, k).values).values
        got = torch.sort(row[row_indices]).values
        assert torch.equal(got, want), (
            f"row {i}: {int((got != want).sum())}/{k} selected scores are not the top-{k}"
        )


@pytest.mark.skipif(
    torch.version.hip is None or torch.cuda.device_count() < 2,
    reason="requires a multi-GPU ROCm runner",
)
@torch.inference_mode()
def test_topk_uses_score_device_and_rejects_mixed_devices() -> None:
    current_device = torch.cuda.current_device()
    score_device = (current_device + 1) % torch.cuda.device_count()
    score = torch.randn(1, 16384, dtype=torch.float32, device=score_device)
    lengths = torch.full((1,), 16384, dtype=torch.int32, device=score_device)

    indices = fast_topk_v2(score, lengths, 2048)
    assert_exact(score, indices, 16384, 2048)
    assert torch.cuda.current_device() == current_device

    with pytest.raises(RuntimeError, match="same device"):
        fast_topk_v2(score, lengths.to(f"cuda:{current_device}"), 2048)


@pytest.mark.skipif(
    torch.version.hip is None,
    reason="the CUDA kernel in csrc/elementwise/topk.cu shares this limitation; only "
    "the ROCm one (csrc/elementwise/topk.hip) is exact on these distributions",
)
@pytest.mark.parametrize("kind", ["diffuse", "banded", "narrow", "subnormal"])
@pytest.mark.parametrize("bs", [1, 4, 64])
@pytest.mark.parametrize("seq_len", [16384, 65536, 100500])
@torch.inference_mode()
def test_topk_is_exact_for_indexer_distributions(
    kind: str, bs: int, seq_len: int
) -> None:
    k = 2048
    score = _make_scores(kind, bs, MAX_SEQ_LEN, seed=seq_len + bs)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    assert_exact(score, fast_topk_v2(score, lengths, k), seq_len, k)


@pytest.mark.skipif(
    torch.version.hip is None,
    reason="the cooperative top-k implementation is only built on ROCm",
)
@pytest.mark.parametrize("kind", ["banded", "narrow", "subnormal"])
@pytest.mark.parametrize("force_one_block", [False, True])
@torch.inference_mode()
def test_topk_variants_are_exact_on_variable_windows(
    kind: str, force_one_block: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover both dispatches and all three APIs on adversarial score distributions."""
    if force_one_block:
        monkeypatch.setenv("SGL_DSA_TOPK_ROW_SPLIT", "0")
    else:
        monkeypatch.delenv("SGL_DSA_TOPK_ROW_SPLIT", raising=False)

    bs, k = 4, 2048
    score = _make_scores(kind, bs, MAX_SEQ_LEN, seed=1200 + force_one_block)
    lengths = torch.tensor(
        [65536, 70000, 90000, 100500], dtype=torch.int32, device="cuda"
    )
    row_starts = torch.tensor([0, 127, 511, 0], dtype=torch.int32, device="cuda")

    raw = fast_topk_v2(score, lengths, k, row_starts=row_starts)
    assert_exact_rows(score, raw, lengths, k, row_starts)

    ragged_offsets = torch.tensor(
        [17, 200000, 400000, 600000], dtype=torch.int32, device="cuda"
    )
    ragged = fast_topk_transform_ragged_fused(
        score, lengths, ragged_offsets, k, row_starts=row_starts
    )
    assert_exact_rows(score, ragged - ragged_offsets[:, None], lengths, k, row_starts)

    logical = torch.arange(MAX_SEQ_LEN, dtype=torch.int32, device="cuda")
    multipliers = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device="cuda")
    shifts = torch.tensor([19, 43, 71, 101], dtype=torch.int32, device="cuda")
    page_table = (
        logical[None, :] * multipliers[:, None] + shifts[:, None]
    ) % MAX_SEQ_LEN
    cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
    mapped = fast_topk_transform_fused(score, lengths, page_table, cu_seqlens_q, k)
    inverse_page_table = torch.empty_like(page_table)
    inverse_page_table.scatter_(1, page_table.long(), logical[None, :].expand(bs, -1))
    mapped_raw = inverse_page_table.gather(1, mapped.long())
    assert_exact_rows(score, mapped_raw, lengths, k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
