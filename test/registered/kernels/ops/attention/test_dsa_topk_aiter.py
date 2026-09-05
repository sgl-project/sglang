from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
    aiter_topk_transform_paged,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    torch.version.hip is None, reason="AITER DSA top-k requires ROCm"
)


def test_aiter_backend_resolves_for_target_and_draft() -> None:
    with get_context().override_server_args(
        dsa_topk_backend="aiter",
        speculative_dsa_topk_backend="sgl-kernel",
    ):
        target = DSATopKBackend.resolve(SimpleNamespace(is_draft_worker=False))
        draft = DSATopKBackend.resolve(SimpleNamespace(is_draft_worker=True))

    assert target is DSATopKBackend.AITER
    assert draft is DSATopKBackend.SGL_KERNEL

    with get_context().override_server_args(
        dsa_topk_backend="sgl-kernel",
        speculative_dsa_topk_backend="aiter",
    ):
        target = DSATopKBackend.resolve(SimpleNamespace(is_draft_worker=False))
        draft = DSATopKBackend.resolve(SimpleNamespace(is_draft_worker=True))

    assert target is DSATopKBackend.SGL_KERNEL
    assert draft is DSATopKBackend.AITER
    assert not DSATopKBackend.AITER.should_use_topk_v2()

    with (
        get_context().override_server_args(dsa_topk_backend="aiter"),
        patch(
            "sglang.srt.layers.attention.dsa.dsa_topk_backend.is_hip",
            return_value=False,
        ),
        pytest.raises(ValueError, match="requires ROCm"),
    ):
        DSATopKBackend.resolve(SimpleNamespace(is_draft_worker=False))


def _assert_selected_values(
    scores: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    for row in range(scores.shape[0]):
        length = int(lengths[row])
        count = min(length, indices.shape[1])
        selected = indices[row, :count].to(torch.long)
        assert selected.unique().numel() == count
        assert bool(((selected >= 0) & (selected < length)).all())
        expected = (
            torch.topk(scores[row, starts[row] : starts[row] + length], count)
            .values.sort()
            .values
        )
        actual = scores[row, starts[row] + selected].sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert bool((indices[row, count:] == -1).all())


@torch.inference_mode()
def test_aiter_topk_returns_row_relative_indices() -> None:
    torch.manual_seed(20260903)
    scores = torch.randn(4, 4096, dtype=torch.float32, device="cuda")
    starts = torch.tensor([0, 17, 64, 101], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([0, 127, 2048, 3995], dtype=torch.int32, device="cuda")

    indices = DSATopKBackend.AITER.topk_func(scores, lengths, 1024, row_starts=starts)
    _assert_selected_values(scores, starts, lengths, indices)


@torch.inference_mode()
def test_aiter_topk_compact_page_and_raw_outputs() -> None:
    torch.manual_seed(20260904)
    batch, width, topk, page_size = 4, 512, 64, 64
    scores = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    lengths = torch.tensor([32, 129, 384, 512], dtype=torch.int32, device="cuda")
    page_table = torch.stack(
        [torch.randperm(width // page_size, device="cuda") for _ in range(batch)]
    ).to(torch.int32)
    mapped = torch.empty((batch, topk), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(mapped)

    aiter_topk_transform_paged(
        scores,
        lengths,
        page_table,
        page_size,
        topk,
        out=mapped,
        out_raw_indices=raw,
    )

    valid = raw >= 0
    safe = raw.clamp_min(0)
    expected = (page_table.gather(1, (safe // page_size).long()) * page_size) + (
        safe % page_size
    )
    expected.masked_fill_(~valid, -1)
    torch.testing.assert_close(mapped, expected, rtol=0, atol=0)
    _assert_selected_values(
        scores,
        torch.zeros(batch, dtype=torch.int32, device="cuda"),
        lengths,
        raw,
    )


@torch.inference_mode()
def test_aiter_topk_ragged_offsets() -> None:
    torch.manual_seed(20260905)
    scores = torch.randn(3, 256, dtype=torch.float32, device="cuda")
    starts = torch.tensor([3, 19, 47], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([64, 128, 200], dtype=torch.int32, device="cuda")
    offsets = torch.tensor([1000, 2000, 3000], dtype=torch.int32, device="cuda")

    raw = DSATopKBackend.AITER.topk_func(scores, lengths, 32, row_starts=starts)
    metadata = type("Metadata", (), {"page_table_1": None})()
    ragged = DSATopKBackend.AITER.topk_transform(
        scores,
        lengths,
        32,
        TopkTransformMethod.RAGGED,
        metadata,
        topk_indices_offset=offsets,
        row_starts=starts,
    )

    torch.testing.assert_close(
        ragged.sort(dim=1).values,
        torch.where(raw >= 0, raw + offsets.unsqueeze(1), -1).sort(dim=1).values,
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_aiter_topk_expanded_paged_rows() -> None:
    torch.manual_seed(20260906)
    num_rows, width, topk = 4, 256, 32
    scores = torch.randn(num_rows, width, dtype=torch.float32, device="cuda")
    lengths = torch.full((num_rows,), width, dtype=torch.int32, device="cuda")
    page_table = torch.stack(
        [
            torch.arange(width, dtype=torch.int32, device="cuda"),
            torch.arange(width, dtype=torch.int32, device="cuda") + 10_000,
        ]
    )
    metadata = type(
        "Metadata",
        (),
        {
            "page_table_1": page_table,
            "cu_seqlens_k": torch.tensor(
                [0, width, 2 * width], dtype=torch.int32, device="cuda"
            ),
        },
    )()
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")

    mapped = DSATopKBackend.AITER.topk_transform(
        scores,
        lengths,
        topk,
        TopkTransformMethod.PAGED,
        metadata,
        cu_seqlens_q_topk=cu_seqlens_q,
    )
    raw = DSATopKBackend.AITER.topk_func(scores, lengths, topk)
    row_to_batch = torch.tensor([0, 0, 1, 1], device="cuda")
    expected = page_table[row_to_batch].gather(1, raw.long())
    torch.testing.assert_close(
        mapped.sort(dim=1).values,
        expected.sort(dim=1).values,
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_aiter_topk_graph_replay() -> None:
    torch.manual_seed(20260907)
    batch, width, topk, page_size = 4, 65536, 64, 64
    scores = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), width, dtype=torch.int32, device="cuda")
    page_table = torch.stack(
        [torch.randperm(width // page_size, device="cuda") for _ in range(batch)]
    ).to(torch.int32)
    mapped = torch.empty((batch, topk), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(mapped)

    aiter_topk_transform_paged(
        scores,
        lengths,
        page_table,
        page_size,
        topk,
        out=mapped,
        out_raw_indices=raw,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        aiter_topk_transform_paged(
            scores,
            lengths,
            page_table,
            page_size,
            topk,
            out=mapped,
            out_raw_indices=raw,
        )

    scores.copy_(
        torch.arange(width, 0, -1, dtype=torch.float32, device="cuda")
        .unsqueeze(0)
        .expand(batch, -1)
    )
    lengths.copy_(torch.tensor([32, 127, 32769, 65536], device="cuda"))
    page_table.add_(100)
    graph.replay()
    torch.cuda.synchronize()

    valid = raw >= 0
    safe = raw.clamp_min(0)
    expected = (page_table.gather(1, (safe // page_size).long()) * page_size) + (
        safe % page_size
    )
    expected.masked_fill_(~valid, -1)
    torch.testing.assert_close(mapped, expected, rtol=0, atol=0)
    _assert_selected_values(
        scores,
        torch.zeros(batch, dtype=torch.int32, device="cuda"),
        lengths,
        raw,
    )


@torch.inference_mode()
def test_aiter_topk_empty_paged_rows() -> None:
    scores = torch.empty((2, 0), dtype=torch.float32, device="cuda")
    lengths = torch.zeros(2, dtype=torch.int32, device="cuda")
    page_table = torch.empty((2, 0), dtype=torch.int32, device="cuda")
    mapped = torch.empty((2, 16), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(mapped)

    aiter_topk_transform_paged(
        scores,
        lengths,
        page_table,
        64,
        16,
        out=mapped,
        out_raw_indices=raw,
    )

    assert bool((mapped == -1).all())
    assert bool((raw == -1).all())


@torch.inference_mode()
def test_aiter_topk_shifted_graph_replay() -> None:
    torch.manual_seed(20260908)
    batch, width, topk, page_size = 4, 4096, 64, 64
    scores = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    starts = torch.tensor([3, 17, 65, 129], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([512, 1024, 2048, 3967], dtype=torch.int32, device="cuda")
    page_table = torch.stack(
        [torch.randperm(width // page_size, device="cuda") for _ in range(batch)]
    ).to(torch.int32)
    mapped = torch.empty((batch, topk), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(mapped)

    aiter_topk_transform_paged(
        scores,
        lengths,
        page_table,
        page_size,
        topk,
        row_starts=starts,
        out=mapped,
        out_raw_indices=raw,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        aiter_topk_transform_paged(
            scores,
            lengths,
            page_table,
            page_size,
            topk,
            row_starts=starts,
            out=mapped,
            out_raw_indices=raw,
        )

    scores.normal_()
    lengths.copy_(torch.tensor([256, 777, 1537, 3000], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_selected_values(scores, starts, lengths, raw)
