import pytest
import torch

from sglang.kernels.ops.attention.dsa.hip_cooperative_topk import (
    hip_cooperative_topk,
    hip_cooperative_topk_page_size_one,
    hip_cooperative_topk_paged,
    hip_cooperative_topk_ragged,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    torch.version.hip is None, reason="HIP cooperative top-k requires ROCm"
)


def _assert_exact(
    score: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    for row in range(score.shape[0]):
        length = int(lengths[row])
        start = 0 if row_starts is None else int(row_starts[row])
        if length <= topk:
            assert torch.equal(
                indices[row, :length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert torch.all(indices[row, length:] == -1)
            continue
        section = score[row, start : start + length]
        row_indices = indices[row]
        assert torch.all((row_indices >= 0) & (row_indices < length))
        assert torch.unique(row_indices).numel() == topk
        got = torch.sort(section[row_indices.long()]).values
        want = torch.sort(torch.topk(section, topk).values).values
        assert torch.equal(got, want)


@pytest.mark.parametrize("topk", [256, 512, 1024, 2048, 4096])
def test_hip_cooperative_topk_supported_widths(topk: int) -> None:
    torch.manual_seed(42)
    width = 8192
    score = torch.randn(2, width, dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), width, dtype=torch.int32, device="cuda")

    indices = hip_cooperative_topk(score, lengths, topk)

    _assert_exact(score, indices, lengths, topk)


def test_hip_cooperative_topk_million_scores_k2028() -> None:
    torch.manual_seed(42)
    width = 1 << 20
    topk = 2028
    score = torch.randn(1, width, dtype=torch.float32, device="cuda")
    lengths = torch.full((1,), width, dtype=torch.int32, device="cuda")

    indices = hip_cooperative_topk(score, lengths, topk)

    _assert_exact(score, indices, lengths, topk)


def test_hip_cooperative_topk_short_and_empty_paged_rows() -> None:
    torch.manual_seed(42)
    topk = 512
    page_size = 64
    width = 65536
    lengths = torch.tensor([0, 1, 255, 511], dtype=torch.int32, device="cuda")
    score = torch.randn(len(lengths), width, dtype=torch.float32, device="cuda")
    num_pages = (topk + page_size - 1) // page_size
    page_table = torch.stack(
        [
            torch.randperm(num_pages, dtype=torch.int32, device="cuda")
            + row * num_pages
            for row in range(len(lengths))
        ]
    )
    page_indices = torch.full(
        (len(lengths), topk), 123, dtype=torch.int32, device="cuda"
    )
    raw_indices = torch.full_like(page_indices, 123)

    hip_cooperative_topk_paged(
        score,
        lengths,
        page_table,
        page_indices,
        page_size,
        raw_indices,
    )

    positions = torch.arange(topk, dtype=torch.int32, device="cuda").expand(
        len(lengths), -1
    )
    expected_raw = torch.where(
        positions < lengths[:, None], positions, torch.full_like(positions, -1)
    )
    assert torch.equal(raw_indices, expected_raw)

    safe_raw = expected_raw.clamp_min(0)
    expected_pages = (
        page_table.gather(
            1, torch.div(safe_raw, page_size, rounding_mode="floor").long()
        )
        * page_size
        + safe_raw % page_size
    )
    expected_pages = torch.where(
        expected_raw >= 0, expected_pages, torch.full_like(expected_pages, -1)
    )
    assert torch.equal(page_indices, expected_pages)


def test_hip_cooperative_topk_graph_capture_with_raw_indices() -> None:
    torch.manual_seed(42)
    batch = 2
    topk = 512
    width = 65536
    page_size = 64
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    lengths = torch.tensor([width, width - 123], dtype=torch.int32, device="cuda")
    num_pages = width // page_size
    page_table = torch.stack(
        [
            torch.randperm(num_pages, dtype=torch.int32, device="cuda")
            + row * num_pages
            for row in range(batch)
        ]
    )
    page_indices = torch.full((batch, topk), -1, dtype=torch.int32, device="cuda")
    raw_indices = torch.full_like(page_indices, -1)

    hip_cooperative_topk_paged(
        score, lengths, page_table, page_indices, page_size, raw_indices
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hip_cooperative_topk_paged(
            score, lengths, page_table, page_indices, page_size, raw_indices
        )
    graph.replay()
    torch.cuda.synchronize()

    _assert_exact(score, raw_indices, lengths, topk)
    expected_page_indices = (
        page_table.gather(
            1, torch.div(raw_indices, page_size, rounding_mode="floor").long()
        )
        * page_size
        + raw_indices % page_size
    )
    assert torch.equal(page_indices, expected_page_indices)


def test_hip_cooperative_topk_page_size_one_and_ragged_transforms() -> None:
    torch.manual_seed(42)
    batch = 3
    topk = 256
    width = 4096
    row_starts = torch.tensor([0, 11, 29], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [width, width - 11, width - 29], dtype=torch.int32, device="cuda"
    )
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")

    page_table = torch.stack(
        [
            torch.randperm(width, dtype=torch.int32, device="cuda") + row * width
            for row in range(batch)
        ]
    )
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    paged = hip_cooperative_topk_page_size_one(
        score, lengths, page_table, cu_seqlens_q, topk, row_starts
    )

    raw = hip_cooperative_topk(score, lengths, topk, row_starts)
    expected_paged = page_table.gather(1, raw.long())
    assert torch.equal(torch.sort(paged).values, torch.sort(expected_paged).values)

    offsets = torch.tensor([10, 20, 30], dtype=torch.int32, device="cuda")
    ragged = hip_cooperative_topk_ragged(score, lengths, offsets, topk, row_starts)
    expected_ragged = raw + offsets[:, None]
    assert torch.equal(torch.sort(ragged).values, torch.sort(expected_ragged).values)
