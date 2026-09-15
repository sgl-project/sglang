"""K-pool top-k membership, tail, and graph-replay contracts on CUDA/ROCm."""

import sys

import pytest
import torch

from sglang.kernels.ops.moe.kpool_topk_transform import fast_kpool_topk_transform_fused
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: compile the HIP JIT path and exercise wave64 graph replay.
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")


@pytest.mark.parametrize(
    "distribution,start",
    [
        ("diffuse", 0),
        ("narrow", 0),
        ("banded", 0),
        ("inf", 8192),
        ("equal", 0),
        ("late_higher", 8192),
    ],
)
def test_topk_membership_including_overfull_coarse_bins(distribution, start):
    torch.manual_seed(1234)
    rows, width, length, group_topk, pool_size = 4, 50000, 32768, 512, 4
    scores = torch.randn(rows, width, device="cuda")
    if distribution == "banded":
        scores = 54 + 34 * torch.rand_like(scores)
    elif distribution == "equal":
        scores.fill_(1)
    elif distribution == "late_higher":
        scores.fill_(1)
        scores[:, start + length // 2 : start + length] = 1.001
    elif distribution in ("narrow", "inf"):
        scores += 70
        if distribution == "inf":
            scores[:, start + 8192 : start + 8224] = float("inf")
    lengths = torch.full((rows,), length, dtype=torch.int32, device="cuda")
    starts = torch.full_like(lengths, start)
    result = fast_kpool_topk_transform_fused(
        scores,
        lengths,
        pool_size,
        group_topk * pool_size,
        row_starts=starts,
        seq_lens=lengths * pool_size + 3,
    )
    groups = result[:, :2048:pool_size].long() // pool_size
    assert bool(((groups >= 0) & (groups < length)).all())
    for row in groups:
        assert torch.unique(row).numel() == group_topk
    torch.testing.assert_close(
        result[:, :2048].reshape(rows, group_topk, pool_size).long(),
        groups.unsqueeze(-1) * pool_size + torch.arange(pool_size, device="cuda"),
        atol=0,
        rtol=0,
    )
    # Ties may select any tied group and output order is unspecified.
    actual = scores.gather(1, groups + start).sort(dim=1).values
    expected = (
        scores[:, start : start + length].topk(group_topk).values.sort(dim=1).values
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        result[:, -3:],
        torch.arange(
            length * pool_size, length * pool_size + 3, dtype=torch.int32, device="cuda"
        ).expand(rows, -1),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("long_length", [600, 32768])
def test_graph_replay_short_rows_page_mapping_and_tail(long_length):
    width = long_length + 40
    scores = (
        torch.arange(width, dtype=torch.float32, device="cuda")
        .expand(3, -1)
        .contiguous()
    )
    lengths = torch.tensor([0, 3, long_length], dtype=torch.int32, device="cuda")
    seq_lens = lengths * 4 + torch.tensor([0, 2, 1], device="cuda", dtype=torch.int32)
    page_width = width * 4
    pages = torch.arange(4 * page_width, dtype=torch.int32, device="cuda").reshape(
        4, page_width
    )
    row_index = torch.tensor([2, 0, 3], dtype=torch.int32, device="cuda")

    def run():
        return fast_kpool_topk_transform_fused(
            scores,
            lengths,
            4,
            2048,
            page_table=pages,
            page_table_row_index=row_index,
            seq_lens=seq_lens,
        )

    run()  # Compile before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for reverse in (False, True):
        if reverse:
            scores.copy_(scores.flip(1))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.full_like(result, -1)
        for row, (length, tail) in enumerate(((0, 0), (3, 2), (long_length, 1))):
            selected = scores[row, :length].topk(min(length, 512)).indices
            tokens = (selected[:, None] * 4 + torch.arange(4, device="cuda")).flatten()
            n = tokens.numel()
            expected[row, :n] = pages[row_index[row], tokens]
            expected[row, n : n + tail] = pages[
                row_index[row], length * 4 : length * 4 + tail
            ]
        torch.testing.assert_close(
            result.sort(dim=1).values,
            expected.sort(dim=1).values,
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
