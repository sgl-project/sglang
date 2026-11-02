"""Parity coverage for the fused DSA k-pool top-k / pool-expansion / tail JIT kernel.

``fast_kpool_topk_transform_fused`` is the only implementation for the pooled
group budgets GLM-5.3-Flash uses (``index_topk=2048`` over ``index_kpool=4``
gives ``group_topk=512``); ``kpool_fp8_index`` has no Python fallback in that
range, so a build or numerical break here takes the model down rather than
making it slower.

The radix selector does not specify an output order and DSA attention is
permutation-invariant over the selected set, so the pooled columns are compared
as a set. The tail columns are positional and are compared exactly.

The overfull-bin rescan is shared by CUDA and ROCm, so the tie and graph-replay
cases run on both.
"""

import unittest

import pytest
import torch

from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
    fast_kpool_topk_transform_fused,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: compile the HIP JIT path and exercise wave64 graph replay.
register_amd_ci(est_time=90, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestKpoolTopkTransformFused(CustomTestCase):
    POOL_SIZE = 4

    def _distinct_scores(self, rows: int, groups: int) -> torch.Tensor:
        """Strictly distinct scores per row, so top-k selection has no ties to break."""
        return torch.stack(
            [torch.randperm(groups, dtype=torch.float32) for _ in range(rows)]
        ).cuda()

    def _expected_tokens(
        self, score_row: torch.Tensor, group_topk: int
    ) -> torch.Tensor:
        """The pooled top-k groups expanded to their ``kpool`` token ids."""
        groups = torch.topk(score_row.float().cpu(), group_topk).indices
        offsets = torch.arange(self.POOL_SIZE, dtype=torch.int64)
        return (groups.unsqueeze(1) * self.POOL_SIZE + offsets).reshape(-1)

    def _run(self, rows, groups, topk, seq_lens_host=None):
        torch.manual_seed(0)
        score = self._distinct_scores(rows, groups)
        lengths = torch.full((rows,), groups, dtype=torch.int32, device="cuda")
        seq_lens = (
            torch.tensor(seq_lens_host, dtype=torch.int32, device="cuda")
            if seq_lens_host is not None
            else None
        )
        out = fast_kpool_topk_transform_fused(
            score=score,
            lengths=lengths,
            kpool=self.POOL_SIZE,
            topk=topk,
            seq_lens=seq_lens,
        )
        return score, out.cpu()

    def _assert_pooled_columns(self, score, out, topk):
        group_topk = topk // self.POOL_SIZE
        for row in range(score.shape[0]):
            selected = out[row, :topk]
            expected = self._expected_tokens(score[row], group_topk)
            self.assertEqual(
                sorted(selected.tolist()),
                sorted(expected.tolist()),
                msg=f"row {row}: selected token set differs from torch.topk",
            )

    def test_group_topk_512_matches_reference(self):
        # GLM-5.3-Flash: index_topk=2048 over index_kpool=4.
        score, out = self._run(rows=2, groups=1024, topk=2048)
        self._assert_pooled_columns(score, out, topk=2048)

    def test_runtime_topk_matches_reference(self):
        """Runtime dispatch must accept the same k-pool ratio as the fused kernel."""
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            topk_from_pooled_history_logits,
        )

        score = self._distinct_scores(rows=2, groups=1024)
        lengths = torch.full((2,), 1024, dtype=torch.int32, device="cuda")
        out = topk_from_pooled_history_logits(
            logits=score,
            group_lengths=lengths,
            kpool=self.POOL_SIZE,
            topk=2048,
        )
        self._assert_pooled_columns(score, out.cpu(), topk=2048)

    def test_group_topk_128_matches_reference(self):
        score, out = self._run(rows=2, groups=512, topk=512)
        self._assert_pooled_columns(score, out, topk=512)

    def test_tail_columns_hold_the_trailing_partial_pool(self):
        groups, topk = 1024, 2048
        for extra in range(self.POOL_SIZE):
            with self.subTest(tail=extra):
                seq_len = groups * self.POOL_SIZE + extra
                score, out = self._run(
                    rows=2, groups=groups, topk=topk, seq_lens_host=[seq_len] * 2
                )
                self._assert_pooled_columns(score, out, topk=topk)
                expected_tail = [seq_len - extra + i for i in range(extra)]
                expected_tail += [-1] * (self.POOL_SIZE - 1 - extra)
                for row in range(out.shape[0]):
                    self.assertEqual(out[row, topk:].tolist(), expected_tail)

    def test_output_width_carries_the_tail_columns(self):
        # kpool_fp8_index feeds this width straight into the page-table transform,
        # so it is 2048 + 3 = 2051 for GLM-5.3-Flash rather than a round 2048.
        topk = 2048
        _, out = self._run(rows=1, groups=1024, topk=topk, seq_lens_host=[1024 * 4 + 1])
        self.assertEqual(tuple(out.shape), (1, topk + self.POOL_SIZE - 1))


@pytest.mark.parametrize(
    "distribution,start",
    [
        ("equal", 0),
        ("late_higher", 8192),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_topk_membership_including_overfull_coarse_bins(distribution, start):
    torch.manual_seed(1234)
    rows, width, length, group_topk, kpool = 4, 50000, 32768, 512, 4
    scores = torch.randn(rows, width, device="cuda")
    if distribution == "equal":
        scores.fill_(1)
    elif distribution == "late_higher":
        scores.fill_(1)
        scores[:, start + length // 2 : start + length] = 1.001
    lengths = torch.full((rows,), length, dtype=torch.int32, device="cuda")
    starts = torch.full_like(lengths, start)
    result = fast_kpool_topk_transform_fused(
        scores,
        lengths,
        kpool,
        group_topk * kpool,
        row_starts=starts,
        seq_lens=lengths * kpool + 3,
    )
    groups = result[:, :2048:kpool].long() // kpool
    assert bool(((groups >= 0) & (groups < length)).all())
    for row in groups:
        assert torch.unique(row).numel() == group_topk
    torch.testing.assert_close(
        result[:, :2048].reshape(rows, group_topk, kpool).long(),
        groups.unsqueeze(-1) * kpool + torch.arange(kpool, device="cuda"),
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
            length * kpool, length * kpool + 3, dtype=torch.int32, device="cuda"
        ).expand(rows, -1),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
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
    unittest.main()
