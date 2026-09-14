"""Parity coverage for the fused DSA k-pool top-k / pool-expansion / tail JIT kernel.

``fast_kpool_topk_transform_fused`` is the only implementation for the pooled
group budgets GLM-5.3-Flash uses (``index_topk=2048`` over ``index_kpool=4``
gives ``group_topk=512``); ``kpool_fp8_index`` has no Python fallback in that
range, so a build or numerical break here takes the model down rather than
making it slower.

The radix selector does not specify an output order and DSA attention is
permutation-invariant over the selected set, so the pooled columns are compared
as a set. The tail columns are positional and are compared exactly.

Registered for AMD only: the kernel had no direct coverage on any platform, and
adding CUDA coverage for it is not this change's call to make.
"""

import unittest

import torch

from sglang.kernels.ops.moe.kpool_topk_transform import fast_kpool_topk_transform_fused
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


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
        """The pooled top-k groups expanded to their ``pool_size`` token ids."""
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
            pool_size=self.POOL_SIZE,
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


if __name__ == "__main__":
    unittest.main()
