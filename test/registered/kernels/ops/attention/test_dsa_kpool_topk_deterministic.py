"""The DSA k-pool top-k in deterministic inference: torch.topk with the selected
groups in ascending order, instead of the fused radix-select kernels (which
resolve ties and write their output in atomicAdd order).

The same logits must give the same output from run to run and whatever rows
share the call, including when every score ties, and the output must hold the
same tokens as the fused path (page-table mapping and partial-pool tail
included)."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa import kpool_fp8_index
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

KPOOL, TOPK = 4, 2048  # GLM-5.3: index_topk 2048 over index_kpool 4


def _topk(deterministic, scores, lengths, **kwargs):
    with patch.object(
        kpool_fp8_index, "_deterministic_inference", return_value=deterministic
    ):
        return kpool_fp8_index.topk_from_pooled_history_logits(
            logits=scores, group_lengths=lengths, kpool=KPOOL, topk=TOPK, **kwargs
        )


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestKpoolTopkDeterministic(CustomTestCase):
    def test_same_tokens_as_the_fused_path(self):
        width = 640
        scores = torch.stack(
            [torch.randperm(width, dtype=torch.float32) for _ in range(3)]
        ).cuda()
        lengths = torch.tensor([0, 3, 600], dtype=torch.int32, device="cuda")
        seq_lens = lengths * KPOOL + torch.tensor(
            [0, 2, 1], dtype=torch.int32, device="cuda"
        )
        pages = torch.arange(
            4 * width * KPOOL, dtype=torch.int32, device="cuda"
        ).reshape(4, -1)
        row_index = torch.tensor([2, 0, 3], dtype=torch.int32, device="cuda")
        kwargs = dict(
            page_table=pages, page_table_row_index=row_index, seq_lens=seq_lens
        )

        out = _topk(True, scores, lengths, **kwargs)
        fused = _topk(False, scores, lengths, **kwargs)
        self.assertEqual(out.shape, fused.shape)
        torch.testing.assert_close(
            out.sort(dim=1).values, fused.sort(dim=1).values, atol=0, rtol=0
        )
        # Selected groups in ascending order: their page entries ascend too.
        pooled = out[:, :TOPK]
        valid = pooled[:, 1:] >= 0
        self.assertTrue(bool((pooled[:, 1:] > pooled[:, :-1])[valid].all()))

    def test_ties_run_to_run_and_whatever_rows_share_the_call(self):
        rows, width = 8, 4096
        scores = torch.ones(rows, width, dtype=torch.float32, device="cuda")
        scores[::2, width // 2 :] = 1.001  # ties at the selection threshold
        lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
        seq_lens = lengths * KPOOL + 3

        full = _topk(True, scores, lengths, seq_lens=seq_lens)
        for _ in range(3):
            torch.testing.assert_close(
                _topk(True, scores, lengths, seq_lens=seq_lens), full, atol=0, rtol=0
            )
        for start, n in ((0, 1), (3, 1), (2, 5)):
            part = _topk(
                True,
                scores[start : start + n].contiguous(),
                lengths[start : start + n],
                seq_lens=seq_lens[start : start + n],
            )
            torch.testing.assert_close(part, full[start : start + n], atol=0, rtol=0)

    def test_out_rows_pads_with_minus_one(self):
        scores = torch.randn(2, 1024, device="cuda")
        lengths = torch.full((2,), 1024, dtype=torch.int32, device="cuda")
        out = _topk(True, scores, lengths, out_rows=5)
        self.assertEqual(tuple(out.shape), (5, TOPK))
        self.assertTrue(bool((out[2:] == -1).all()))


if __name__ == "__main__":
    unittest.main()
