"""Identical FP4 index scores must retain the same KV set on Hopper decode.

The C1 candidate path previously changed keys tied at the selection boundary
between identical calls. Sorting the chosen indices did not fix the changing
set. Exercise the actual selector and physical-slot writeback without weights.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    PagedDecodeScores,
    select_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-large")

WIDTH = 8196
TOPK = 512


def make_scores(rows):
    generator = torch.Generator().manual_seed(24)
    # FP4 index logits round through BF16, so many keys have equal scores.
    values = torch.randn(rows, WIDTH, generator=generator).bfloat16().float()
    padded_width = ((WIDTH + 31) // 32) * 32
    storage = torch.empty(rows, padded_width, device="cuda", dtype=torch.float32)
    scores = storage[:, :WIDTH]
    scores.copy_(values)
    scores[:, 4130:] = float("nan")
    return scores


def select(scores, lengths, slots, output):
    data = PagedDecodeScores(
        bs=scores.shape[0],
        lmax=WIDTH,
        lens=lengths,
        scores=scores,
        req=torch.arange(scores.shape[0], device="cuda", dtype=torch.int64),
        req_to_token=slots,
        ratio=1,
        plan=plan_topk_v2(lengths),
        has_candidate_mask=True,
    )
    select_decode(SimpleNamespace(out_page_indices=output), data, TOPK)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "Requires Hopper CUDA",
)
class TestDSV41PagedDecodeTopK(CustomTestCase):
    @torch.inference_mode()
    def test_tied_scores_keep_the_same_kv_set(self):
        scores = make_scores(1)
        scores[:, :4130] = -4
        scores[:, :2000] = float("-inf")
        scores[:, 2000:2250] = -2
        lengths = torch.tensor([4130], device="cuda", dtype=torch.int32)
        slots = (torch.arange(WIDTH, device="cuda", dtype=torch.int32) * 3 + 51)[
            None, :
        ]
        reference = None
        expected_values = scores[0, :4130].topk(TOPK).values.sort().values.cpu()
        floor = expected_values[0]
        self.assertGreater(int((scores[0, :4130].cpu() == floor).sum()), 1)
        for _ in range(20):
            output = torch.full((1, TOPK), -7, device="cuda", dtype=torch.int32)
            select(scores, lengths, slots, output)
            logical = (output.cpu()[0].long() - 51) // 3
            self.assertEqual(logical.unique().numel(), TOPK)
            self.assertTrue(bool(((logical >= 0) & (logical < 4130)).all()))
            self.assertTrue(bool((logical[1:] > logical[:-1]).all()))
            chosen = scores[0].cpu()[logical].sort().values
            torch.testing.assert_close(chosen, expected_values, rtol=0, atol=0)
            if reference is None:
                reference = logical
            else:
                torch.testing.assert_close(logical, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_graph_refreshes_lengths_and_discards_masked_underfill(self):
        scores = make_scores(3)
        # A candidate consumer can retain fewer than TOPK finite values.
        scores[1, :4130] = float("-inf")
        scores[1, :4130:3] = 1
        lengths = torch.tensor([4130, 17, 0], device="cuda", dtype=torch.int32)
        slots = (
            torch.arange(WIDTH, device="cuda", dtype=torch.int32)[None, :] * 3
            + torch.tensor([51, 102, 153], device="cuda")[:, None]
        ).to(torch.int32)
        output = torch.empty((3, TOPK), device="cuda", dtype=torch.int32)
        select(scores, lengths, slots, output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            select(scores, lengths, slots, output)
        for values in ([4130, 17, 0], [4100, 63, 9], [0, 23, 4130]):
            lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
            output.fill_(-7)
            graph.replay()
            result = output.cpu()
            for row, length in enumerate(values):
                logical = (result[row][result[row] >= 0].long() - 51 * (row + 1)) // 3
                visible = scores[row, :length].cpu()
                expected = (
                    visible[torch.isfinite(visible)]
                    .topk(min(TOPK, int(torch.isfinite(visible).sum())))
                    .values.sort()
                    .values
                )
                self.assertEqual(logical.numel(), expected.numel())
                self.assertTrue(bool(((logical >= 0) & (logical < length)).all()))
                self.assertEqual(logical.unique().numel(), logical.numel())
                self.assertTrue(bool((result[row][logical.numel() :] == -1).all()))
                torch.testing.assert_close(
                    visible[logical].sort().values, expected, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
