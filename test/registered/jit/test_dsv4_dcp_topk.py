import unittest

import torch

from sglang.jit_kernel.dsv4 import (
    merge_dcp_topk_candidates_512,
    topk_candidates_512,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


TOPK = 512
PAGE_SIZE = 64
DCP_SIZE = 2


def _local_seq_len(seq_len: int, rank: int) -> int:
    full_pages, tail = divmod(seq_len, PAGE_SIZE)
    local_pages = (full_pages + DCP_SIZE - 1 - rank) // DCP_SIZE
    return local_pages * PAGE_SIZE + (tail if full_pages % DCP_SIZE == rank else 0)


def _shard_scores(
    scores: torch.Tensor, seq_lens: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    local_lens = torch.tensor(
        [_local_seq_len(int(seq_len), rank) for seq_len in seq_lens.cpu()],
        device=scores.device,
        dtype=torch.int32,
    )
    local_scores = torch.full(
        (scores.shape[0], max(int(local_lens.max()), 1)),
        float("-inf"),
        device=scores.device,
        dtype=torch.float32,
    )
    for batch, seq_len in enumerate(seq_lens.cpu().tolist()):
        local_offset = 0
        num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        for page in range(rank, num_pages, DCP_SIZE):
            begin = page * PAGE_SIZE
            length = min(PAGE_SIZE, seq_len - begin)
            local_scores[
                batch, local_offset : local_offset + length
            ] = scores[batch, begin : begin + length]
            local_offset += length
    return local_scores, local_lens


class TestDSV4DCPTopK(CustomTestCase):
    def _run_case(self, seq_lens_list: list[int], tied: bool = False) -> None:
        device = torch.device("cuda")
        seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
        batch_size = len(seq_lens_list)
        max_seq_len = max(seq_lens_list)
        max_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        generator = torch.Generator(device=device).manual_seed(20260717)

        if tied:
            scores = torch.zeros(
                (batch_size, max_seq_len), device=device, dtype=torch.float32
            )
        else:
            scores = torch.randn(
                (batch_size, max_seq_len),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )

        page_tables = torch.stack(
            [
                torch.randperm(max_pages, device=device, generator=generator)
                for _ in range(batch_size)
            ]
        ).to(torch.int32)

        rank_candidates = []
        for rank in range(DCP_SIZE):
            local_scores, local_lens = _shard_scores(scores, seq_lens, rank)
            candidates = torch.empty(
                (batch_size, TOPK), device=device, dtype=torch.int64
            )
            topk_candidates_512(
                local_scores,
                local_lens,
                candidates,
                PAGE_SIZE,
                DCP_SIZE,
                rank,
            )
            rank_candidates.append(candidates)

        gathered = torch.cat(rank_candidates, dim=0)
        actual_raw = torch.empty(
            (batch_size, TOPK), device=device, dtype=torch.int32
        )
        actual_physical = torch.empty_like(actual_raw)
        merge_dcp_topk_candidates_512(
            gathered,
            seq_lens,
            page_tables,
            actual_physical,
            PAGE_SIZE,
            DCP_SIZE,
            actual_raw,
        )
        torch.cuda.synchronize()

        for batch, seq_len in enumerate(seq_lens_list):
            valid_count = min(seq_len, TOPK)
            raw = actual_raw[batch, :valid_count].to(torch.int64)
            self.assertTrue(torch.all((raw >= 0) & (raw < seq_len)).item())
            self.assertEqual(torch.unique(raw).numel(), valid_count)
            self.assertTrue(
                torch.all(actual_raw[batch, valid_count:] == -1).item()
            )

            if seq_len <= TOPK:
                expected_raw = torch.arange(seq_len, device=device)
                self.assertTrue(torch.equal(raw, expected_raw))
            else:
                expected_raw = torch.topk(scores[batch, :seq_len], TOPK).indices
                expected_scores = torch.sort(scores[batch, expected_raw]).values
                actual_scores = torch.sort(scores[batch, raw]).values
                self.assertTrue(torch.equal(actual_scores, expected_scores))

            expected_physical = (
                page_tables[batch, raw // PAGE_SIZE] * PAGE_SIZE
                + raw % PAGE_SIZE
            )
            self.assertTrue(
                torch.equal(
                    actual_physical[batch, :valid_count], expected_physical
                )
            )
            self.assertTrue(
                torch.all(actual_physical[batch, valid_count:] == -1).item()
            )

    def test_random_short_tail_and_long_history(self) -> None:
        self._run_case([1, 64, 65, 511, 512, 513, 875, 2051])

    def test_ties_and_empty_shard(self) -> None:
        self._run_case([1, 513, 1025], tied=True)


if __name__ == "__main__":
    unittest.main()
