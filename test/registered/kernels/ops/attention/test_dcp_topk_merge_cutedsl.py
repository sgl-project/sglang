from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsa.dcp_localize_index_kv import (
    dcp_local_capacity,
    dcp_localize_page_table,
)
from sglang.kernels.ops.attention.dsa.dcp_topk_merge_cutedsl import (
    pack_dcp_topk_candidates_cutedsl,
    stable_topk_from_gathered_candidates_cutedsl,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")


def _rank_local_topk(
    scores: torch.Tensor,
    page_table_1: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    topk: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This rank's local logits/top-k/local_to_global, mirroring what
    dsa_indexer.py's _get_topk_paged would compute against a real (sharded)
    index K-cache: score each locally-owned slot, then top-k locally."""
    capacity = dcp_local_capacity(page_table_1.shape[1], dcp_size, page_size)
    local_page_table, local_to_global, local_causal_count = dcp_localize_page_table(
        page_table_1, dcp_size, dcp_rank, capacity, page_size
    )
    local_len = local_causal_count[:, -1]
    global_idx = torch.where(
        local_to_global >= 0,
        local_to_global.long(),
        torch.zeros_like(local_to_global.long()),
    )
    local_logits = torch.gather(scores, 1, global_idx)
    local_logits = torch.where(
        local_to_global >= 0,
        local_logits,
        torch.full_like(local_logits, float("-inf")),
    )
    k = min(topk, local_logits.shape[1])
    local_topk_idx = local_logits.topk(k, dim=-1).indices.to(torch.int32)
    valid_col = torch.arange(local_logits.shape[1], device=scores.device).unsqueeze(
        0
    ) < local_len.unsqueeze(1)
    keep = torch.gather(valid_col, 1, local_topk_idx.long())
    local_topk_idx = torch.where(
        keep, local_topk_idx, torch.full_like(local_topk_idx, -1)
    )
    if k < topk:
        pad = torch.full(
            (page_table_1.shape[0], topk - k),
            -1,
            dtype=torch.int32,
            device=scores.device,
        )
        local_topk_idx = torch.cat([local_topk_idx, pad], dim=1)
    return (
        local_logits.contiguous(),
        local_topk_idx.contiguous(),
        local_to_global.contiguous(),
    )


def _sharded_merged_topk(
    scores: torch.Tensor,
    page_table_1: torch.Tensor,
    dcp_size: int,
    topk: int,
    page_size: int,
) -> torch.Tensor:
    packed_per_rank = []
    for rank in range(dcp_size):
        local_logits, local_topk_idx, local_to_global = _rank_local_topk(
            scores, page_table_1, dcp_size, rank, topk, page_size
        )
        packed = torch.empty(
            (page_table_1.shape[0], topk, 2), dtype=torch.float32, device=scores.device
        )
        pack_dcp_topk_candidates_cutedsl(
            local_logits, local_topk_idx, local_to_global, packed, None
        )
        packed_per_rank.append(packed)
    gathered = torch.cat(packed_per_rank, dim=1).contiguous()
    return stable_topk_from_gathered_candidates_cutedsl(gathered, topk)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDCPTopkMergeCutedsl(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _check(
        self,
        num_rows: int,
        max_seq_len: int,
        dcp_size: int,
        topk: int,
        seed: int,
        page_size: int = 64,
    ) -> None:
        torch.manual_seed(seed)
        scores = torch.randn(num_rows, max_seq_len, device=self.device)
        page_table_1 = torch.stack(
            [
                torch.randperm(max_seq_len, device=self.device, dtype=torch.int32)
                for _ in range(num_rows)
            ]
        )

        merged = _sharded_merged_topk(scores, page_table_1, dcp_size, topk, page_size)
        torch.cuda.synchronize()

        ref = scores.topk(min(topk, max_seq_len), dim=-1).indices.to(torch.int32)
        for row in range(num_rows):
            merged_set = set(merged[row].tolist()) - {-1}
            ref_set = set(ref[row].tolist())
            self.assertEqual(
                merged_set,
                ref_set,
                f"row {row}: sharded pack+merge top-k diverged from the true "
                "global top-k over the unsharded scores",
            )

    def test_sharded_topk_matches_global_topk_dcp2(self):
        # candidate count (dcp_size * topk) must be a multiple of 512 for the
        # radix-select kernel; 2 * 512 = 1024.
        self._check(num_rows=4, max_seq_len=3000, dcp_size=2, topk=512, seed=0)

    def test_sharded_topk_matches_global_topk_dcp4(self):
        self._check(num_rows=3, max_seq_len=5000, dcp_size=4, topk=512, seed=1)

    def test_sharded_topk_matches_global_topk_short_context(self):
        # seq_len << topk: every rank's local top-k is surplus-padded with -1,
        # exercising the padding/valid-count path through pack + merge.
        self._check(num_rows=2, max_seq_len=300, dcp_size=2, topk=512, seed=2)


if __name__ == "__main__":
    unittest.main()
