"""Correctness test for the DCP-sharded DSA indexer's target-verify /
draft-extend-v2 (MTP) top-k pipeline: per-request local page table (shared
with decode) -> per-(request, draft-position) local top-k, expanding
local_to_global across next_n draft rows via repeat_interleave -> pack ->
merge.

Unlike decode (one query row per request, 1:1 with local_to_global) and
extend (ragged, one flat packed buffer per request), target-verify has
B*next_n query rows sharing B per-request local_to_global rows -- each
request's row is reused for all of its next_n draft positions, and each
draft position has its own (larger) causal window since MTP/target-verify
draft position j attends up to base_seq_len + j + 1.
"""

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


def _rank_local_packed(
    scores: torch.Tensor,
    global_page_table_1: torch.Tensor,
    seqlens_expanded_2d: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    topk: int,
    page_size: int,
):
    device = scores.device
    B, next_n = seqlens_expanded_2d.shape
    max_len = global_page_table_1.shape[1]
    capacity = dcp_local_capacity(max_len, dcp_size, page_size)
    _, local_to_global, local_causal_count = dcp_localize_page_table(
        global_page_table_1, dcp_size, dcp_rank, capacity, page_size
    )

    col_idx = (seqlens_expanded_2d.long() - 1).clamp(min=0)
    local_len_2d = torch.gather(local_causal_count, 1, col_idx)
    local_len_2d = torch.where(
        seqlens_expanded_2d > 0, local_len_2d, torch.zeros_like(local_len_2d)
    )
    local_len_flat = local_len_2d.reshape(-1)

    local_to_global_expanded = local_to_global.repeat_interleave(next_n, dim=0)

    req_of_row = torch.arange(B * next_n, device=device) // next_n
    req_rows = scores[req_of_row]
    global_idx = torch.where(
        local_to_global_expanded >= 0,
        local_to_global_expanded.long(),
        torch.zeros(1, dtype=torch.long, device=device),
    )
    logits = torch.gather(req_rows, 1, global_idx)
    col_ids = torch.arange(capacity, device=device).unsqueeze(0)
    in_window = col_ids < local_len_flat.unsqueeze(1)
    logits = torch.where(in_window, logits, torch.full_like(logits, float("-inf")))

    # Paged/decode-style: local_to_global's own column IS the local index (no
    # packing row_starts offset, unlike the ragged extend layout), so topk's
    # raw column indices are already the "local" indices the pack kernel wants.
    k_eff = min(topk, capacity)
    topk_scores, topk_col = logits.topk(k_eff, dim=-1)
    local_topk = topk_col.to(torch.int32)
    local_topk = local_topk.masked_fill(topk_scores == float("-inf"), -1)
    if k_eff < topk:
        pad = torch.full(
            (B * next_n, topk - k_eff), -1, dtype=torch.int32, device=device
        )
        local_topk = torch.cat([local_topk, pad], dim=1)

    packed = torch.empty((B * next_n, topk, 2), dtype=torch.float32, device=device)
    pack_dcp_topk_candidates_cutedsl(
        logits.contiguous(),
        local_topk.contiguous(),
        local_to_global_expanded.contiguous(),
        packed,
        row_starts=None,
    )
    return packed


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDCPIndexerTopkTargetVerify(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def test_sharded_target_verify_topk_matches_global_topk(self) -> None:
        torch.manual_seed(2)
        dcp_size = 2
        B = 4
        next_n = 3
        # Headroom for base_seq_len + next_n: target-verify's kv_len already
        # includes the draft tokens (see _cal_indexer_k_start_end).
        max_len = 310
        topk = 512
        page_size = 64

        global_page_table_1 = torch.stack(
            [
                torch.randperm(max_len, dtype=torch.int32, device=self.device)
                for _ in range(B)
            ]
        )
        scores = torch.randn(B, max_len, device=self.device)
        base_seq_len = torch.tensor(
            [40, 300, 5, 100], dtype=torch.int32, device=self.device
        )
        seqlens_expanded_2d = (
            base_seq_len.unsqueeze(1)
            + torch.arange(1, next_n + 1, device=self.device).unsqueeze(0)
        ).to(torch.int32)

        packed_per_rank = [
            _rank_local_packed(
                scores,
                global_page_table_1,
                seqlens_expanded_2d,
                dcp_size,
                rank,
                topk,
                page_size,
            )
            for rank in range(dcp_size)
        ]
        gathered = torch.cat(packed_per_rank, dim=1).contiguous()
        merged = stable_topk_from_gathered_candidates_cutedsl(gathered, topk)
        torch.cuda.synchronize()

        for b in range(B):
            for j in range(next_n):
                row = b * next_n + j
                causal_len = int(seqlens_expanded_2d[b, j])
                valid_slots = global_page_table_1[b, :causal_len].long()
                valid_scores = scores[b, valid_slots]
                k_row = min(topk, causal_len)
                top_local = valid_scores.topk(k_row).indices
                expected = set(valid_slots[top_local].tolist())
                got = set(merged[row].tolist()) - {-1}
                self.assertEqual(
                    got,
                    expected,
                    f"request {b} draft-position {j} (causal_len {causal_len}): "
                    "sharded target-verify pack+merge diverged from the true "
                    "global top-k",
                )


if __name__ == "__main__":
    unittest.main()
