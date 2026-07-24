"""Correctness test for the DCP-sharded DSA indexer's extend/chunked-prefill
(ragged) top-k pipeline: per-request local packing (dcp_pack_local_to_global)
-> per-query-token local top-k (window-relative indices, matching
_topk_unfused's real ``row_starts`` contract) -> pack -> merge.

Unlike decode (one query row per request, one shared local window), extend
packs each request's ENTIRE local K history into one flat ragged buffer, and
each query token in a multi-token extend chunk gets its own causal-bounded
sub-window (``local_ks``/``local_ke``) into that same buffer -- this is the
new piece dsa_indexer.py's _get_topk_ragged sharded path adds on top of the
primitives decode already validates.
"""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsa.dcp_localize_index_kv import (
    dcp_local_capacity,
    dcp_localize_page_table,
    dcp_pack_local_to_global,
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
    indexer_seq_len: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    seq_lens_expanded: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    topk: int,
    page_size: int,
):
    device = scores.device
    max_len = global_page_table_1.shape[1]
    num_tokens = token_to_batch_idx.shape[0]
    capacity = dcp_local_capacity(max_len, dcp_size, page_size)
    _, _, local_causal_count = dcp_localize_page_table(
        global_page_table_1, dcp_size, dcp_rank, capacity, page_size
    )
    local_row_totals = torch.gather(
        local_causal_count, 1, (indexer_seq_len.long() - 1).clamp(min=0).unsqueeze(1)
    ).squeeze(1)
    local_row_totals = torch.where(
        indexer_seq_len > 0, local_row_totals, torch.zeros_like(local_row_totals)
    ).to(torch.int32)
    local_row_offsets = (torch.cumsum(local_row_totals, dim=0) - local_row_totals).to(
        torch.int32
    )
    local_seq_len_sum = int(local_row_totals.sum().item())

    local_ks = local_row_offsets[token_to_batch_idx]
    per_token_causal_count = local_causal_count[token_to_batch_idx]
    col_idx = (seq_lens_expanded.long() - 1).clamp(min=0).unsqueeze(1)
    local_len_at_token = torch.gather(per_token_causal_count, 1, col_idx).squeeze(1)
    local_len_at_token = torch.where(
        seq_lens_expanded > 0, local_len_at_token, torch.zeros_like(local_len_at_token)
    )
    local_ke = local_ks + local_len_at_token

    packed_l2g = dcp_pack_local_to_global(
        global_page_table_1,
        dcp_size,
        dcp_rank,
        indexer_seq_len,
        local_row_offsets,
        local_seq_len_sum,
        page_size,
    )

    global_idx = torch.where(
        packed_l2g >= 0,
        packed_l2g.long(),
        torch.zeros(1, dtype=torch.long, device=device),
    )
    req_rows = scores[token_to_batch_idx]
    logits = torch.gather(req_rows, 1, global_idx.unsqueeze(0).expand(num_tokens, -1))
    valid_col = torch.arange(local_seq_len_sum, device=device).unsqueeze(0)
    in_window = (valid_col >= local_ks.unsqueeze(1)) & (
        valid_col < local_ke.unsqueeze(1)
    )
    logits = torch.where(in_window, logits, torch.full_like(logits, float("-inf")))

    # Mirrors _topk_unfused's real contract (dsa_topk_backend.py): absolute
    # topk indices, then subtract row_starts to get window-relative indices,
    # which is what pack_dcp_topk_candidates_cutedsl's row_starts expects.
    k_eff = min(topk, local_seq_len_sum)
    topk_scores, topk_col_indices = logits.topk(k_eff, dim=-1)
    local_topk = topk_col_indices.to(torch.int32) - local_ks.unsqueeze(1)
    local_topk = local_topk.masked_fill(topk_scores == float("-inf"), -1)
    if k_eff < topk:
        pad = torch.full(
            (num_tokens, topk - k_eff), -1, dtype=torch.int32, device=device
        )
        local_topk = torch.cat([local_topk, pad], dim=1)

    l2g_2d = packed_l2g.unsqueeze(0).expand(num_tokens, -1)
    packed = torch.empty((num_tokens, topk, 2), dtype=torch.float32, device=device)
    pack_dcp_topk_candidates_cutedsl(
        logits.contiguous(),
        local_topk.contiguous(),
        l2g_2d,
        packed,
        row_starts=local_ks,
    )
    return packed


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDCPIndexerTopkRagged(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def test_sharded_ragged_topk_matches_global_topk_multi_token_chunk(self) -> None:
        torch.manual_seed(1)
        dcp_size = 2
        num_reqs = 3
        max_len = 200
        topk = 512
        page_size = 64

        global_page_table_1 = torch.stack(
            [
                torch.randperm(max_len, dtype=torch.int32, device=self.device)
                for _ in range(num_reqs)
            ]
        )
        scores = torch.randn(num_reqs, max_len, device=self.device)
        indexer_seq_len = torch.tensor(
            [50, 200, 15], dtype=torch.int32, device=self.device
        )
        extend_lens = [4, 6, 3]  # varying multi-token extend chunks per request
        token_to_batch_idx = torch.tensor(
            sum(([i] * n for i, n in enumerate(extend_lens)), []),
            dtype=torch.int64,
            device=self.device,
        )
        seq_lens_expanded = torch.cat(
            [
                torch.arange(
                    indexer_seq_len[i] - extend_lens[i] + 1,
                    indexer_seq_len[i] + 1,
                    device=self.device,
                )
                for i in range(num_reqs)
            ]
        ).to(torch.int32)
        num_tokens = token_to_batch_idx.shape[0]

        packed_per_rank = [
            _rank_local_packed(
                scores,
                global_page_table_1,
                indexer_seq_len,
                token_to_batch_idx,
                seq_lens_expanded,
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

        for t in range(num_tokens):
            req = int(token_to_batch_idx[t])
            causal_len = int(seq_lens_expanded[t])
            valid_slots = global_page_table_1[req, :causal_len].long()
            valid_scores = scores[req, valid_slots]
            k_row = min(topk, causal_len)
            top_local = valid_scores.topk(k_row).indices
            expected = set(valid_slots[top_local].tolist())
            got = set(merged[t].tolist()) - {-1}
            self.assertEqual(
                got,
                expected,
                f"token {t} (req {req}, causal_len {causal_len}): sharded "
                "ragged pack+merge diverged from the true global top-k",
            )


if __name__ == "__main__":
    unittest.main()
