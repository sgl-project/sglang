"""Packed per-request index scoring must select the same blocks as per-row scoring."""

import unittest

import torch

from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

BLOCK_SIZE = 64
HEAD_DIM = 128
TOPK = 4


class TestPackedIndexScoring(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        torch.manual_seed(0)

    def _topk_sets(self, topk_idx: torch.Tensor):
        # [heads, rows, topk], front-packed and -1 padded.
        out = []
        for h in range(topk_idx.shape[0]):
            out.append(
                [sorted(int(v) for v in row.tolist() if v >= 0) for row in topk_idx[h]]
            )
        return out

    def _assert_packed_matches_per_row(
        self, score_type: str, num_q_heads: int, pack: int, prefixes, local_blocks=1
    ):
        dev = self.device
        num_reqs = len(prefixes)
        rows = num_reqs * pack
        max_len = max(prefixes) + pack
        max_reqs = num_reqs + 2
        # Identity slot mapping with a request offset, so requests do not share slots.
        req_to_token = (
            torch.arange(max_reqs * max_len, dtype=torch.int32, device=dev)
            .reshape(max_reqs, max_len)
            .contiguous()
        )
        k_cache = torch.randn(
            max_reqs * max_len, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev
        )
        q = torch.randn(rows, num_q_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        # tail keys point away from every query, so a shorter row cannot see them win a block
        for r, prefix in enumerate(prefixes):
            req_slot = r + 1
            request_queries = q[r * pack : (r + 1) * pack].float().reshape(-1, HEAD_DIM)
            away_from_queries = -(request_queries.sum(0)) * 10.0
            for j in range(1, pack):
                k_cache[req_to_token[req_slot, prefix + j], 0] = away_from_queries.to(
                    torch.bfloat16
                )
        slot_ids = torch.tensor(
            [r + 1 for r in range(num_reqs) for _ in range(pack)],
            dtype=torch.int32,
            device=dev,
        )
        seq_lens = torch.tensor(
            [p + j + 1 for p in prefixes for j in range(pack)],
            dtype=torch.int32,
            device=dev,
        )
        kernel_kwargs = dict(
            sink=None,
            k_cache=k_cache,
            v_cache=None,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            max_seqlen=max_len,
            slot_ids=slot_ids,
            block_size=BLOCK_SIZE,
            topk=TOPK,
            init_blocks=1,
            local_blocks=local_blocks,
            score_type=score_type,
            disable_index_value=True,
            page_size=1,
        )
        _, ref_idx, _ = flash_decode_with_topk_idx(q, **kernel_kwargs, packed_queries=1)
        _, packed_idx, _ = flash_decode_with_topk_idx(
            q, **kernel_kwargs, packed_queries=pack
        )
        self.assertEqual(ref_idx.shape, packed_idx.shape)
        self.assertEqual(self._topk_sets(ref_idx), self._topk_sets(packed_idx))

    def test_max_score_two_requests(self):
        """A permuted un-pack would hand one row another row's blocks."""
        self._assert_packed_matches_per_row(
            "max", num_q_heads=1, pack=4, prefixes=[700, 1200]
        )

    def test_max_score_multi_head_at_a_block_boundary(self):
        """A draft tail crossing a block boundary must keep every row's local block."""
        self._assert_packed_matches_per_row(
            "max", num_q_heads=2, pack=3, prefixes=[BLOCK_SIZE * 5 - 1, 333]
        )

    def test_lse_score(self):
        """The lse path shares the un-pack and must match per-row scoring as well."""
        self._assert_packed_matches_per_row(
            "lse", num_q_heads=1, pack=3, prefixes=[500, 900, 1300]
        )

    def test_no_local_blocks_falls_back_to_per_row(self):
        """Without local blocks packing cannot be exact, so it must be skipped."""
        self._assert_packed_matches_per_row(
            "max",
            num_q_heads=1,
            pack=3,
            prefixes=[BLOCK_SIZE * 5 - 1, 333],
            local_blocks=0,
        )


if __name__ == "__main__":
    unittest.main()
