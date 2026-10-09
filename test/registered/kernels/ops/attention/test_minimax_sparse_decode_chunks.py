"""MiniMax-M3 sparse decode against a torch reference, with one and with several topk chunks.

The split-K launch picks NUM_TOPK_CHUNKS from batch_size * num_kv_heads; at 256 rows
there is a single chunk and the merge kernel is skipped, at 4 rows there are several.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

NQH, NKH, HD, BLK, TOPK = 16, 1, 128, 128, 16
SEQ_LEN = 4096


def _inputs(bs):
    q = torch.randn(bs, NQH, HD, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(bs * SEQ_LEN, NKH, HD, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    req_to_token = torch.stack(
        [torch.randperm(SEQ_LEN, device="cuda") + b * SEQ_LEN for b in range(bs)]
    ).to(torch.int32)
    seq_lens = SEQ_LEN - torch.randint(1, BLK, (bs,), dtype=torch.int32, device="cuda")
    slot_ids = torch.arange(bs, dtype=torch.int64, device="cuda")
    topk_idx = torch.full((NKH, bs, TOPK), -1, dtype=torch.int32, device="cuda")
    for b in range(bs):
        num_blocks = (int(seq_lens[b]) + BLK - 1) // BLK
        chosen = torch.randperm(num_blocks, device="cuda")[:TOPK]
        topk_idx[0, b, : chosen.numel()] = chosen.to(torch.int32)
    return q, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx


def _reference(q, k_cache, v_cache, req_to_token, seq_lens, topk_idx):
    out = torch.empty_like(q, dtype=torch.float32)
    for b in range(q.shape[0]):
        blocks = topk_idx[0, b]
        blocks = blocks[blocks >= 0].long()
        pos = (blocks[:, None] * BLK + torch.arange(BLK, device="cuda")).reshape(-1)
        pos = pos[pos < int(seq_lens[b])]
        slots = req_to_token[b, pos].long()
        k = k_cache[slots, 0].float()
        v = v_cache[slots, 0].float()
        scores = q[b].float() @ k.T * HD**-0.5
        out[b] = torch.softmax(scores, dim=-1) @ v
    return out


@pytest.mark.parametrize("bs", [4, 256])
def test_sparse_decode_matches_reference(bs):
    torch.manual_seed(0)
    q, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx = _inputs(bs)
    out = flash_decode_with_gqa_share_sparse(
        q=q,
        sink=None,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        slot_ids=slot_ids,
        block_size=BLK,
        topk_idx=topk_idx,
    )
    expected = _reference(q, k_cache, v_cache, req_to_token, seq_lens, topk_idx)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
