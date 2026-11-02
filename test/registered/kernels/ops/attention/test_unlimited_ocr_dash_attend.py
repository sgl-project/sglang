"""DASH bias-aware block-sparse attention (prefill, split-K decode + merge) must
match the PyTorch reference: biased remote blocks plus an unbiased local window."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.unlimited_ocr_dash.attend import (
    dash_sparse_attn,
    dash_sparse_attn_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.unlimited_ocr_dash import sparse_attn_reference

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_PAGES, NUM_KV_HEADS, NUM_HEADS, HEAD_DIM, TOPK = 48, 2, 8, 128, 6


def _cache(block_size: int, fp8: bool, gen: torch.Generator):
    shape = (NUM_PAGES, block_size, NUM_KV_HEADS, HEAD_DIM)
    k = torch.randn(shape, device="cuda", generator=gen).to(torch.bfloat16)
    v = torch.randn(shape, device="cuda", generator=gen).to(torch.bfloat16)
    if fp8:
        k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    return k, v


def _select(query_pos, local_blocks, block_size, gen, pad_slots=0):
    """Packed remote blocks per (kv head, row), never the query's own local window."""
    rows = len(query_pos)
    idx = torch.full((NUM_KV_HEADS, rows, TOPK + pad_slots), -1, dtype=torch.int32)
    for t, pos in enumerate(query_pos):
        # With no local window the query block itself must be selected.
        n_remote = pos // block_size + (1 if local_blocks == 0 else 1 - local_blocks)
        for g in range(NUM_KV_HEADS):
            pool = torch.randperm(max(n_remote, 0), generator=gen)[:TOPK]
            idx[g, t, : pool.numel()] = pool.to(torch.int32)
    return idx.cuda()


def _bias(idx, gen):
    return torch.randn(idx.shape, generator=gen).cuda() * (idx >= 0)


def _tolerance(fp8: bool):
    return dict(atol=6e-2, rtol=6e-2) if fp8 else dict(atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("local_blocks", [0, 1, 2])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("fp8", [False, True])
def test_prefill_matches_reference(block_size, local_blocks, with_bias, fp8):
    gen = torch.Generator().manual_seed(1)
    k, v = _cache(block_size, fp8, torch.Generator("cuda").manual_seed(1))
    # Prefix lengths hit position 0, a block edge and a mid-block start.
    prefix = [0, 2 * block_size - 1, 3 * block_size + 5]
    q_lens = [3, 4, 2 * block_size + 1]
    seq = [p + n for p, n in zip(prefix, q_lens)]
    bt = torch.stack([torch.randperm(NUM_PAGES, generator=gen)[:12] for _ in seq])
    query_pos = [p + j for p, n in zip(prefix, q_lens) for j in range(n)]
    row_req = [r for r, n in enumerate(q_lens) for _ in range(n)]
    idx = _select(query_pos, local_blocks, block_size, gen)
    bias = _bias(idx, gen) if with_bias else None
    q = torch.randn(len(query_pos), NUM_HEADS, HEAD_DIM, generator=gen)
    q = q.to(torch.bfloat16).cuda()
    cu = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    out = torch.zeros_like(q)
    dash_sparse_attn(
        q,
        k,
        v,
        idx,
        bias,
        bt.to(torch.int32).cuda(),
        cu.cuda(),
        torch.tensor(seq, dtype=torch.int32, device="cuda"),
        torch.tensor(prefix, dtype=torch.int32, device="cuda"),
        max(q_lens),
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        out,
        local_blocks=local_blocks,
        sparse_block_size=block_size,
    )
    ref = sparse_attn_reference(
        q=q,
        k_cache=k,
        v_cache=v,
        topk_idx=idx,
        blk_bias=bias,
        block_table=bt.cuda(),
        query_pos=query_pos,
        row_req=row_req,
        local_blocks=local_blocks,
        block_size=block_size,
        sm_scale=HEAD_DIM**-0.5,
    )
    torch.testing.assert_close(out.float(), ref, **_tolerance(fp8))


@pytest.mark.parametrize("decode_query_len", [1, 2])
@pytest.mark.parametrize("local_blocks", [0, 1])
@pytest.mark.parametrize("chunk_width", [None, 1, 64])
@pytest.mark.parametrize("fp8", [False, True])
def test_decode_matches_reference(decode_query_len, local_blocks, chunk_width, fp8):
    """chunk_width=1 forces a single split-K chunk; 64 forces more chunks than
    selected blocks, so the merge must ignore empty (-inf) chunks."""
    block_size = 128
    gen = torch.Generator().manual_seed(2)
    k, v = _cache(block_size, fp8, torch.Generator("cuda").manual_seed(2))
    # seq 1: only the query itself; 256: block-aligned end; 1000: deep context.
    seq = [1, 2 * block_size, 1000, block_size + 1]
    seq = [max(s, decode_query_len) for s in seq]
    bt = torch.stack([torch.randperm(NUM_PAGES, generator=gen)[:12] for _ in seq])
    query_pos = [s - decode_query_len + j for s in seq for j in range(decode_query_len)]
    row_req = [r for r in range(len(seq)) for _ in range(decode_query_len)]
    # Trailing -1 slots keep the stable buffer wider than any row's selection.
    idx = _select(query_pos, local_blocks, block_size, gen, pad_slots=2)
    bias = _bias(idx, gen)
    q = torch.randn(len(query_pos), NUM_HEADS, HEAD_DIM, generator=gen)
    q = q.to(torch.bfloat16).cuda()
    out = torch.full_like(q, float("nan"))
    dash_sparse_attn_decode(
        q,
        k,
        v,
        idx,
        bias,
        bt.to(torch.int32).cuda(),
        torch.tensor(seq, dtype=torch.int32, device="cuda"),
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        out,
        decode_query_len,
        local_blocks=local_blocks,
        sparse_block_size=block_size,
        chunk_width=chunk_width,
    )
    ref = sparse_attn_reference(
        q=q,
        k_cache=k,
        v_cache=v,
        topk_idx=idx,
        blk_bias=bias,
        block_table=bt.cuda(),
        query_pos=query_pos,
        row_req=row_req,
        local_blocks=local_blocks,
        block_size=block_size,
        sm_scale=HEAD_DIM**-0.5,
    )
    torch.testing.assert_close(out.float(), ref, **_tolerance(fp8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
