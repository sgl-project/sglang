"""Unlimited-OCR v2 DASH and MoE ops on Apple MPS: Metal kernels vs torch fallback.

Times one decode layer's route / sparse attend / page-summary refresh and the
MoE expert compute at checkpoint geometry, once through the Metal kernels and
once through the torch fallback in ``unlimited_ocr_dash/mps.py``. Each loop
repeats the same inputs, so caches stay warm: the numbers compare the two
paths with each other, not with a serving step.

Run on an Apple Silicon Mac:
    python test/manual/kernels/attention/bench_unlimited_ocr_dash_mps.py
"""

import time

import torch

from sglang.kernels.ops.attention.unlimited_ocr_dash import mps
from sglang.kernels.ops.moe import moe_mps

DEV, BF16 = "mps", torch.bfloat16
BLOCK, HEADS, HEAD_DIM, INDEX_DIM, LOCAL, SLOTS = 32, 10, 128, 128, 4, 16
HIDDEN, INTER, EXPERTS, TOP_K = 1280, 896, 64, 6


def _ms(fn, iters=50):
    fn()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def _decode_inputs(batch, context):
    blocks = -(-context // BLOCK)
    pages = batch * blocks + 1
    kv = torch.randn(pages, BLOCK, HEADS, HEAD_DIM, device=DEV, dtype=BF16)
    table = torch.arange(1, pages, device=DEV, dtype=torch.int32).view(batch, blocks)
    seq = torch.full((batch,), context, device=DEV, dtype=torch.int64)
    sel = torch.randint(0, blocks - LOCAL - 1, (batch, SLOTS), device=DEV)
    sel = sel.to(torch.int32)[None].expand(HEADS, -1, -1)
    bias = torch.randn(batch, SLOTS, device=DEV)[None].expand(HEADS, -1, -1)
    return kv, table, seq, sel, bias, blocks


def bench_decode(batch, context):
    kv, table, seq, sel, bias, blocks = _decode_inputs(batch, context)
    q = torch.randn(batch, HEADS, HEAD_DIM, device=DEV, dtype=BF16)
    out = torch.empty_like(q)
    idx_q = torch.randn(batch, INDEX_DIM, device=DEV, dtype=BF16)
    k_bar = torch.randn(kv.shape[0], INDEX_DIM, device=DEV)
    out_idx = torch.empty(batch, SLOTS, device=DEV, dtype=torch.int32)
    out_bias = torch.empty(batch, SLOTS, device=DEV)
    route = dict(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=table,
        seq_lens=seq,
        image_allow=None,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=blocks,
        block_size=BLOCK,
        local_blocks=LOCAL,
        decode_query_len=1,
        idx_scale=0.1,
        prior_floor=0.01,
        sigma=1.0,
        budget=12,
        M=SLOTS,
    )
    attend = (q, kv, kv, sel, bias, table, seq, HEADS, HEAD_DIM**-0.5, out, 1)
    rows = {
        "route": (
            lambda: mps.route_decode(**route),
            lambda: mps.route_decode_torch(**route),
        ),
        "attend": (
            lambda: mps.dash_sparse_attn_decode(
                *attend, local_blocks=LOCAL, sparse_block_size=BLOCK
            ),
            lambda: mps.dash_sparse_attn_decode_torch(
                *attend, local_blocks=LOCAL, sparse_block_size=BLOCK
            ),
        ),
    }
    for name, (fast, slow) in rows.items():
        print(
            f"decode {name:7s} bs={batch} ctx={context}: metal {_ms(fast):.3f} ms  torch {_ms(slow):.3f} ms"
        )


def bench_moe(tokens):
    x = torch.randn(tokens, HIDDEN, device=DEV, dtype=BF16)
    w1 = torch.randn(EXPERTS, 2 * INTER, HIDDEN, device=DEV, dtype=BF16) * 0.02
    w2 = torch.randn(EXPERTS, HIDDEN, INTER, device=DEV, dtype=BF16) * 0.02
    ids = torch.stack([torch.randperm(EXPERTS)[:TOP_K] for _ in range(tokens)])
    ids = ids.to(DEV, torch.int32)
    weights = torch.rand(tokens, TOP_K, device=DEV).softmax(-1)
    args = (x, w1, w2, weights, ids)
    gemv = (
        _ms(lambda: moe_mps._experts_gemv(*args))
        if tokens * TOP_K <= 128
        else float("nan")
    )
    sorted_ms = _ms(lambda: moe_mps._experts_sorted(*args))
    print(
        f"moe tokens={tokens}: metal gemv {gemv:.3f} ms  torch sorted {sorted_ms:.3f} ms"
    )


if __name__ == "__main__":
    assert torch.backends.mps.is_available(), "requires Apple MPS"
    with torch.no_grad():
        for batch, context in ((1, 2048), (8, 2048), (8, 8192)):
            bench_decode(batch, context)
        for tokens in (1, 8, 1024):
            bench_moe(tokens)
