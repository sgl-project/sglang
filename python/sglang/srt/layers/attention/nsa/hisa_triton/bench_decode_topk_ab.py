"""Rigorous A/B for fast_topk vs torch.topk on the decode hot path.

Mirrors the full decode step (stage 2 → stage 3 → stage 4) exactly as the
orchestrator runs it. Interleaves torch.topk and fast_topk runs in the
same process (controls for thermal / GPU-state drift across separate
process invocations). Reports per-call min / median / mean, plus a
paired-iteration delta.

Run: CUDA_VISIBLE_DEVICES=0 python bench_decode_topk_ab.py
"""
from __future__ import annotations

import statistics
import time

import torch

from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import (
    MAX_TOPK,
    fast_topk_runtime,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_triton,
    sparse_paged_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.profile_e2e_stages import (
    POOL_PAGE,
    make_decode_inputs,
)


DEVICE = torch.device("cuda")
H, D = 64, 128


def decode_step_torch_topk(q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req,
                           K, block_topk):
    score = batch_decode_pool_mqa_triton(
        q_fp8=q, pool_k_pages=pp, pool_page_tables=ppt,
        weights_f32=w, context_lens_pool=num_pool_per_req,
        pool_page_size=POOL_PAGE,
    )
    idx = torch.topk(
        score, k=min(block_topk, score.shape[-1]),
        dim=-1, sorted=False,
    ).indices
    out = sparse_paged_mqa_triton(
        q_fp8=q, kv_cache_fp8=kv, topk_block_index=idx,
        kv_block_size=K, weights=w,
        context_lens=ctxl, block_tables=bt,
    )
    return out


def decode_step_fast_topk(q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req,
                          K, block_topk):
    score = batch_decode_pool_mqa_triton(
        q_fp8=q, pool_k_pages=pp, pool_page_tables=ppt,
        weights_f32=w, context_lens_pool=num_pool_per_req,
        pool_page_size=POOL_PAGE,
    )
    B, S, L = score.shape
    topk = min(block_topk, L)
    idx = fast_topk_runtime(score.view(B, L), topk).view(B, S, topk)
    out = sparse_paged_mqa_triton(
        q_fp8=q, kv_cache_fp8=kv, topk_block_index=idx,
        kv_block_size=K, weights=w,
        context_lens=ctxl, block_tables=bt,
    )
    return out


def measure(fn, n_iters: int) -> float:
    """Time n_iters calls in a tight CUDA-event window."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / n_iters  # μs/call


def run_ab(K: int, B: int, ctx: int, n_blocks: int = 30, iters_per_block: int = 200):
    """Interleaved A/B. Each block: warm both, time block of torch then fast.

    Returns parallel lists of per-block μs/call for torch and fast topk.
    """
    block_topk = 8192 // K
    inputs = make_decode_inputs(K, B, ctx)
    q, kv, pp, ppt, w, ctxl, bt, num_pool_per_req = inputs

    def _fn_torch():
        return decode_step_torch_topk(q, kv, pp, ppt, w, ctxl, bt,
                                       num_pool_per_req, K, block_topk)

    def _fn_fast():
        return decode_step_fast_topk(q, kv, pp, ppt, w, ctxl, bt,
                                      num_pool_per_req, K, block_topk)

    # Heavy warmup — pull both kernels into instruction cache, prime caching
    # allocator, fix any first-call setup costs (cudaFuncSetAttribute etc.).
    for _ in range(50):
        _fn_torch()
        _fn_fast()
    torch.cuda.synchronize()

    torch_us, fast_us = [], []
    for blk in range(n_blocks):
        # Order alternates per-block to balance any state-drift effects:
        #   even block: torch first, fast second
        #   odd  block: fast first,  torch second
        if blk % 2 == 0:
            t1 = measure(_fn_torch, iters_per_block)
            t2 = measure(_fn_fast, iters_per_block)
            torch_us.append(t1); fast_us.append(t2)
        else:
            t2 = measure(_fn_fast, iters_per_block)
            t1 = measure(_fn_torch, iters_per_block)
            torch_us.append(t1); fast_us.append(t2)

    return torch_us, fast_us


def fmt_stats(label: str, vals: list[float]) -> str:
    return (f"{label:>10}  min={min(vals):>6.2f}  med={statistics.median(vals):>6.2f}"
            f"  mean={statistics.mean(vals):>6.2f}  max={max(vals):>6.2f}  "
            f"std={statistics.stdev(vals):>5.2f}")


def main():
    cases = [
        (16, 1, 128 * 1024),
        (16, 4, 128 * 1024),
        (32, 1, 128 * 1024),
        (32, 4, 128 * 1024),
    ]

    LAYERS = 61
    N_DECODE_STEPS = 256

    print("=" * 100)
    print(f"Decode step A/B  fast_topk vs torch.topk  "
          f"(30 blocks × 200 iters interleaved, full warmup)")
    print("=" * 100)
    print(f"{'K':>3} {'B':>3} {'ctx':>6} {'block_topk':>10} | "
          f"{'metric':>10}  {'min':>6}  {'med':>6}  {'mean':>6}  {'max':>6}  {'std':>5}")
    print("-" * 100)

    for K, B, ctx in cases:
        block_topk = 8192 // K
        torch_us, fast_us = run_ab(K, B, ctx)
        # Per-block paired delta (fast - torch).
        deltas = [f - t for f, t in zip(fast_us, torch_us)]

        print(f"\n  K={K} B={B} ctx={ctx//1024}K block_topk={block_topk}")
        print(f"     {fmt_stats('torch.topk', torch_us)}  μs/step")
        print(f"     {fmt_stats('fast_topk',  fast_us)}  μs/step")
        print(f"     {fmt_stats('Δ (f-t)',    deltas)}  μs/step")

        # Aggregated to e2e ms, using median (robust to outliers).
        torch_med = statistics.median(torch_us)
        fast_med = statistics.median(fast_us)
        decode_calls = LAYERS * N_DECODE_STEPS
        torch_total = torch_med * decode_calls / 1e3
        fast_total = fast_med * decode_calls / 1e3
        diff_ms = fast_total - torch_total
        pct = (fast_total / torch_total - 1) * 100
        print(f"     decode (× {LAYERS}L × {N_DECODE_STEPS} steps) median: "
              f"torch={torch_total:>8.1f}ms  fast={fast_total:>8.1f}ms  "
              f"Δ={diff_ms:+.1f}ms ({pct:+.2f}%)")


if __name__ == "__main__":
    main()
