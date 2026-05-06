"""Correctness + speed test for hisa fast_topk_runtime.

Correctness: index-set equivalence with torch.topk (top-k is unsorted in
both, so we compare sets per row). Tests:
  * topk in {64, 128, 256, 512, 1024, 2048} over full row
  * caller-side -inf masking (mimics stage-2 padding contract)
  * cuda graph capture

The runtime API is minimal: ``fast_topk_runtime(score, topk)``. Caller
masks invalid positions to -inf — the radix select naturally filters
them out.

Speed: μs/call vs torch.topk(bf16, sorted=False) on production hisa
stage 3 shapes — prefill rows × num_blocks, decode B × max_pool_pages.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import fast_topk_runtime


DEVICE = torch.device("cuda")


def make_score(B, L, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, L, device=DEVICE, dtype=torch.float32)


def topk_score_equal(
    out: torch.Tensor, ref: torch.Tensor, score: torch.Tensor,
) -> bool:
    """Per-row score-multiset equality.

    Index sets can disagree at ties (e.g., two values exactly equal the
    K-th largest — torch picks one, radix picks the other; both valid
    top-k). Compare the *sorted score values* at the chosen indices: if
    they match, both answers are correct top-k.
    """
    if out.shape != ref.shape:
        return False
    B = out.size(0)
    for b in range(B):
        oi = out[b].long()
        ri = ref[b].long()
        if oi.numel() != ri.numel():
            return False
        os_v = score[b, oi].sort().values
        rs_v = score[b, ri].sort().values
        if not torch.equal(os_v, rs_v):
            return False
    return True


def correctness_full_row():
    """torch.topk vs fast_topk_runtime over full rows."""
    print("=" * 80)
    print("Correctness — full-row topk")
    print("=" * 80)
    print(f"{'B':>3} {'L':>5} {'topk':>5} | {'set_eq':>7} {'fail_rows':>9}")
    print("-" * 80)
    cases = []
    for B in (1, 4, 32):
        for L in (1024, 2048, 4096, 8192):
            for K in (64, 128, 256, 512, 1024, 2048):
                if K > L:
                    continue
                cases.append((B, L, K))
    fails = 0
    for B, L, K in cases:
        score = make_score(B, L)
        out = fast_topk_runtime(score, K)
        ref = torch.topk(score, K, dim=-1, sorted=False).indices.to(torch.int32)
        ok = topk_score_equal(out, ref, score)
        if not ok:
            fails += 1
            print(f"{B:>3} {L:>5} {K:>5} | {str(ok):>7} {'?':>9}")
    if fails == 0:
        print(f"  all {len(cases)} cases PASS")
    return fails


def correctness_inf_masking():
    """Caller masks invalid columns to -inf → fast_topk must skip them.

    This is the production contract: stage 2 writes -inf at out-of-range
    pool positions, fast_topk picks only from the valid prefix.
    """
    print()
    print("=" * 80)
    print("Correctness — -inf masking (mimics stage-2 padding contract)")
    print("=" * 80)
    print(f"{'B':>3} {'L':>5} {'topk':>5} {'valid':>6} | {'set_eq':>7}")
    print("-" * 80)
    cases = [
        (4,  512, 256,  300),
        (4, 2048, 512, 1024),
        (4, 8192, 1024, 2048),
        (8, 8192, 2048, 4096),
    ]
    fails = 0
    for B, L, K, valid in cases:
        assert valid >= K, "test invariant: valid prefix must be >= topk"
        score = make_score(B, L)
        score[:, valid:] = float("-inf")
        out = fast_topk_runtime(score, K)
        # Reference: torch.topk over the full (already -inf padded) row
        ref = torch.topk(score, K, dim=-1, sorted=False).indices.to(torch.int32)
        ok = topk_score_equal(out, ref, score)
        # All chosen indices must fall inside [0, valid).
        in_range = (out < valid).all().item()
        if not (ok and in_range):
            fails += 1
        print(f"{B:>3} {L:>5} {K:>5} {valid:>6} | {str(ok and in_range):>7}")
    return fails


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters  # μs


def speed_full_row():
    """fast_topk_runtime vs torch.topk(bf16) on production stage 3 shapes."""
    print()
    print("=" * 100)
    print("Speed — fast_topk_runtime vs torch.topk(bf16) (μs/call)")
    print("=" * 100)
    print(f"{'mode':>10} {'B':>5} {'L':>6} {'K':>5} | "
          f"{'topk_bf16':>10} {'topk_f32':>10} {'fast_topk':>10} | {'speedup':>8}")
    print("-" * 100)

    cases = []
    # Prefill: rows = 8192, cols = ctx/k_block (production formula
    # block_topk = 8192 // k_block_size).
    for k_blk, K, k_label in [(16, 512, 16), (32, 256, 32), (64, 128, 64), (128, 64, 128)]:
        ctx_blocks = 131072 // k_blk
        cases.append(("prefill", 8192, ctx_blocks, K))
    # Decode: rows = B, cols = ctx_pool_pages.
    for B in (1, 4, 32):
        for k_blk, K in [(16, 512), (32, 256), (64, 128), (128, 64)]:
            ctx_blocks = 131072 // k_blk
            cases.append(("decode", B, ctx_blocks, K))

    for label, B, L, K in cases:
        score_f32 = make_score(B, L)
        score_bf16 = score_f32.to(torch.bfloat16)

        t_bf16 = bench(lambda: torch.topk(score_bf16, K, dim=-1, sorted=False))
        t_f32 = bench(lambda: torch.topk(score_f32, K, dim=-1, sorted=False))
        t_fast = bench(lambda: fast_topk_runtime(score_f32, K))
        sp = t_bf16 / t_fast
        print(f"{label:>10} {B:>5} {L:>6} {K:>5} | "
              f"{t_bf16:>10.1f} {t_f32:>10.1f} {t_fast:>10.1f} | {sp:>7.2f}x")


def correctness_cuda_graph():
    """Regression: capturing fast_topk_runtime into a fresh CUDA graph
    must succeed. The .cu deliberately skips ``cudaFuncSetAttribute``
    (kSmem=32KB <= sm_70+ default 48KB) so capture is safe with no
    eager-warmup hack required.
    """
    print()
    print("=" * 80)
    print("Correctness — CUDA graph capture")
    print("=" * 80)
    score = torch.randn(1, 8192, device=DEVICE, dtype=torch.float32)
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fast_topk_runtime(score, 512)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(g):
            fast_topk_runtime(score, 512)
        g.replay()
        torch.cuda.synchronize()
        print("  capture + replay  | True")
        return 0
    except Exception as e:
        print(f"  capture + replay  | False  ({type(e).__name__}: {e})")
        return 1


def main():
    fails = 0
    fails += correctness_full_row()
    fails += correctness_inf_masking()
    fails += correctness_cuda_graph()
    print()
    print(f"{'TOTAL CORRECTNESS FAILURES:':>30} {fails}")
    if fails == 0:
        print("  ALL_CORRECTNESS_OK")
    speed_full_row()


if __name__ == "__main__":
    main()
