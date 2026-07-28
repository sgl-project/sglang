#!/usr/bin/env python3
"""Microbenchmark for vectorized DSA _cal_indexer_k_start_end.

Compares the original per-request Python loop (3 kernel launches per request)
against the vectorized implementation (~5 total kernel launches).

Usage:
    python scripts/bench_glm52_dsa_hotpath.py
"""

import argparse
import random
import sys
import time
import torch

sys.path.insert(0, "python")


def reference_cal_indexer(extend_lens_list, seq_lens_list, device, draft_tokens=0):
    """Original per-request loop logic."""
    ks_list = []
    ke_list = []
    token_to_batch_idx = []
    k_offset = 0
    for i in range(len(extend_lens_list)):
        sl = seq_lens_list[i]
        el = extend_lens_list[i]
        kv_len = sl + draft_tokens
        ks = torch.full((el,), k_offset, dtype=torch.int32, device=device)
        seq_lens_expanded = torch.arange(
            kv_len - el + 1, kv_len + 1, dtype=torch.int32, device=device
        )
        ke = ks + seq_lens_expanded
        ks_list.append(ks)
        ke_list.append(ke)
        tb = torch.full((el,), i, dtype=torch.int32, device=device)
        token_to_batch_idx.append(tb)
        k_offset += sl + draft_tokens
    ks = torch.cat(ks_list, dim=0)
    ke = torch.cat(ke_list, dim=0)
    token_to_batch_idx = torch.cat(token_to_batch_idx, dim=0)
    return ks, ke, token_to_batch_idx


def vectorized_cal_indexer(extend_lens_list, seq_lens_list, device, draft_tokens=0):
    """Vectorized implementation."""
    bs = len(extend_lens_list)
    extend_lens = torch.tensor(extend_lens_list, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    if draft_tokens > 0:
        seq_lens = seq_lens + draft_tokens

    k_offsets = torch.zeros(bs, dtype=torch.int32, device=device)
    if bs > 1:
        k_offsets[1:] = torch.cumsum(seq_lens[:-1], dim=0).to(torch.int32)
    ks = torch.repeat_interleave(k_offsets, extend_lens)

    kv_minus_extend = seq_lens - extend_lens
    arange_starts = kv_minus_extend + 1
    total_len = extend_lens.sum().item()

    intra_offsets = torch.arange(total_len, dtype=torch.int32, device=device)
    cumsum_extend = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.cumsum(extend_lens[:-1], dim=0).to(torch.int32),
    ])
    cumsum_expanded = torch.repeat_interleave(cumsum_extend, extend_lens)
    intra_pos = (intra_offsets - cumsum_expanded).to(torch.int32)
    arange_start_expanded = torch.repeat_interleave(arange_starts, extend_lens)
    ke = ks + arange_start_expanded + intra_pos

    batch_indices = torch.arange(bs, dtype=torch.int32, device=device)
    token_to_batch_idx = torch.repeat_interleave(batch_indices, extend_lens)
    return ks, ke, token_to_batch_idx


def benchmark_impl(fn, extend_lens, seq_lens, device, warmup=5, iters=50):
    """Benchmark a function with CUDA events."""
    for _ in range(warmup):
        ks, ke, tb = fn(extend_lens, seq_lens, device)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        ks, ke, tb = fn(extend_lens, seq_lens, device)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return median, p95, times


def validate_correctness(fn_ref, fn_opt, extend_lens, seq_lens, device):
    """Validate that both implementations produce the same output."""
    ks_ref, ke_ref, tb_ref = fn_ref(extend_lens, seq_lens, device)
    ks_opt, ke_opt, tb_opt = fn_opt(extend_lens, seq_lens, device)
    assert ks_ref.tolist() == ks_opt.tolist(), f"ks mismatch"
    assert ke_ref.tolist() == ke_opt.tolist(), f"ke mismatch"
    assert tb_ref.tolist() == tb_opt.tolist(), f"tb mismatch"
    return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark DSA indexer vectorization")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"DSA Indexer Vectorization Microbenchmark")
    print(f"Device: {device}")
    print(f"{'='*90}")

    # Representative GLM-5.2 shapes
    test_cases = [
        ("decode_bs1", [1], [32000]),
        ("decode_bs4", [1, 1, 1, 1], [32000, 64000, 128000, 150000]),
        ("decode_bs16", [1] * 16, [random.randint(32000, 150000) for _ in range(16)]),
        ("decode_bs32", [1] * 32, [random.randint(32000, 150000) for _ in range(32)]),
        ("decode_bs64", [1] * 64, [random.randint(32000, 150000) for _ in range(64)]),
        ("prefill_256", [256], [65536]),
        ("prefill_1024", [1024], [65536]),
        ("prefill_4096", [4096], [131072]),
        ("prefill_16384", [16384], [150000]),
        ("mixed_4req", [256, 1024, 1, 1], [64000, 128000, 32000, 150000]),
    ]

    random.seed(42)
    print(f"{'Test Case':<20} {'Batch':>6} {'Tokens':>8} {'Ref Med(ms)':>14} {'Vec Med(ms)':>14} {'Speedup':>8} {'Ref P95':>10} {'Vec P95':>10}")
    print("-" * 90)

    for name, extend_lens, seq_lens in test_cases:
        bs = len(extend_lens)
        total_tokens = sum(extend_lens)

        # Validate correctness
        try:
            validate_correctness(reference_cal_indexer, vectorized_cal_indexer, extend_lens, seq_lens, device)
        except Exception as e:
            print(f"{name:<20} CORRECTNESS FAILED: {e}")
            continue

        # Benchmark
        ref_med, ref_p95, _ = benchmark_impl(reference_cal_indexer, extend_lens, seq_lens, device)
        vec_med, vec_p95, _ = benchmark_impl(vectorized_cal_indexer, extend_lens, seq_lens, device)

        speedup = ref_med / vec_med if vec_med > 0 else float('inf')
        print(f"{name:<20} {bs:>6} {total_tokens:>8} {ref_med:>14.4f} {vec_med:>14.4f} {speedup:>7.2f}x {ref_p95:>10.4f} {vec_p95:>10.4f}")

    # Also measure kernel launch count difference
    print(f"\n{'='*90}")
    print("Kernel launch count analysis (using torch.profiler):")
    print(f"{'='*90}")

    for name, extend_lens, seq_lens in [("decode_bs64", [1]*64, [64000]*64), ("prefill_4096", [4096], [131072])]:
        from torch.profiler import profile, ProfilerActivity

        # Reference
        with profile(activities=[ProfilerActivity.CUDA]) as prof_ref:
            for _ in range(10):
                reference_cal_indexer(extend_lens, seq_lens, device)
            torch.cuda.synchronize()
        ref_ops = len([e for e in prof_ref.events() if e.device_type == "cuda"])

        with profile(activities=[ProfilerActivity.CUDA]) as prof_vec:
            for _ in range(10):
                vectorized_cal_indexer(extend_lens, seq_lens, device)
            torch.cuda.synchronize()
        vec_ops = len([e for e in prof_vec.events() if e.device_type == "cuda"])

        print(f"  {name}: ref={ref_ops} cuda events, vec={vec_ops} cuda events")

    print(f"\n{'='*90}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
