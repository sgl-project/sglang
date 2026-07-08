"""
Phase 3-4: Benchmark prefill attention kernel variants.

Sweep: num_warps in {4,8,16}, num_stages in {1,2,3}, BLOCK_M in {64,128}, BLOCK_N in {64,128}
Sizes: seq_len in {128,512,2048,8192}, num_heads in {4,32}, head_dim in {64,128}
"""
import json
import os
import re
import shutil
import itertools
from pathlib import Path

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_gpu_1"

import torch
import triton
import triton.language as tl

# Import the kernel
from sglang.srt.layers.attention.triton_ops.prefill_attention import _fwd_kernel


def run_kernel(q, k, v, o, b_start_loc, b_seq_len, max_input_len,
               is_causal, sm_scale, num_warps, num_stages, BLOCK_M, BLOCK_N):
    """Launch _fwd_kernel with specified config."""
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    BLOCK_DMODEL = triton.next_power_of_2(Lk)

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))

    _fwd_kernel[grid](
        q, k, v, sm_scale,
        b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
        Lk=Lk,
    )


def create_inputs(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.float16):
    """Create test inputs for a single batch."""
    batch = 1
    kv_heads = num_heads  # MHA
    total_tokens = batch * seq_len

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, kv_heads, head_dim, dtype=dtype, device=device)
    o = torch.zeros_like(q)

    b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)

    return q, k, v, o, b_start_loc, b_seq_len, sm_scale


def check_correctness(seq_len, num_heads, head_dim, num_warps, num_stages, BLOCK_M, BLOCK_N):
    """Check variant correctness against baseline."""
    q, k, v, o_ref, b_start_loc, b_seq_len, sm_scale = create_inputs(seq_len, num_heads, head_dim)
    o_test = torch.zeros_like(o_ref)

    # Baseline: num_warps=8, num_stages=1, BLOCK=128
    run_kernel(q, k, v, o_ref, b_start_loc, b_seq_len, seq_len,
               True, sm_scale, 8, 1, 128, 128)

    # Variant
    run_kernel(q, k, v, o_test, b_start_loc, b_seq_len, seq_len,
               True, sm_scale, num_warps, num_stages, BLOCK_M, BLOCK_N)

    torch.cuda.synchronize()

    close = torch.allclose(o_ref, o_test, atol=1e-2, rtol=1e-2)
    max_diff = (o_ref.float() - o_test.float()).abs().max().item()
    return close, max_diff


def benchmark_variant(seq_len, num_heads, head_dim, num_warps, num_stages, BLOCK_M, BLOCK_N):
    """Benchmark a specific variant."""
    q, k, v, o, b_start_loc, b_seq_len, sm_scale = create_inputs(seq_len, num_heads, head_dim)

    def fn():
        o.zero_()
        run_kernel(q, k, v, o, b_start_loc, b_seq_len, seq_len,
                   True, sm_scale, num_warps, num_stages, BLOCK_M, BLOCK_N)

    # Warmup and benchmark
    ms = triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median")
    return ms


def main():
    device = "cuda:0"
    results = []

    # Variant configurations
    configs = list(itertools.product(
        [4, 8, 16],      # num_warps
        [1, 2, 3],        # num_stages
        [64, 128],         # BLOCK_M
        [64, 128],         # BLOCK_N
    ))

    # Benchmark sizes
    sizes = list(itertools.product(
        [128, 512, 2048, 8192],  # seq_len
        [4, 32],                  # num_heads
        [64, 128],                # head_dim
    ))

    print(f"Total configs: {len(configs)}, Total sizes: {len(sizes)}")
    print(f"Total benchmarks: {len(configs) * len(sizes)}")

    # First pass: correctness check on a representative size
    print("\n" + "=" * 60)
    print("CORRECTNESS CHECK (seq=512, heads=32, dim=128)")
    print("=" * 60)
    valid_configs = []
    for num_warps, num_stages, BLOCK_M, BLOCK_N in configs:
        name = f"w{num_warps}_s{num_stages}_m{BLOCK_M}_n{BLOCK_N}"
        try:
            close, max_diff = check_correctness(512, 32, 128, num_warps, num_stages, BLOCK_M, BLOCK_N)
            status = "PASS" if close else f"FAIL (max_diff={max_diff:.6f})"
            if close:
                valid_configs.append((num_warps, num_stages, BLOCK_M, BLOCK_N))
            print(f"  {name}: {status}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    print(f"\nValid configs: {len(valid_configs)} / {len(configs)}")

    # Also check correctness with head_dim=64
    print("\n" + "=" * 60)
    print("CORRECTNESS CHECK (seq=512, heads=32, dim=64)")
    print("=" * 60)
    valid_configs_64 = []
    for num_warps, num_stages, BLOCK_M, BLOCK_N in configs:
        name = f"w{num_warps}_s{num_stages}_m{BLOCK_M}_n{BLOCK_N}"
        try:
            close, max_diff = check_correctness(512, 32, 64, num_warps, num_stages, BLOCK_M, BLOCK_N)
            status = "PASS" if close else f"FAIL (max_diff={max_diff:.6f})"
            if close:
                valid_configs_64.append((num_warps, num_stages, BLOCK_M, BLOCK_N))
            print(f"  {name}: {status}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Benchmark only valid configs
    print("\n" + "=" * 60)
    print("BENCHMARKING")
    print("=" * 60)

    # Identify baseline config
    baseline_key = "w8_s1_m128_n128"

    for seq_len, num_heads, head_dim in sizes:
        print(f"\n--- seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim} ---")

        # Use the appropriate valid config set
        use_configs = valid_configs if head_dim == 128 else valid_configs_64

        for num_warps, num_stages, BLOCK_M, BLOCK_N in use_configs:
            name = f"w{num_warps}_s{num_stages}_m{BLOCK_M}_n{BLOCK_N}"
            try:
                ms = benchmark_variant(seq_len, num_heads, head_dim, num_warps, num_stages, BLOCK_M, BLOCK_N)
                result = {
                    "config": name,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                    "BLOCK_M": BLOCK_M,
                    "BLOCK_N": BLOCK_N,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "time_ms": round(ms, 4),
                }
                results.append(result)
                print(f"  {name}: {ms:.4f} ms")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")

    # Save results
    out_dir = Path(__file__).parent
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Analysis: find best config per size
    print("\n" + "=" * 60)
    print("BEST CONFIG PER SIZE")
    print("=" * 60)

    best_overall = {}
    for seq_len, num_heads, head_dim in sizes:
        size_key = f"seq={seq_len}_heads={num_heads}_dim={head_dim}"
        size_results = [r for r in results
                        if r["seq_len"] == seq_len
                        and r["num_heads"] == num_heads
                        and r["head_dim"] == head_dim]
        if not size_results:
            continue

        baseline_results = [r for r in size_results if r["config"] == baseline_key]
        baseline_ms = baseline_results[0]["time_ms"] if baseline_results else None

        best = min(size_results, key=lambda r: r["time_ms"])
        speedup = baseline_ms / best["time_ms"] if baseline_ms else 0

        print(f"  {size_key}")
        print(f"    Baseline ({baseline_key}): {baseline_ms:.4f} ms" if baseline_ms else f"    Baseline: N/A")
        print(f"    Best: {best['config']} = {best['time_ms']:.4f} ms (speedup: {speedup:.2f}x)")

        best_overall[size_key] = {
            "best_config": best["config"],
            "best_ms": best["time_ms"],
            "baseline_ms": baseline_ms,
            "speedup": speedup,
        }

    # Overall winner (geometric mean speedup)
    print("\n" + "=" * 60)
    print("OVERALL ANALYSIS")
    print("=" * 60)

    # Count how often each config wins
    win_counts = {}
    for info in best_overall.values():
        cfg = info["best_config"]
        win_counts[cfg] = win_counts.get(cfg, 0) + 1

    print("Win counts:")
    for cfg, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {cfg}: {count} wins")

    # Compute average speedup per config
    config_speedups = {}
    for r in results:
        cfg = r["config"]
        size_key = f"seq={r['seq_len']}_heads={r['num_heads']}_dim={r['head_dim']}"
        baseline = [x for x in results
                    if x["config"] == baseline_key
                    and x["seq_len"] == r["seq_len"]
                    and x["num_heads"] == r["num_heads"]
                    and x["head_dim"] == r["head_dim"]]
        if baseline:
            speedup = baseline[0]["time_ms"] / r["time_ms"]
            if cfg not in config_speedups:
                config_speedups[cfg] = []
            config_speedups[cfg].append(speedup)

    print("\nAverage speedup vs baseline:")
    import statistics
    for cfg, speedups in sorted(config_speedups.items(), key=lambda x: -statistics.geometric_mean(x[1])):
        gmean = statistics.geometric_mean(speedups)
        print(f"  {cfg}: {gmean:.3f}x geomean (min={min(speedups):.3f}x, max={max(speedups):.3f}x)")


if __name__ == "__main__":
    main()
