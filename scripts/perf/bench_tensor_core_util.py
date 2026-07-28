#!/usr/bin/env python3
"""Tensor Core microbenchmark for H100 PCIe.

Compares FP32 (no TF32), TF32, BF16, FP16, and FP8 GEMM performance
across decode-like (small M) and prefill-like (large M) shapes.

Methodology (v3 — corrected):
- Per-shape FP32 baseline: each dtype compared against FP32 at the SAME M, N, K
  (never reuses M=1 baseline for other M values)
- CUDA events for GPU timing (not perf_counter)
- Multiple independent trials with median-of-medians
- p50/p95/p99 latency and coefficient of variation
- Hot / rotating / flushed cache modes
- No allocation or I/O inside timed region
- Reports std, CV, and contamination status

Usage:
    python scripts/perf/bench_tensor_core_util.py \
        --device cuda:0 --dtypes fp32_no_tf32,tf32,bfloat16,float16,fp8_torch \
        --m-values 1,4,16,64,128,256,512,1024 --n 4096 --k 4096 \
        --iterations 100 --trials 5 --cache-mode hot \
        --output-json /tmp/glm52_tensor_core.json
"""
import argparse, json, os, statistics, math, sys
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Tensor Core benchmark")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtypes", default="fp32_no_tf32,tf32,bfloat16,float16")
    p.add_argument("--m-values", default="1,4,8,16,32,64,128,256,512,1024")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--trials", type=int, default=5, help="Independent trials; final = median of trial medians")
    p.add_argument("--cache-mode", default="hot", choices=["hot", "rotating", "flushed"])
    p.add_argument("--num-rotating", type=int, default=8, help="Number of rotating B matrices")
    p.add_argument("--contamination", default="LIGHTLY_CONTENDED")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


DTYPE_MAP = {
    "fp32_no_tf32": (torch.float32, False),
    "tf32": (torch.float32, True),
    "bfloat16": (torch.bfloat16, False),
    "float16": (torch.float16, False),
}


def compute_stats(latencies_us, flops):
    """Compute latency stats from a list of latency values in microseconds."""
    if not latencies_us:
        return None
    s = sorted(latencies_us)
    n = len(s)
    median = s[n // 2]
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    p95 = s[p95_idx] if n > 1 else s[0]
    p99 = s[p99_idx] if n > 1 else s[0]
    mean = sum(s) / n
    if n > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in s) / (n - 1))
        cv = std / mean if mean > 0 else 0
    else:
        std = 0
        cv = 0
    tflops = flops / median * 1e-6 if median > 0 else 0  # flops / (us * 1e-6) = TFLOP/s

    # Suspicious result detection
    warnings = []
    if median <= 0:
        warnings.append("SUSPICIOUS_RESULT: zero or negative median latency")
    if n > 1 and p95 < median:
        warnings.append("SUSPICIOUS_RESULT: p95 below median")
    if n > 1 and p99 < p95:
        warnings.append("SUSPICIOUS_RESULT: p99 below p95")
    # H100 PCIe theoretical FP8 peak ~3958 TFLOP/s, BF16 ~989 TFLOP/s
    # Use a generous ceiling
    if tflops > 5000:
        warnings.append(f"SUSPICIOUS_RESULT: TFLOP/s={tflops:.1f} exceeds generous hardware ceiling")

    return {
        "median_latency_us": median,
        "p95_latency_us": p95,
        "p99_latency_us": p99,
        "mean_latency_us": mean,
        "std_us": std,
        "cv": cv,
        "tflops": tflops,
        "iterations": n,
        "warnings": warnings,
    }


def benchmark_gemm(M, N, K, dtype, allow_tf32, device, warmup, iters, cache_mode, num_rotating):
    """Run GEMM benchmark with CUDA event timing and per-shape baseline."""
    a = torch.randn(M, K, dtype=torch.float32, device=device)
    b = torch.randn(K, N, dtype=torch.float32, device=device)

    # Cast to target dtype outside timing
    a_t = a.to(dtype)
    b_t = b.to(dtype)

    # Reference for correctness
    ref = a @ b

    # Prepare rotating weights
    if cache_mode == "rotating":
        b_list = [torch.randn(K, N, dtype=dtype, device=device) for _ in range(num_rotating)]
        rotating_total_bytes = sum(b.nelement() * b.element_size() for b in b_list)
    else:
        b_list = [b_t]
        rotating_total_bytes = b_t.nelement() * b_t.element_size()

    # Flush buffer for flushed mode (~60 MiB > H100 L2 of 50 MiB)
    flush_buf = None
    if cache_mode == "flushed":
        flush_buf = torch.zeros(60 * 1024 * 1024 // 4, dtype=torch.float32, device=device)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # Warmup
    for i in range(warmup):
        idx = i % len(b_list)
        c = a_t @ b_list[idx]
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    latencies = []
    for i in range(iters):
        idx = i % len(b_list)

        if flush_buf is not None:
            # Touch flush buffer to evict L2 (outside timing)
            flush_buf.fill_(float(i))
            torch.cuda.synchronize()

        start.record()
        c = a_t @ b_list[idx]
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)  # ms -> us

    torch.backends.cuda.matmul.allow_tf32 = old_tf32

    # Correctness
    out = a_t @ b_list[0]
    max_err = (out.float() - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-9)

    flops = 2 * M * N * K
    stats = compute_stats(latencies, flops)
    stats["max_error"] = max_err
    stats["rel_error"] = rel_err
    stats["M"] = M
    stats["N"] = N
    stats["K"] = K
    stats["cache_mode"] = cache_mode
    stats["num_rotating"] = len(b_list) if cache_mode == "rotating" else 1
    stats["rotating_total_bytes"] = rotating_total_bytes
    stats["rotating_total_mib"] = rotating_total_bytes / (1024 * 1024)
    stats["tf32_enabled"] = allow_tf32
    stats["backend"] = "torch.matmul"
    stats["warmup_iterations"] = warmup
    stats["timed_iterations"] = iters
    stats["independent_trials"] = 1  # per-trial; outer loop sets actual count
    return stats


def benchmark_fp8_torch(M, N, K, device, warmup, iters, cache_mode, num_rotating):
    """FP8 GEMM via torch._scaled_mm."""
    try:
        a = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(K, N, device=device).to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, device=device)

        # Reference
        ref = (a.float() @ b.float())

        if cache_mode == "rotating":
            b_list = [torch.randn(K, N, device=device).to(torch.float8_e4m3fn) for _ in range(num_rotating)]
            rotating_total_bytes = sum(b.nelement() * b.element_size() for b in b_list)
        else:
            b_list = [b]
            rotating_total_bytes = b.nelement() * b.element_size()

        flush_buf = None
        if cache_mode == "flushed":
            flush_buf = torch.zeros(60 * 1024 * 1024 // 4, dtype=torch.float32, device=device)

        # Warmup
        for i in range(warmup):
            idx = i % len(b_list)
            c = torch._scaled_mm(a, b_list[idx].t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        latencies = []
        for i in range(iters):
            idx = i % len(b_list)
            if flush_buf is not None:
                flush_buf.fill_(float(i))
                torch.cuda.synchronize()

            start.record()
            c = torch._scaled_mm(a, b_list[idx].t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end) * 1000)

        flops = 2 * M * N * K
        stats = compute_stats(latencies, flops)
        # Correctness
        out = torch._scaled_mm(a, b_list[0].t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
        max_err = (out.float() - ref).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-9)
        stats["max_error"] = max_err
        stats["rel_error"] = rel_err
        stats["M"] = M
        stats["N"] = N
        stats["K"] = K
        stats["cache_mode"] = cache_mode
        stats["num_rotating"] = len(b_list) if cache_mode == "rotating" else 1
        stats["rotating_total_bytes"] = rotating_total_bytes
        stats["rotating_total_mib"] = rotating_total_bytes / (1024 * 1024)
        stats["tf32_enabled"] = False
        stats["backend"] = "torch._scaled_mm"
        stats["warmup_iterations"] = warmup
        stats["timed_iterations"] = iters
        stats["independent_trials"] = 1
        return stats
    except Exception as e:
        return {"M": M, "N": N, "K": K, "error": str(e), "cache_mode": cache_mode}


def main():
    args = parse_args()
    device = torch.device(args.device)
    m_values = [int(x) for x in args.m_values.split(",")]
    dtypes = args.dtypes.split(",")
    gpu_name = torch.cuda.get_device_name(device)
    contamination = args.contamination

    print(f"GPU: {gpu_name}")
    print(f"N={args.n}, K={args.k}, cache_mode={args.cache_mode}, trials={args.trials}")

    # Phase 1: Run FP32 baseline for every shape
    fp32_baseline = {}
    if "fp32_no_tf32" in dtypes:
        for M in m_values:
            trial_medians = []
            for t in range(args.trials):
                r = benchmark_gemm(M, args.n, args.k, torch.float32, False, device,
                                   args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                trial_medians.append(r["median_latency_us"])
            med = sorted(trial_medians)[len(trial_medians) // 2]
            fp32_baseline[(M, args.n, args.k)] = med
            print(f"  fp32_no_tf32    M={M:>5d}  median={med:.2f}us  TFLOP/s={2*M*args.n*args.k/med*1e-6:.1f}")

    all_results = []
    for dt in dtypes:
        for M in m_values:
            trial_results = []
            for t in range(args.trials):
                if dt == "fp32_no_tf32":
                    r = benchmark_gemm(M, args.n, args.k, torch.float32, False, device,
                                       args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                elif dt == "tf32":
                    r = benchmark_gemm(M, args.n, args.k, torch.float32, True, device,
                                       args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                elif dt == "bfloat16":
                    r = benchmark_gemm(M, args.n, args.k, torch.bfloat16, False, device,
                                       args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                elif dt == "float16":
                    r = benchmark_gemm(M, args.n, args.k, torch.float16, False, device,
                                       args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                elif dt == "fp8_torch":
                    r = benchmark_fp8_torch(M, args.n, args.k, device,
                                            args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                else:
                    continue
                trial_results.append(r)

            if not trial_results:
                continue

            # Aggregate across trials: median of trial medians
            valid = [r for r in trial_results if "error" not in r]
            if not valid:
                r = trial_results[0]
                r["dtype"] = dt
                r["speedup_vs_fp32"] = "NOT_AVAILABLE"
                all_results.append(r)
                print(f"  {dt:15s} M={M:>5d}  ERROR: {r.get('error','')}")
                continue

            trial_medians = sorted([r["median_latency_us"] for r in valid])
            median_of_medians = trial_medians[len(trial_medians) // 2]
            p95_vals = sorted([r["p95_latency_us"] for r in valid])
            p99_vals = sorted([r["p99_latency_us"] for r in valid])
            mean_vals = [r["mean_latency_us"] for r in valid]

            flops = 2 * M * args.n * args.k
            tflops = flops / median_of_medians * 1e-6 if median_of_medians > 0 else 0
            base = fp32_baseline.get((M, args.n, args.k))
            speedup = base / median_of_medians if base else 1.0

            result = {
                "dtype": dt,
                "backend": valid[0].get("backend", "torch.matmul"),
                "tf32_enabled": valid[0].get("tf32_enabled", False),
                "M": M, "N": args.n, "K": args.k,
                "batch_or_group_count": M,
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "independent_trials": args.trials,
                "median_latency_us": median_of_medians,
                "p95_latency_us": p95_vals[len(p95_vals) // 2],
                "p99_latency_us": p99_vals[len(p99_vals) // 2],
                "mean_latency_us": sum(mean_vals) / len(mean_vals),
                "std_us": sorted([r["std_us"] for r in valid])[len(valid) // 2],
                "cv": sorted([r["cv"] for r in valid])[len(valid) // 2],
                "tflops": tflops,
                "speedup_vs_fp32_same_shape": speedup,
                "max_error": valid[0]["max_error"],
                "rel_error": valid[0]["rel_error"],
                "cache_mode": args.cache_mode,
                "rotating_total_mib": valid[0].get("rotating_total_mib", 0),
                "num_rotating": valid[0].get("num_rotating", 1),
                "gpu": gpu_name,
                "contamination": contamination,
                "warnings": valid[0].get("warnings", []),
            }
            all_results.append(result)
            print(f"  {dt:15s} M={M:>5d}  median={result['median_latency_us']:.2f}us  "
                  f"TFLOP/s={result['tflops']:.1f}  speedup={result['speedup_vs_fp32_same_shape']:.2f}x  "
                  f"cv={valid[0]['cv']:.3f}")

    if args.output_json:
        out = {
            "gpu_name": gpu_name,
            "N": args.n, "K": args.k,
            "cache_mode": args.cache_mode,
            "trials": args.trials,
            "results": all_results,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
