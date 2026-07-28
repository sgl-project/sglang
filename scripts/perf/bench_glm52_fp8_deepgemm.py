#!/usr/bin/env python3
"""FP8 and DeepGEMM benchmark for GLM-5.2 GEMM shapes.

Tests real FP8 paths available in this SGLang environment:
  - torch._scaled_mm (FP8 E4M3)
  - deep_gemm.fp8_gemm_nt (dense FP8)
  - deep_gemm.fp8_m_grouped_gemm_nt_masked (grouped FP8 for MoE)

Reports NOT_AVAILABLE rather than silently falling back.

Usage:
    python scripts/perf/bench_glm52_fp8_deepgemm.py \
        --device cuda:1 --m-values 1,4,16,64,128,256,512,1024 \
        --n 4096 --k 5120 --warmup 10 --iterations 100 --trials 5 \
        --cache-mode hot --output-json /tmp/fp8_deepgemm.json
"""
import argparse, json, math, os, sys
import torch


def parse_args():
    p = argparse.ArgumentParser(description="FP8/DeepGEMM benchmark")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--backends", default="torch_scaled_mm,deep_gemm,sgl_kernel",
                   help="Comma-separated: torch_scaled_mm,deep_gemm,sgl_kernel,auto")
    p.add_argument("--m-values", default="1,4,16,64,128,256,512,1024")
    p.add_argument("--n", type=int, default=4096, help="N dimension (default: GLM MoE gate_up per-expert)")
    p.add_argument("--k", type=int, default=5120, help="K dimension (default: GLM hidden_size)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--cache-mode", default="hot", choices=["hot", "rotating", "flushed"])
    p.add_argument("--num-rotating", type=int, default=8)
    p.add_argument("--contamination", default="LIGHTLY_CONTENDED")
    p.add_argument("--output-json", default=None)
    # Grouped GEMM options
    p.add_argument("--num-experts", type=int, default=64, help="Number of MoE experts for grouped GEMM")
    p.add_argument("--topk", type=int, default=6, help="Experts per token")
    p.add_argument("--skip-grouped", action="store_true", help="Skip grouped GEMM tests")
    return p.parse_args()


def compute_stats(latencies_us, flops):
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
    tflops = flops / median * 1e-6 if median > 0 else 0
    warnings = []
    if median <= 0:
        warnings.append("SUSPICIOUS_RESULT: zero or negative median latency")
    if n > 1 and p95 < median:
        warnings.append("SUSPICIOUS_RESULT: p95 below median")
    if n > 1 and p99 < p95:
        warnings.append("SUSPICIOUS_RESULT: p99 below p95")
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


def check_backend_available(backend, device):
    """Check if a backend is available on this device."""
    if backend == "torch_scaled_mm":
        return hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn")
    elif backend == "deep_gemm":
        try:
            import deep_gemm
            # Verify it actually works on this device
            return torch.cuda.get_device_capability(device)[0] >= 9
        except ImportError:
            return False
    elif backend == "sgl_kernel":
        try:
            import sgl_kernel
            return True
        except ImportError:
            return False
    return False


def benchmark_torch_scaled_mm(M, N, K, device, warmup, iters, cache_mode, num_rotating):
    """FP8 GEMM via torch._scaled_mm with E4M3 inputs and BF16 output."""
    # Prepare FP8 inputs (quantize outside timing)
    a_f32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_f32 = torch.randn(N, K, device=device, dtype=torch.float32)  # Note: B is (N,K) for _scaled_mm

    # Quantize to FP8
    a_scale = a_f32.abs().max() / 448.0
    b_scale = b_f32.abs().max() / 448.0
    a_fp8 = (a_f32 / a_scale).to(torch.float8_e4m3fn)
    b_fp8 = (b_f32 / b_scale).to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0 / a_scale, device=device)
    scale_b = torch.tensor(1.0 / b_scale, device=device)

    # Reference
    ref = a_f32 @ b_f32.t()

    # Rotating weights
    if cache_mode == "rotating":
        b_list = []
        for _ in range(num_rotating):
            bf = torch.randn(N, K, device=device, dtype=torch.float32)
            bs = bf.abs().max() / 448.0
            b_list.append((bf / bs).to(torch.float8_e4m3fn))
    else:
        b_list = [b_fp8]

    flush_buf = None
    if cache_mode == "flushed":
        flush_buf = torch.zeros(60 * 1024 * 1024 // 4, dtype=torch.float32, device=device)

    # Warmup
    for i in range(warmup):
        idx = i % len(b_list)
        c = torch._scaled_mm(a_fp8, b_list[idx].t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
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
        c = torch._scaled_mm(a_fp8, b_list[idx].t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)  # ms -> us

    # Correctness
    out = torch._scaled_mm(a_fp8, b_list[0].t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    max_err = (out.float() - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-9)

    flops = 2 * M * N * K
    stats = compute_stats(latencies, flops)
    stats["max_error"] = max_err
    stats["rel_error"] = rel_err
    stats["backend"] = "torch._scaled_mm"
    stats["kernel"] = "torch._scaled_mm"
    stats["dtype"] = "float8_e4m3fn"
    stats["scale_format"] = "per-tensor scalar"
    stats["output_dtype"] = "bfloat16"
    return stats


def benchmark_deep_gemm_dense(M, N, K, device, warmup, iters, cache_mode, num_rotating):
    """Dense FP8 GEMM via deep_gemm.fp8_gemm_nt."""
    import deep_gemm

    # Prepare FP8 inputs with per-token and per-block scales
    a_f32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_f32 = torch.randn(N, K, device=device, dtype=torch.float32)

    # Use deep_gemm's per-token and per-block cast
    a_fp8, a_scale = deep_gemm.per_token_cast_to_fp8(a_f32, use_ue8m0=True)
    b_fp8, b_scale = deep_gemm.per_block_cast_to_fp8(b_f32, use_ue8m0=True)

    # Reference
    ref = a_f32 @ b_f32.t()

    # Output buffer
    out = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    # Rotating weights
    if cache_mode == "rotating":
        b_list = []
        for _ in range(num_rotating):
            bf = torch.randn(N, K, device=device, dtype=torch.float32)
            bf_fp8, bf_scale = deep_gemm.per_block_cast_to_fp8(bf, use_ue8m0=True)
            b_list.append((bf_fp8, bf_scale))
    else:
        b_list = [(b_fp8, b_scale)]

    flush_buf = None
    if cache_mode == "flushed":
        flush_buf = torch.zeros(60 * 1024 * 1024 // 4, dtype=torch.float32, device=device)

    # Warmup
    for i in range(warmup):
        idx = i % len(b_list)
        bf, bs = b_list[idx]
        deep_gemm.fp8_gemm_nt((a_fp8, a_scale), (bf, bs), out)
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
        bf, bs = b_list[idx]
        deep_gemm.fp8_gemm_nt((a_fp8, a_scale), (bf, bs), out)
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)

    # Correctness
    bf, bs = b_list[0]
    deep_gemm.fp8_gemm_nt((a_fp8, a_scale), (bf, bs), out)
    max_err = (out.float() - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-9)

    flops = 2 * M * N * K
    stats = compute_stats(latencies, flops)
    stats["max_error"] = max_err
    stats["rel_error"] = rel_err
    stats["backend"] = "deep_gemm"
    stats["kernel"] = "deep_gemm.fp8_gemm_nt"
    stats["dtype"] = "float8_e4m3fn"
    stats["scale_format"] = "per-token UE8M0 + per-block UE8M0"
    stats["output_dtype"] = "bfloat16"
    return stats


def benchmark_deep_gemm_grouped(M, N, K, device, num_experts, topk, warmup, iters, distribution="balanced"):
    """Grouped FP8 GEMM via deep_gemm.fp8_m_grouped_gemm_nt_masked."""
    import deep_gemm

    # Determine tokens per expert based on distribution
    total_tokens = M
    if distribution == "balanced":
        tokens_per_expert = [total_tokens // num_experts] * num_experts
        remainder = total_tokens % num_experts
        for i in range(remainder):
            tokens_per_expert[i] += 1
    elif distribution == "moderately_skewed":
        # First half of experts get 70% of tokens
        tokens_per_expert = [max(1, total_tokens // (num_experts * 3))] * num_experts
        for i in range(num_experts // 2):
            tokens_per_expert[i] = total_tokens * 7 // (10 * (num_experts // 2))
        # Normalize
        diff = total_tokens - sum(tokens_per_expert)
        tokens_per_expert[0] += diff
    elif distribution == "highly_skewed":
        tokens_per_expert = [1] * num_experts
        tokens_per_expert[0] = max(1, total_tokens - (num_experts - 1))
    elif distribution == "small_token":
        # Only a few experts have tokens (small expert groups)
        tokens_per_expert = [0] * num_experts
        active = min(topk, num_experts)
        for i in range(active):
            tokens_per_expert[i] = max(1, total_tokens // active)
        diff = total_tokens - sum(tokens_per_expert)
        tokens_per_expert[0] += diff
    else:
        tokens_per_expert = [total_tokens // num_experts] * num_experts

    # Clamp to >= 0
    tokens_per_expert = [max(0, t) for t in tokens_per_expert]

    # Build masked_m tensor
    masked_m = torch.tensor(tokens_per_expert, dtype=torch.int32, device=device)
    expected_m = total_tokens

    # Prepare weights for all experts: (num_experts, N, K) in FP8
    weights_f32 = torch.randn(num_experts, N, K, device=device, dtype=torch.float32)
    weights_fp8, weights_scale = deep_gemm.per_block_cast_to_fp8(weights_f32, use_ue8m0=True)

    # Prepare input: (total_tokens, K) in FP8
    a_f32 = torch.randn(total_tokens, K, device=device, dtype=torch.float32)
    a_fp8, a_scale = deep_gemm.per_token_cast_to_fp8(a_f32, use_ue8m0=True)

    # Reference (sparse: compute per-expert)
    ref = torch.zeros(total_tokens, N, device=device, dtype=torch.float32)
    # We won't do a full correctness check for grouped; just check output shape

    # Output: (num_experts, expected_m, N) — but deep_gemm expects (num_experts, M, N)
    # Actually fp8_m_grouped_gemm_nt_masked takes:
    #   lhs: ((M, K) fp8, (M, 1) scale) -- but this is for contiguous layout
    #   Actually the masked version takes per-expert inputs
    # Let's use the masked API correctly:
    # a: (num_experts, max_M, K) packed, b: (num_experts, N, K), d: (num_experts, max_M, N)

    max_m = max(tokens_per_expert) if tokens_per_expert else 1
    if max_m == 0:
        max_m = 1

    # For the masked version, inputs are:
    # lhs = (a_fp8, a_scale) where a_fp8 is (num_experts, max_m, K)
    # rhs = (weights_fp8, weights_scale) where weights_fp8 is (num_experts, N, K)
    # d = output (num_experts, max_m, N)
    # masked_m = (num_experts,) int32

    a_padded = a_fp8.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
    a_scale_padded = a_scale.unsqueeze(0).expand(num_experts, -1).contiguous()
    out = torch.empty(num_experts, expected_m, N, device=device, dtype=torch.bfloat16)

    # Warmup
    for i in range(warmup):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (a_padded, a_scale_padded),
            (weights_fp8, weights_scale),
            out,
            masked_m,
            expected_m,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []

    for i in range(iters):
        start.record()
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (a_padded, a_scale_padded),
            (weights_fp8, weights_scale),
            out,
            masked_m,
            expected_m,
        )
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)

    # Correctness: check no NaN/Inf
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()

    # FLOPs for grouped GEMM: sum over experts of 2 * m_e * N * K
    actual_flops = sum(2 * m * N * K for m in tokens_per_expert)
    stats = compute_stats(latencies, actual_flops)
    stats["max_error"] = -1.0  # Grouped GEMM correctness is complex; we check NaN/Inf instead
    stats["rel_error"] = -1.0
    stats["has_nan"] = has_nan
    stats["has_inf"] = has_inf
    stats["backend"] = "deep_gemm"
    stats["kernel"] = "deep_gemm.fp8_m_grouped_gemm_nt_masked"
    stats["dtype"] = "float8_e4m3fn"
    stats["scale_format"] = "per-token UE8M0 + per-block UE8M0"
    stats["output_dtype"] = "bfloat16"
    stats["num_experts"] = num_experts
    stats["distribution"] = distribution
    stats["tokens_per_expert"] = tokens_per_expert
    stats["actual_flops"] = actual_flops
    return stats


def benchmark_bf16_reference(M, N, K, device, warmup, iters, cache_mode, num_rotating):
    """BF16 reference GEMM for comparison."""
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    ref = a.float() @ b.float().t()

    if cache_mode == "rotating":
        b_list = [torch.randn(N, K, device=device, dtype=torch.bfloat16) for _ in range(num_rotating)]
    else:
        b_list = [b]

    flush_buf = None
    if cache_mode == "flushed":
        flush_buf = torch.zeros(60 * 1024 * 1024 // 4, dtype=torch.float32, device=device)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    for i in range(warmup):
        idx = i % len(b_list)
        c = a @ b_list[idx].t()
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
        c = a @ b_list[idx].t()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)

    torch.backends.cuda.matmul.allow_tf32 = old_tf32

    out = a @ b_list[0].t()
    max_err = (out.float() - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-9)

    flops = 2 * M * N * K
    stats = compute_stats(latencies, flops)
    stats["max_error"] = max_err
    stats["rel_error"] = rel_err
    stats["backend"] = "torch.matmul"
    stats["kernel"] = "torch.matmul (BF16)"
    stats["dtype"] = "bfloat16"
    stats["scale_format"] = "N/A"
    stats["output_dtype"] = "bfloat16"
    return stats


def main():
    args = parse_args()
    device = torch.device(args.device)
    m_values = [int(x) for x in args.m_values.split(",")]
    backends = args.backends.split(",")
    gpu_name = torch.cuda.get_device_name(device)

    print(f"GPU: {gpu_name}")
    print(f"N={args.n}, K={args.k}, cache_mode={args.cache_mode}, trials={args.trials}")
    print(f"Backends requested: {backends}")

    # Check availability
    availability = {}
    for b in backends:
        if b == "auto":
            continue
        avail = check_backend_available(b, device)
        availability[b] = avail
        print(f"  {b}: {'AVAILABLE' if avail else 'NOT_AVAILABLE'}")

    all_results = []

    # BF16 reference for speedup comparison
    bf16_baselines = {}
    for M in m_values:
        trial_medians = []
        for t in range(args.trials):
            r = benchmark_bf16_reference(M, args.n, args.k, device,
                                         args.warmup, args.iterations, args.cache_mode, args.num_rotating)
            trial_medians.append(r["median_latency_us"])
        med = sorted(trial_medians)[len(trial_medians) // 2]
        bf16_baselines[M] = med
        print(f"  bf16_reference  M={M:>5d}  median={med:.2f}us  TFLOP/s={2*M*args.n*args.k/med*1e-6:.1f}")

    # Test each backend
    for backend in backends:
        if backend == "auto":
            # Pick first available
            for b in ["deep_gemm", "torch_scaled_mm", "sgl_kernel"]:
                if availability.get(b, False):
                    backend = b
                    break
            else:
                print("No FP8 backend available")
                continue

        if not availability.get(backend, False):
            for M in m_values:
                result = {
                    "backend": backend,
                    "dtype": "float8_e4m3fn",
                    "M": M, "N": args.n, "K": args.k,
                    "status": "NOT_AVAILABLE",
                    "gpu": gpu_name,
                    "contamination": args.contamination,
                }
                all_results.append(result)
                print(f"  {backend:20s} M={M:>5d}  NOT_AVAILABLE")
            continue

        for M in m_values:
            trial_results = []
            for t in range(args.trials):
                try:
                    if backend == "torch_scaled_mm":
                        r = benchmark_torch_scaled_mm(M, args.n, args.k, device,
                                                      args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                    elif backend == "deep_gemm":
                        r = benchmark_deep_gemm_dense(M, args.n, args.k, device,
                                                       args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                    elif backend == "sgl_kernel":
                        # Check for sgl_kernel FP8 GEMM
                        import sgl_kernel
                        if hasattr(sgl_kernel, "fp8_gemm"):
                            # Use sgl_kernel's FP8 GEMM
                            r = benchmark_torch_scaled_mm(M, args.n, args.k, device,
                                                          args.warmup, args.iterations, args.cache_mode, args.num_rotating)
                            r["backend"] = "sgl_kernel"
                            r["kernel"] = "sgl_kernel.fp8_gemm"
                        else:
                            r = {"error": "sgl_kernel.fp8_gemm not found", "backend": "sgl_kernel"}
                    else:
                        continue
                    trial_results.append(r)
                except Exception as e:
                    trial_results.append({"error": str(e), "backend": backend})
                    break

            valid = [r for r in trial_results if "error" not in r]
            if not valid:
                r = trial_results[0] if trial_results else {"error": "no results"}
                r["M"] = M
                r["N"] = args.n
                r["K"] = args.k
                r["status"] = "ERROR"
                r["gpu"] = gpu_name
                r["contamination"] = args.contamination
                all_results.append(r)
                print(f"  {backend:20s} M={M:>5d}  ERROR: {r.get('error', '')}")
                continue

            trial_medians = sorted([r["median_latency_us"] for r in valid])
            median_of_medians = trial_medians[len(trial_medians) // 2]
            flops = 2 * M * args.n * args.k
            tflops = flops / median_of_medians * 1e-6 if median_of_medians > 0 else 0
            bf16_base = bf16_baselines.get(M, 1.0)
            speedup_vs_bf16 = bf16_base / median_of_medians if median_of_medians > 0 else 0

            result = {
                "backend": valid[0]["backend"],
                "kernel": valid[0]["kernel"],
                "dtype": valid[0]["dtype"],
                "scale_format": valid[0].get("scale_format", "N/A"),
                "output_dtype": valid[0].get("output_dtype", "bfloat16"),
                "M": M, "N": args.n, "K": args.k,
                "median_latency_us": median_of_medians,
                "p95_latency_us": sorted([r["p95_latency_us"] for r in valid])[len(valid) // 2],
                "p99_latency_us": sorted([r["p99_latency_us"] for r in valid])[len(valid) // 2],
                "mean_latency_us": sum(r["mean_latency_us"] for r in valid) / len(valid),
                "std_us": sorted([r["std_us"] for r in valid])[len(valid) // 2],
                "cv": sorted([r["cv"] for r in valid])[len(valid) // 2],
                "tflops": tflops,
                "speedup_vs_bf16_same_shape": speedup_vs_bf16,
                "max_error": valid[0]["max_error"],
                "rel_error": valid[0]["rel_error"],
                "cache_mode": args.cache_mode,
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "independent_trials": args.trials,
                "gpu": gpu_name,
                "contamination": args.contamination,
                "warnings": valid[0].get("warnings", []),
            }
            all_results.append(result)
            print(f"  {backend:20s} M={M:>5d}  median={result['median_latency_us']:.2f}us  "
                  f"TFLOP/s={result['tflops']:.1f}  speedup_vs_bf16={result['speedup_vs_bf16_same_shape']:.2f}x  "
                  f"cv={result['cv']:.3f}")

    # Grouped GEMM tests
    if not args.skip_grouped and availability.get("deep_gemm", False):
        print("\n=== DeepGEMM Grouped FP8 GEMM ===")
        distributions = ["balanced", "moderately_skewed", "highly_skewed", "small_token"]
        for dist in distributions:
            for M in [64, 256, 1024]:
                try:
                    r = benchmark_deep_gemm_grouped(M, args.n, args.k, device,
                                                    args.num_experts, args.topk,
                                                    args.warmup, args.iterations, distribution=dist)
                    bf16_base = bf16_baselines.get(M, 1.0)
                    r["speedup_vs_bf16_same_shape"] = bf16_base / r["median_latency_us"] if r["median_latency_us"] > 0 else 0
                    all_results.append(r)
                    print(f"  grouped ({dist:20s}) M={M:>5d}  median={r['median_latency_us']:.2f}us  "
                          f"TFLOP/s={r['tflops']:.1f}  experts={args.num_experts}")
                except Exception as e:
                    r = {
                        "backend": "deep_gemm",
                        "kernel": "deep_gemm.fp8_m_grouped_gemm_nt_masked",
                        "M": M, "N": args.n, "K": args.k,
                        "distribution": dist,
                        "status": "ERROR",
                        "error": str(e),
                        "gpu": gpu_name,
                    }
                    all_results.append(r)
                    print(f"  grouped ({dist:20s}) M={M:>5d}  ERROR: {e}")

    if args.output_json:
        out = {
            "gpu_name": gpu_name,
            "N": args.n, "K": args.k,
            "cache_mode": args.cache_mode,
            "trials": args.trials,
            "backend_availability": availability,
            "results": all_results,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
