#!/usr/bin/env python3
"""
Benchmark: Attention Capture Overhead Measurement

This script measures the performance impact of attention token capture
across different modes to prove:
1. Zero regression when capture is disabled
2. Overhead breakdown per mode (fingerprint, sketch, raw)

Usage:
    # Start server WITHOUT attention capture (baseline):
    python -m sglang.launch_server --model Qwen/Qwen2.5-1.5B-Instruct --port 30000

    # Run baseline benchmark:
    python scripts/benchmark_attention_capture.py --base-url http://localhost:30000 --mode baseline

    # Start server WITH attention capture enabled:
    python -m sglang.launch_server --model Qwen/Qwen2.5-1.5B-Instruct --port 30001 \
        --attention-fingerprint-mode

    # Run all modes:
    python scripts/benchmark_attention_capture.py --base-url http://localhost:30001 --mode all

    # Compare two servers:
    python scripts/benchmark_attention_capture.py \
        --baseline-url http://localhost:30000 \
        --capture-url http://localhost:30001 \
        --mode compare

Output: tokens/sec, p50/p99 latency, memory delta
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    mode: str
    requests_completed: int
    total_tokens: int
    total_time: float
    latencies: List[float]
    tokens_per_sec: float
    p50_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    errors: int


def get_server_info(base_url: str) -> Optional[Dict]:
    """Get server configuration info."""
    try:
        response = requests.get(f"{base_url}/get_server_info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def run_single_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    return_attention_tokens: bool = False,
    attention_mode: Optional[str] = None,
) -> Tuple[float, int, bool]:
    """
    Run a single request and return (latency_seconds, tokens_generated, success).
    """
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    if return_attention_tokens:
        body["return_attention_tokens"] = True
        if attention_mode:
            body["attention_mode"] = attention_mode

    try:
        start = time.perf_counter()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            timeout=120,
        )
        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            return elapsed, 0, False

        data = response.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return elapsed, tokens, True

    except Exception as e:
        return 0.0, 0, False


def run_benchmark(
    base_url: str,
    mode: str,
    num_requests: int = 50,
    max_tokens: int = 100,
    warmup: int = 5,
) -> BenchmarkResult:
    """
    Run benchmark for a specific mode.

    Modes:
    - baseline: No attention capture requested
    - disabled: Server has capture, but request doesn't ask for it
    - fingerprint: Request fingerprint mode
    - sketch: Request sketch mode
    - raw: Request raw mode
    """
    # Determine request parameters
    return_attention = mode not in ("baseline", "disabled")
    attention_mode = mode if mode in ("fingerprint", "sketch", "raw") else None

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the main differences between Python and JavaScript?",
        "Describe the water cycle step by step.",
        "How do neural networks learn from data?",
    ]

    # Warmup
    print(f"  Warming up ({warmup} requests)...", end="", flush=True)
    for i in range(warmup):
        run_single_request(
            base_url,
            prompts[i % len(prompts)],
            max_tokens,
            return_attention,
            attention_mode,
        )
    print(" done")

    # Benchmark
    print(f"  Running {num_requests} requests...", end="", flush=True)
    latencies = []
    total_tokens = 0
    errors = 0

    start_total = time.perf_counter()
    for i in range(num_requests):
        elapsed, tokens, success = run_single_request(
            base_url,
            prompts[i % len(prompts)],
            max_tokens,
            return_attention,
            attention_mode,
        )
        if success:
            latencies.append(elapsed)
            total_tokens += tokens
        else:
            errors += 1

        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)

    total_time = time.perf_counter() - start_total
    print(" done")

    if not latencies:
        return BenchmarkResult(
            mode=mode,
            requests_completed=0,
            total_tokens=0,
            total_time=total_time,
            latencies=[],
            tokens_per_sec=0,
            p50_latency_ms=0,
            p99_latency_ms=0,
            avg_latency_ms=0,
            errors=errors,
        )

    sorted_latencies = sorted(latencies)
    p50_idx = int(len(sorted_latencies) * 0.50)
    p99_idx = int(len(sorted_latencies) * 0.99)

    return BenchmarkResult(
        mode=mode,
        requests_completed=len(latencies),
        total_tokens=total_tokens,
        total_time=total_time,
        latencies=latencies,
        tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        p50_latency_ms=sorted_latencies[p50_idx] * 1000,
        p99_latency_ms=sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)] * 1000,
        avg_latency_ms=statistics.mean(latencies) * 1000,
        errors=errors,
    )


def print_result(result: BenchmarkResult, baseline: Optional[BenchmarkResult] = None):
    """Print benchmark result with optional comparison to baseline."""
    print(f"\n  Mode: {result.mode}")
    print(f"  Requests: {result.requests_completed} completed, {result.errors} errors")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Throughput: {result.tokens_per_sec:.1f} tok/s", end="")

    if baseline and baseline.tokens_per_sec > 0:
        delta = (result.tokens_per_sec - baseline.tokens_per_sec) / baseline.tokens_per_sec * 100
        print(f" ({delta:+.1f}% vs baseline)")
    else:
        print()

    print(f"  Latency p50: {result.p50_latency_ms:.1f} ms", end="")
    if baseline and baseline.p50_latency_ms > 0:
        delta = (result.p50_latency_ms - baseline.p50_latency_ms) / baseline.p50_latency_ms * 100
        print(f" ({delta:+.1f}%)")
    else:
        print()

    print(f"  Latency p99: {result.p99_latency_ms:.1f} ms", end="")
    if baseline and baseline.p99_latency_ms > 0:
        delta = (result.p99_latency_ms - baseline.p99_latency_ms) / baseline.p99_latency_ms * 100
        print(f" ({delta:+.1f}%)")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention capture overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="Server URL to benchmark",
    )
    parser.add_argument(
        "--baseline-url",
        type=str,
        help="Baseline server URL (for compare mode)",
    )
    parser.add_argument(
        "--capture-url",
        type=str,
        help="Capture-enabled server URL (for compare mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "disabled", "fingerprint", "sketch", "raw", "all", "compare"],
        default="all",
        help="Benchmark mode(s) to run",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Number of requests per mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per request",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup requests before measurement",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Attention Capture Overhead Benchmark")
    print("=" * 60)

    results = {}

    if args.mode == "compare":
        if not args.baseline_url or not args.capture_url:
            print("Error: --baseline-url and --capture-url required for compare mode")
            sys.exit(1)

        # Baseline server (no capture)
        print(f"\n[1/4] Baseline server: {args.baseline_url}")
        info = get_server_info(args.baseline_url)
        if info:
            print(f"  Model: {info.get('model_path', 'unknown')}")

        baseline = run_benchmark(
            args.baseline_url, "baseline", args.requests, args.max_tokens, args.warmup
        )
        results["baseline"] = baseline
        print_result(baseline)

        # Capture server - disabled mode
        print(f"\n[2/4] Capture server (disabled): {args.capture_url}")
        disabled = run_benchmark(
            args.capture_url, "disabled", args.requests, args.max_tokens, args.warmup
        )
        results["disabled"] = disabled
        print_result(disabled, baseline)

        # Capture server - fingerprint mode
        print(f"\n[3/4] Capture server (fingerprint): {args.capture_url}")
        fingerprint = run_benchmark(
            args.capture_url, "fingerprint", args.requests, args.max_tokens, args.warmup
        )
        results["fingerprint"] = fingerprint
        print_result(fingerprint, baseline)

        # Capture server - raw mode (if supported)
        print(f"\n[4/4] Capture server (raw): {args.capture_url}")
        raw = run_benchmark(
            args.capture_url, "raw", args.requests, args.max_tokens, args.warmup
        )
        results["raw"] = raw
        print_result(raw, baseline)

    elif args.mode == "all":
        print(f"\nServer: {args.base_url}")
        info = get_server_info(args.base_url)
        if info:
            print(f"Model: {info.get('model_path', 'unknown')}")

        modes = ["disabled", "fingerprint", "raw"]
        baseline = None

        for i, mode in enumerate(modes, 1):
            print(f"\n[{i}/{len(modes)}] Mode: {mode}")
            result = run_benchmark(
                args.base_url, mode, args.requests, args.max_tokens, args.warmup
            )
            results[mode] = result

            if mode == "disabled":
                baseline = result

            print_result(result, baseline)

    else:
        print(f"\nServer: {args.base_url}")
        result = run_benchmark(
            args.base_url, args.mode, args.requests, args.max_tokens, args.warmup
        )
        results[args.mode] = result
        print_result(result)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if "baseline" in results or "disabled" in results:
        baseline = results.get("baseline") or results.get("disabled")
        print(f"\nBaseline throughput: {baseline.tokens_per_sec:.1f} tok/s")

        for mode, result in results.items():
            if mode in ("baseline", "disabled"):
                continue
            if baseline.tokens_per_sec > 0:
                overhead = (baseline.tokens_per_sec - result.tokens_per_sec) / baseline.tokens_per_sec * 100
                print(f"{mode}: {result.tokens_per_sec:.1f} tok/s ({overhead:+.1f}% overhead)")

    # Regression check
    if "baseline" in results and "disabled" in results:
        baseline = results["baseline"]
        disabled = results["disabled"]

        if baseline.tokens_per_sec > 0:
            regression = (baseline.tokens_per_sec - disabled.tokens_per_sec) / baseline.tokens_per_sec * 100

            print(f"\n*** REGRESSION CHECK ***")
            if abs(regression) < 2.0:
                print(f"PASS: Disabled mode has {regression:+.1f}% throughput delta (< 2% threshold)")
            else:
                print(f"WARN: Disabled mode has {regression:+.1f}% throughput delta (>= 2% threshold)")

    # Save results
    if args.output:
        output_data = {
            mode: {
                "mode": r.mode,
                "requests_completed": r.requests_completed,
                "total_tokens": r.total_tokens,
                "tokens_per_sec": r.tokens_per_sec,
                "p50_latency_ms": r.p50_latency_ms,
                "p99_latency_ms": r.p99_latency_ms,
                "avg_latency_ms": r.avg_latency_ms,
                "errors": r.errors,
            }
            for mode, r in results.items()
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
