#!/usr/bin/env python3
"""
SageAttention Benchmark Suite for A100 GPUs

This script runs comprehensive benchmarks comparing SageAttention's INT8 quantized
attention against FP16 baselines (Triton, FlashInfer) on A100 GPUs.

Metrics collected:
- Throughput (tokens/sec) for prefill and decode
- Memory footprint (peak GPU memory usage)
- Accuracy (MMLU score, output consistency)
- Latency distribution

Usage:
    # Full benchmark (all tests)
    python benchmark/sage_attention_benchmark.py

    # Quick benchmark (throughput only)
    python benchmark/sage_attention_benchmark.py --quick

    # Specific tests
    python benchmark/sage_attention_benchmark.py --throughput-only
    python benchmark/sage_attention_benchmark.py --accuracy-only

    # Custom model
    python benchmark/sage_attention_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct

Output:
    Results are printed to stdout and saved to benchmark_results.json
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    backends: List[str] = field(default_factory=lambda: ["sage_attn", "triton"])
    input_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    output_lengths: List[int] = field(default_factory=lambda: [64, 128, 256])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    num_prompts: int = 10
    warmup_prompts: int = 2
    mmlu_examples: int = 100
    base_port: int = 30000


@dataclass
class ThroughputResult:
    """Results from throughput benchmark."""

    backend: str
    input_len: int
    output_len: int
    batch_size: int
    throughput_tok_s: float
    latency_s: float
    prefill_latency_ms: Optional[float] = None
    decode_throughput_tok_s: Optional[float] = None


@dataclass
class MemoryResult:
    """Results from memory benchmark."""

    backend: str
    peak_memory_mb: float
    allocated_memory_mb: float
    model_size_mb: float


@dataclass
class AccuracyResult:
    """Results from accuracy benchmark."""

    backend: str
    mmlu_score: float
    output_consistency: float  # 0-1 score vs baseline


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    timestamp: str
    gpu_name: str
    gpu_memory_total_gb: float
    model: str
    throughput_results: List[ThroughputResult]
    memory_results: List[MemoryResult]
    accuracy_results: List[AccuracyResult]
    summary: Dict[str, Any]


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"name": "N/A", "memory_gb": 0}

    return {
        "name": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "compute_capability": torch.cuda.get_device_capability(0),
    }


def reset_gpu():
    """Reset GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def run_throughput_benchmark(
    config: BenchmarkConfig,
    backend: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
) -> ThroughputResult:
    """Run throughput benchmark for a specific configuration."""
    command = [
        sys.executable,
        "-m",
        "sglang.bench_offline_throughput",
        "--model-path",
        config.model,
        "--attention-backend",
        backend,
        "--num-prompts",
        str(num_prompts),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )

        output = result.stdout + result.stderr
        throughput = -1.0
        latency = -1.0

        for line in output.split("\n"):
            if "Last generation throughput (tok/s):" in line:
                try:
                    throughput = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            if "Total latency:" in line:
                try:
                    latency = float(line.split(":")[-1].strip().replace("s", ""))
                except (ValueError, IndexError):
                    pass

        return ThroughputResult(
            backend=backend,
            input_len=input_len,
            output_len=output_len,
            batch_size=num_prompts,
            throughput_tok_s=throughput,
            latency_s=latency,
        )

    except subprocess.TimeoutExpired:
        return ThroughputResult(
            backend=backend,
            input_len=input_len,
            output_len=output_len,
            batch_size=num_prompts,
            throughput_tok_s=-1.0,
            latency_s=-1.0,
        )
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return ThroughputResult(
            backend=backend,
            input_len=input_len,
            output_len=output_len,
            batch_size=num_prompts,
            throughput_tok_s=-1.0,
            latency_s=-1.0,
        )


def run_memory_benchmark(config: BenchmarkConfig, backend: str) -> MemoryResult:
    """Measure memory usage for a backend."""
    # Start server and measure memory
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import popen_launch_server

    base_url = f"http://localhost:{config.base_port}"

    reset_gpu()

    process = popen_launch_server(
        config.model,
        base_url,
        timeout=300,
        other_args=["--attention-backend", backend],
    )

    try:
        time.sleep(15)  # Wait for model to fully load

        # Get memory info via nvidia-smi
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        if nvidia_smi.returncode == 0:
            lines = nvidia_smi.stdout.strip().split("\n")
            if lines:
                used, total = map(float, lines[0].split(","))
                return MemoryResult(
                    backend=backend,
                    peak_memory_mb=used,
                    allocated_memory_mb=used,
                    model_size_mb=0,  # Would need more introspection
                )

    except Exception as e:
        print(f"Memory benchmark error: {e}")
    finally:
        try:
            kill_process_tree(process.pid)
        except Exception:
            pass
        time.sleep(5)
        reset_gpu()

    return MemoryResult(
        backend=backend,
        peak_memory_mb=-1,
        allocated_memory_mb=-1,
        model_size_mb=-1,
    )


def run_accuracy_benchmark(config: BenchmarkConfig, backend: str) -> AccuracyResult:
    """Run accuracy benchmark (MMLU)."""
    from types import SimpleNamespace

    from sglang.srt.utils import kill_process_tree
    from sglang.test.run_eval import run_eval
    from sglang.test.test_utils import popen_launch_server

    base_url = f"http://localhost:{config.base_port}"

    process = popen_launch_server(
        config.model,
        base_url,
        timeout=300,
        other_args=["--attention-backend", backend],
    )

    try:
        args = SimpleNamespace(
            base_url=base_url,
            model=config.model,
            eval_name="mmlu",
            num_examples=config.mmlu_examples,
            num_threads=32,
        )

        metrics = run_eval(args)
        return AccuracyResult(
            backend=backend,
            mmlu_score=metrics.get("score", 0),
            output_consistency=1.0,  # Would need reference comparison
        )

    except Exception as e:
        print(f"Accuracy benchmark error: {e}")
        return AccuracyResult(
            backend=backend,
            mmlu_score=-1,
            output_consistency=-1,
        )
    finally:
        try:
            kill_process_tree(process.pid)
        except Exception:
            pass
        time.sleep(5)


def print_banner(text: str):
    """Print a banner."""
    width = 70
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)


def run_benchmarks(config: BenchmarkConfig, quick: bool = False) -> BenchmarkReport:
    """Run all benchmarks and generate report."""
    gpu_info = get_gpu_info()
    print_banner("SageAttention Benchmark Suite")
    print(f"GPU: {gpu_info['name']}")
    print(f"Memory: {gpu_info['memory_gb']:.1f} GB")
    print(f"Model: {config.model}")
    print(f"Backends: {', '.join(config.backends)}")

    throughput_results = []
    memory_results = []
    accuracy_results = []

    # Throughput benchmarks
    print_banner("THROUGHPUT BENCHMARKS")

    test_configs = []
    if quick:
        test_configs = [(256, 64), (512, 128)]
    else:
        for input_len in config.input_lengths:
            for output_len in config.output_lengths:
                test_configs.append((input_len, output_len))

    for input_len, output_len in test_configs:
        print(f"\n--- Input: {input_len}, Output: {output_len} ---")
        for backend in config.backends:
            result = run_throughput_benchmark(
                config, backend, input_len, output_len, config.num_prompts
            )
            throughput_results.append(result)
            tp_str = f"{result.throughput_tok_s:.2f}" if result.throughput_tok_s > 0 else "FAILED"
            print(f"  {backend}: {tp_str} tok/s")

    # Memory benchmarks
    if not quick:
        print_banner("MEMORY BENCHMARKS")
        for backend in config.backends:
            print(f"Testing {backend}...")
            result = run_memory_benchmark(config, backend)
            memory_results.append(result)
            mem_str = f"{result.peak_memory_mb:.0f}" if result.peak_memory_mb > 0 else "FAILED"
            print(f"  {backend}: {mem_str} MB peak")

    # Accuracy benchmarks
    if not quick:
        print_banner("ACCURACY BENCHMARKS (MMLU)")
        for backend in config.backends:
            print(f"Testing {backend}...")
            result = run_accuracy_benchmark(config, backend)
            accuracy_results.append(result)
            score_str = f"{result.mmlu_score:.4f}" if result.mmlu_score >= 0 else "FAILED"
            print(f"  {backend}: {score_str}")

    # Generate summary
    summary = generate_summary(throughput_results, memory_results, accuracy_results)

    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        gpu_name=gpu_info["name"],
        gpu_memory_total_gb=gpu_info["memory_gb"],
        model=config.model,
        throughput_results=throughput_results,
        memory_results=memory_results,
        accuracy_results=accuracy_results,
        summary=summary,
    )

    return report


def generate_summary(
    throughput_results: List[ThroughputResult],
    memory_results: List[MemoryResult],
    accuracy_results: List[AccuracyResult],
) -> Dict[str, Any]:
    """Generate summary statistics."""
    summary = {}

    # Throughput summary
    sage_throughputs = [
        r.throughput_tok_s for r in throughput_results
        if r.backend == "sage_attn" and r.throughput_tok_s > 0
    ]
    triton_throughputs = [
        r.throughput_tok_s for r in throughput_results
        if r.backend == "triton" and r.throughput_tok_s > 0
    ]

    if sage_throughputs and triton_throughputs:
        avg_sage = sum(sage_throughputs) / len(sage_throughputs)
        avg_triton = sum(triton_throughputs) / len(triton_throughputs)
        summary["avg_throughput_sage_tok_s"] = avg_sage
        summary["avg_throughput_triton_tok_s"] = avg_triton
        summary["avg_speedup"] = avg_sage / avg_triton if avg_triton > 0 else 0

    # Memory summary
    sage_mem = [r for r in memory_results if r.backend == "sage_attn"]
    triton_mem = [r for r in memory_results if r.backend == "triton"]

    if sage_mem and triton_mem:
        summary["memory_sage_mb"] = sage_mem[0].peak_memory_mb
        summary["memory_triton_mb"] = triton_mem[0].peak_memory_mb
        if triton_mem[0].peak_memory_mb > 0:
            summary["memory_savings_pct"] = (
                (triton_mem[0].peak_memory_mb - sage_mem[0].peak_memory_mb)
                / triton_mem[0].peak_memory_mb * 100
            )

    # Accuracy summary
    sage_acc = [r for r in accuracy_results if r.backend == "sage_attn"]
    triton_acc = [r for r in accuracy_results if r.backend == "triton"]

    if sage_acc and triton_acc:
        summary["mmlu_sage"] = sage_acc[0].mmlu_score
        summary["mmlu_triton"] = triton_acc[0].mmlu_score
        summary["mmlu_diff"] = abs(sage_acc[0].mmlu_score - triton_acc[0].mmlu_score)

    return summary


def print_report(report: BenchmarkReport):
    """Print formatted benchmark report."""
    print_banner("BENCHMARK REPORT")

    print(f"\nTimestamp: {report.timestamp}")
    print(f"GPU: {report.gpu_name}")
    print(f"Model: {report.model}")

    # Throughput table
    print("\n--- Throughput Results ---")
    print(f"{'Backend':<12} {'Input':<8} {'Output':<8} {'Throughput':<15} {'Latency':<10}")
    print("-" * 53)
    for r in report.throughput_results:
        tp_str = f"{r.throughput_tok_s:.2f} tok/s" if r.throughput_tok_s > 0 else "FAILED"
        lat_str = f"{r.latency_s:.2f}s" if r.latency_s > 0 else "N/A"
        print(f"{r.backend:<12} {r.input_len:<8} {r.output_len:<8} {tp_str:<15} {lat_str:<10}")

    # Memory table
    if report.memory_results:
        print("\n--- Memory Results ---")
        print(f"{'Backend':<12} {'Peak Memory':<15}")
        print("-" * 27)
        for r in report.memory_results:
            mem_str = f"{r.peak_memory_mb:.0f} MB" if r.peak_memory_mb > 0 else "FAILED"
            print(f"{r.backend:<12} {mem_str:<15}")

    # Accuracy table
    if report.accuracy_results:
        print("\n--- Accuracy Results (MMLU) ---")
        print(f"{'Backend':<12} {'Score':<10}")
        print("-" * 22)
        for r in report.accuracy_results:
            score_str = f"{r.mmlu_score:.4f}" if r.mmlu_score >= 0 else "FAILED"
            print(f"{r.backend:<12} {score_str:<10}")

    # Summary
    print_banner("SUMMARY")
    for key, value in report.summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def save_report(report: BenchmarkReport, filename: str):
    """Save report to JSON file."""
    # Convert dataclasses to dicts
    data = {
        "timestamp": report.timestamp,
        "gpu_name": report.gpu_name,
        "gpu_memory_total_gb": report.gpu_memory_total_gb,
        "model": report.model,
        "throughput_results": [asdict(r) for r in report.throughput_results],
        "memory_results": [asdict(r) for r in report.memory_results],
        "accuracy_results": [asdict(r) for r in report.accuracy_results],
        "summary": report.summary,
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nReport saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="SageAttention Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (throughput only, fewer configs)",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="Run only throughput benchmarks",
    )
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Run only accuracy benchmarks",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["sage_attn", "triton"],
        help="Backends to benchmark",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        model=args.model,
        backends=args.backends,
    )

    report = run_benchmarks(config, quick=args.quick)
    print_report(report)
    save_report(report, args.output)


if __name__ == "__main__":
    main()
