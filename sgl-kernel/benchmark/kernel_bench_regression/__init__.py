"""SGLang kernel benchmark regression harness.

This package wraps the existing ``sgl-kernel/benchmark/bench_*.py`` benchmarks,
captures their throughput / latency into a structured JSON, and compares a fresh
run against a nightly-generated ground truth with a relative tolerance.

The suite is model-agnostic -- it tracks whatever kernels are registered in
``registry.py`` and is meant to grow across model families. See ``README.md`` for
the full ground-truth flow.
"""

from .registry import KERNEL_BENCH_CASES, BenchCase

__all__ = ["KERNEL_BENCH_CASES", "BenchCase"]
