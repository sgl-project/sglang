"""Bridge: run lm-evaluation-harness tasks through sglang.bench_serving."""

from __future__ import annotations

from sglang.benchmark.eval_harness.bench_serving_lm import BenchServingLM
from sglang.benchmark.eval_harness.report import merge_report, write_report

__all__ = ["BenchServingLM", "merge_report", "write_report"]
