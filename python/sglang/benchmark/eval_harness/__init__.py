"""Bridge: run lm-evaluation-harness tasks through sglang.bench_serving."""

from __future__ import annotations


def _not_implemented(*_args, **_kwargs):
    raise NotImplementedError("eval_harness helper not yet implemented")


from sglang.benchmark.eval_harness.bench_serving_lm import BenchServingLM

merge_report = _not_implemented
write_report = _not_implemented

__all__ = ["BenchServingLM", "merge_report", "write_report"]
