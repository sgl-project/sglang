"""Combine simple_evaluate's result dict with bench_serving perf metrics.

The output is one JSON record per run, appended to a JSONL file so multiple
runs (e.g. a sweep across request_rate) accumulate naturally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


_PERF_FIELDS = (
    "duration", "completed",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "input_throughput", "output_throughput", "total_throughput",
    "mean_e2e_latency_ms", "median_e2e_latency_ms", "p99_e2e_latency_ms",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_itl_ms", "median_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "concurrency", "max_output_tokens_per_s",
)


def merge_report(
    *,
    task_name: str,
    lm_eval_results: Dict[str, Any],
    perf: Dict[str, Any],
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_results = dict(lm_eval_results["results"].get(task_name, {}))
    # `alias` is lm-eval bookkeeping, not a metric.
    task_results.pop("alias", None)

    n_samples = lm_eval_results.get("n-samples", {}).get(task_name, {})

    return {
        "task": task_name,
        "accuracy": task_results,
        "n_samples": n_samples,
        "lm_eval_config": lm_eval_results.get("config", {}),
        "perf": {k: perf[k] for k in _PERF_FIELDS if k in perf},
        "run": dict(run_config or {}),
    }


def write_report(path: str, report: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
