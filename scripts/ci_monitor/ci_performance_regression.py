#!/usr/bin/env python3
"""
SGLang CI Performance Regression Tracker

Analyzes performance metrics from CI runs and detects regressions.
Designed to be called from ci_failures_analysis.py.

Usage from ci_failures_analysis.py:
    from ci_performance_regression import analyze_performance_regressions

    perf_summary = analyze_performance_regressions(
        scheduled_runs,
        analyzer.get_jobs_for_run,
        analyzer.get_job_logs
    )
"""

import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def extract_performance_metrics(logs: str) -> Dict[str, float]:
    """
    Extract performance metrics from job logs.

    Returns:
        Dict mapping metric names to values
    """
    logs = strip_ansi_codes(logs)

    # Performance metric patterns
    patterns = {
        # Throughput metrics (higher is better)
        "output_throughput_token_s": r"Output token throughput \(tok/s\):\s*([\d.]+)",
        "input_throughput_token_s": r"Input token throughput \(tok/s\):\s*([\d.]+)",
        # Latency metrics (lower is better)
        "median_e2e_latency_ms": r"Median E2E Latency \(ms\):\s*([\d.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s*([\d.]+)",
        # Diffusion-specific metrics (lower is better)
        "denoise_time_per_step": r"\[DenoisingStage\] average time per step:\s*([\d.]+)\s*seconds",
        "denoise_total_time": r"\[DenoisingStage\] finished in\s*([\d.]+)\s*seconds",
        # Other metrics
        "accept_length": r"Accept length:\s*([\d.]+)",
    }

    metrics = {}
    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, logs, re.IGNORECASE)
        if matches:
            try:
                # Use the last match (usually the final summary)
                metrics[metric_name] = float(matches[-1])
            except (ValueError, IndexError):
                continue

    return metrics


def analyze_performance_regressions(
    scheduled_runs: List[Dict],
    get_jobs_func: Callable[[int], List[Dict]],
    get_logs_func: Callable[[int], str],
    baseline_window: int = 10,
    regression_threshold: float = 0.10,
) -> Dict:
    """
    Analyze performance metrics from scheduled runs and detect regressions.

    Args:
        scheduled_runs: List of workflow runs (most recent first)
        get_jobs_func: Function to get jobs for a run (pass analyzer.get_jobs_for_run)
        get_logs_func: Function to get logs for a job (pass analyzer.get_job_logs)
        baseline_window: Number of older runs to use for baseline (default: 10)
        regression_threshold: Percentage threshold for regression (default: 0.10 = 10%)

    Returns:
        Dict with performance summary:
        {
            "total_tests_with_metrics": int,
            "tests_with_regressions": int,
            "regressions": {
                "job_name": {
                    "job_name": str,
                    "run_number": int,
                    "head_sha": str,
                    "metrics": [
                        {
                            "metric": str,
                            "baseline_avg": float,
                            "recent_value": float,
                            "change_pct": float,
                            "is_regression": bool
                        },
                        ...
                    ]
                }
            }
        }
    """
    print(f"\n{'='*60}")
    print("Analyzing performance metrics from scheduled runs...")
    print(f"Baseline window: {baseline_window} runs")
    print(f"Regression threshold: {regression_threshold * 100}%")
    print(f"{'='*60}\n")

    # Metrics where LOWER is BETTER (latency, time)
    lower_is_better = {
        "median_e2e_latency_ms",
        "median_ttft_ms",
        "denoise_time_per_step",
        "denoise_total_time",
    }

    # Collect performance data per job over time
    # Structure: {job_name: [{run_number, head_sha, metrics}, ...]}
    job_history = defaultdict(list)

    # Process runs (already sorted by GitHub API, newest first)
    for run in scheduled_runs[
        : baseline_window + 1
    ]:  # Only need baseline + 1 most recent
        run_id = run.get("id")
        run_number = run.get("run_number")
        head_sha = (run.get("head_sha") or "")[:8]

        jobs = get_jobs_func(run_id)

        for job in jobs:
            job_name = job.get("name", "unknown")
            job_id = job.get("id")
            conclusion = job.get("conclusion")

            # Skip failed jobs (no valid perf data)
            if conclusion not in ["success", "neutral"]:
                continue

            # Get job logs
            logs = get_logs_func(job_id)
            if not logs:
                continue

            # Extract performance metrics
            metrics = extract_performance_metrics(logs)

            if metrics:
                job_history[job_name].append(
                    {"run_number": run_number, "head_sha": head_sha, "metrics": metrics}
                )

    # Detect regressions
    regressions = {}

    for job_name, history in job_history.items():
        # Need at least baseline_window + 1 runs
        if len(history) < baseline_window + 1:
            continue

        # Most recent run is first (newest)
        recent_run = history[0]

        # Baseline: average of older runs (excluding the most recent)
        baseline_runs = history[1 : baseline_window + 1]

        # Calculate baseline averages for each metric
        baseline_metrics = defaultdict(list)
        for run in baseline_runs:
            for metric_name, value in run["metrics"].items():
                baseline_metrics[metric_name].append(value)

        baseline_avgs = {
            metric: sum(values) / len(values)
            for metric, values in baseline_metrics.items()
        }

        # Check for regressions
        regression_metrics = []
        for metric_name, recent_value in recent_run["metrics"].items():
            if metric_name not in baseline_avgs:
                continue

            baseline_avg = baseline_avgs[metric_name]
            if baseline_avg == 0:
                continue

            # Calculate percentage change
            change_pct = (recent_value - baseline_avg) / baseline_avg

            # Determine if this is a regression based on metric type
            is_regression = False
            if metric_name in lower_is_better:
                # For latency/time metrics, increase is bad
                is_regression = change_pct > regression_threshold
            else:
                # For throughput metrics, decrease is bad
                is_regression = change_pct < -regression_threshold

            regression_metrics.append(
                {
                    "metric": metric_name,
                    "baseline_avg": baseline_avg,
                    "recent_value": recent_value,
                    "change_pct": change_pct * 100,  # Convert to percentage
                    "is_regression": is_regression,
                }
            )

        # Only include if there are actual regressions
        if any(m["is_regression"] for m in regression_metrics):
            regressions[job_name] = {
                "job_name": job_name,
                "run_number": recent_run["run_number"],
                "head_sha": recent_run["head_sha"],
                "metrics": regression_metrics,
            }

    summary = {
        "total_tests_with_metrics": len(job_history),
        "tests_with_regressions": len(regressions),
        "regressions": regressions,
    }

    print(f"✅ Found {len(job_history)} jobs with performance metrics")
    if regressions:
        print(f"⚠️  Detected regressions in {len(regressions)} jobs")
    else:
        print("✅ No regressions detected")

    return summary


def format_performance_summary_for_github(perf_summary: Dict) -> List[str]:
    """
    Format performance summary for GitHub Actions summary.

    Args:
        perf_summary: Output from analyze_performance_regressions()

    Returns:
        List of markdown lines
    """
    lines = []

    lines.append("\n## 📊 Performance Regression Analysis\n")

    if not perf_summary.get("regressions"):
        lines.append("✅ **No performance regressions detected**\n")
        return lines

    regressions = perf_summary["regressions"]
    lines.append(f"⚠️ **Found regressions in {len(regressions)} jobs**\n")

    for job_name, data in sorted(regressions.items()):
        lines.append(f"\n### 📉 {job_name}\n")
        lines.append(f"Run: #{data['run_number']} ({data['head_sha']})\n")
        lines.append("\n| Metric | Baseline | Recent | Change |\n")
        lines.append("|--------|----------|--------|--------|\n")

        for metric in data["metrics"]:
            if not metric["is_regression"]:
                continue

            metric_name = metric["metric"]
            baseline = metric["baseline_avg"]
            recent = metric["recent_value"]
            change = metric["change_pct"]

            emoji = "⬆️" if change > 0 else "⬇️"
            lines.append(
                f"| {metric_name} | {baseline:.2f} | {recent:.2f} | {emoji} {change:+.1f}% |\n"
            )

        lines.append("\n")

    return lines
