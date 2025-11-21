import argparse
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple


def calculate_diff(base: float, new: float) -> Tuple[float, float]:
    """Returns (diff, diff_percent)."""
    diff = new - base
    if base == 0:
        percent = 0.0
    else:
        percent = (diff / base) * 100
    return diff, percent


def calculate_upper_bound(baseline: float, rel_tol: float, min_abs_tol: float) -> float:
    """Calculates the upper bound for performance regression check."""
    rel_limit = baseline * (1 + rel_tol)
    abs_limit = baseline + min_abs_tol
    return max(rel_limit, abs_limit)


def calculate_lower_bound(baseline: float, rel_tol: float, min_abs_tol: float) -> float:
    """Calculates the lower bound for performance improvement check."""
    rel_lower = baseline * (1 - rel_tol)
    abs_lower = baseline - min_abs_tol
    return min(rel_lower, abs_lower)


def get_perf_status_emoji(
    baseline: float,
    new: float,
    rel_tol: float = 0.1,
    min_abs_tol: float = 120.0,
) -> str:
    """
    Determines the status emoji based on performance difference.

    Logic:
      Upper bound (Slower): max(baseline * (1 + rel_tol), baseline + min_abs_tol)
      Lower bound (Faster): min(baseline * (1 - rel_tol), baseline - min_abs_tol)
    """
    upper_bound = calculate_upper_bound(baseline, rel_tol, min_abs_tol)
    lower_bound = calculate_lower_bound(baseline, rel_tol, min_abs_tol)

    if new > upper_bound:
        return "🔴"
    elif new < lower_bound:
        return "🟢"
    else:
        return "⚪️"


def consolidate_steps(
    steps_list: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], List[str], Dict[str, int]]:
    """
    Aggregates specific repeating steps (like denoising_step_*) into groups.
    Returns:
        - aggregated_durations: {name: duration_ms}
        - ordered_names: list of names in execution order
        - counts: {name: count_of_steps_aggregated}
    """
    durations = {}
    counts = {}
    ordered_names = []
    seen_names = set()

    # Regex for steps to group
    # Group "denoising_step_0", "denoising_step_1" -> "Denoising Loop"
    denoise_pattern = re.compile(r"^denoising_step_(\d+)$")
    denoising_group_name = "Denoising Loop"

    for step in steps_list:
        name = step.get("name", "unknown")
        dur = step.get("duration_ms", 0.0)

        match = denoise_pattern.match(name)
        if match:
            key = denoising_group_name
            if key not in durations:
                durations[key] = 0.0
                counts[key] = 0
                if key not in seen_names:
                    ordered_names.append(key)
                    seen_names.add(key)
            durations[key] += dur
            counts[key] += 1
        else:
            # Standard stage (preserve order)
            if name not in durations:
                durations[name] = 0.0
                counts[name] = 0
                if name not in seen_names:
                    ordered_names.append(name)
                    seen_names.add(name)
            durations[name] += dur
            counts[name] += 1

    return durations, ordered_names, counts


def _load_benchmark_file(file_path: str) -> Dict[str, Any]:
    """Loads a benchmark JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_benchmarks(
    baseline_path: str, new_path: str, output_format: str = "markdown"
):
    """
    Compares two benchmark JSON files and prints a report.
    """
    try:
        base_data = _load_benchmark_file(baseline_path)
        new_data = _load_benchmark_file(new_path)
    except Exception as e:
        print(f"Error loading benchmark files: {e}")
        return

    base_e2e = base_data.get("total_duration_ms", 0)
    new_e2e = new_data.get("total_duration_ms", 0)

    diff_ms, diff_pct = calculate_diff(base_e2e, new_e2e)

    if diff_pct < -2.0:
        status = "✅"
    elif diff_pct > 2.0:
        status = "❌"
    else:
        status = ""

    # --- Stage Breakdown ---
    base_durations, base_order, base_counts = consolidate_steps(
        base_data.get("steps", [])
    )
    new_durations, new_order, new_counts = consolidate_steps(new_data.get("steps", []))

    # Merge orders: Start with New order (execution order), append any missing from Base
    combined_order = list(new_order)
    for name in base_order:
        if name not in combined_order:
            combined_order.append(name)

    stage_rows = []
    for stage in combined_order:
        b_val = base_durations.get(stage, 0.0)
        n_val = new_durations.get(stage, 0.0)
        b_count = base_counts.get(stage, 1)
        n_count = new_counts.get(stage, 1)

        s_diff, s_pct = calculate_diff(b_val, n_val)

        # Format count string if aggregated
        count_str = ""
        if stage == "Denoising Loop":
            count_str = (
                f" ({n_count} steps)"
                if n_count == b_count
                else f" ({b_count}->{n_count} steps)"
            )

        # filter noise: show if diff is > 0.5ms OR if it's a major stage (like Denoising Loop)
        # always show Denoising Loop or stages with significant duration/diff
        stage_rows.append((stage + count_str, b_val, n_val, s_diff, s_pct))

    if output_format == "markdown":
        print("### Performance Comparison Report\n")

        # Summary Table
        print("#### 1. High-level Summary")
        print("| Metric | Baseline | New | Diff | Status |")
        print("| :--- | :--- | :--- | :--- | :--- |")
        print(
            f"| **E2E Latency** | {base_e2e:.2f} ms | {new_e2e:.2f} ms | **{diff_ms:+.2f} ms ({diff_pct:+.1f}%)** | {status} |"
        )
        print(
            f"| **Throughput** | {1000 / base_e2e if base_e2e else 0:.2f} req/s | {1000 / new_e2e if new_e2e else 0:.2f} req/s | - | - |"
        )
        print("\n")

        # Detailed Breakdown
        print("#### 2. Stage Breakdown")
        print(
            "| Stage Name | Baseline (ms) | New (ms) | Diff (ms) | Diff (%) | Status |"
        )
        print("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for name, b, n, d, p in stage_rows:
            name_str = name
            status_emoji = get_perf_status_emoji(b, n)
            print(
                f"| {name_str} | {b:.2f} | {n:.2f} | {d:+.2f} | {p:+.1f}% | {status_emoji} |"
            )
        print("\n")

        # Metadata
        print("<details>")
        print("<summary>Metadata</summary>\n")
        print(f"- Baseline Commit: `{base_data.get('commit_hash', 'N/A')}`")
        print(f"- New Commit: `{new_data.get('commit_hash', 'N/A')}`")
        print(f"- Timestamp: {datetime.now().isoformat()}")
        print("</details>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two sglang-diffusion performance JSON files."
    )
    parser.add_argument("baseline", help="Path to the baseline JSON file")
    parser.add_argument("new", help="Path to the new JSON file")
    args = parser.parse_args()

    compare_benchmarks(args.baseline, args.new)
