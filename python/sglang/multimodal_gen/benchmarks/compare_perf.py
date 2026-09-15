import argparse
import json
import os
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


def _get_status_emoji_from_diff_percent(diff_pct):
    if diff_pct < -2.0:
        return "✅"
    elif diff_pct > 2.0:
        return "❌"
    else:
        return "⚪️"


def _print_single_comparison_report(
    others_data, base_e2e, combined_order, base_durations, others_processed, base_counts
):
    new_data = others_data[0]
    new_e2e = new_data.get("total_duration_ms", 0)
    diff_ms, diff_pct = calculate_diff(base_e2e, new_e2e)
    status = _get_status_emoji_from_diff_percent(diff_pct)

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

    print("#### 2. Stage Breakdown")
    print("| Stage Name | Baseline (ms) | New (ms) | Diff (ms) | Diff (%) | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    new_durations, _, new_counts = others_processed[0]

    for stage in combined_order:
        b_val = base_durations.get(stage, 0.0)
        n_val = new_durations.get(stage, 0.0)
        b_count = base_counts.get(stage, 1)
        n_count = new_counts.get(stage, 1)

        s_diff, s_pct = calculate_diff(b_val, n_val)

        count_str = ""
        if stage == "Denoising Loop":
            count_str = (
                f" ({n_count} steps)"
                if n_count == b_count
                else f" ({b_count}->{n_count} steps)"
            )

        status_emoji = get_perf_status_emoji(b_val, n_val)
        print(
            f"| {stage}{count_str} | {b_val:.2f} | {n_val:.2f} | {s_diff:+.2f} | {s_pct:+.1f}% | {status_emoji} |"
        )


def _print_multi_comparison_report(
    base_e2e,
    others_data,
    other_labels,
    combined_order,
    base_durations,
    others_processed,
):
    print("#### 1. High-level Summary")
    header = "| Metric | Baseline | " + " | ".join(other_labels) + " |"
    sep = "| :--- | :--- | " + " | ".join([":---"] * len(other_labels)) + " |"
    print(header)
    print(sep)

    # E2E Row
    row_e2e = f"| **E2E Latency** | {base_e2e:.2f} ms |"
    for i, d in enumerate(others_data):
        val = d.get("total_duration_ms", 0)
        diff_ms, diff_pct = calculate_diff(base_e2e, val)

        status = _get_status_emoji_from_diff_percent(diff_pct)

        row_e2e += f" {val:.2f} ms ({diff_pct:+.1f}%) {status} |"
    print(row_e2e)
    print("\n")

    print("#### 2. Stage Breakdown")
    # Header: Stage | Baseline | Label1 | Label2 ...
    header = "| Stage Name | Baseline | " + " | ".join(other_labels) + " |"
    sep = "| :--- | :--- | " + " | ".join([":---"] * len(other_labels)) + " |"
    print(header)
    print(sep)

    for stage in combined_order:
        b_val = base_durations.get(stage, 0.0)
        row_str = f"| {stage} | {b_val:.2f} |"

        for i, (n_durations, _, n_counts) in enumerate(others_processed):
            n_val = n_durations.get(stage, 0.0)
            _, s_pct = calculate_diff(b_val, n_val)
            status_emoji = get_perf_status_emoji(b_val, n_val)

            row_str += f" {n_val:.2f} ({s_pct:+.1f}%) {status_emoji} |"
        print(row_str)


def compare_benchmarks(file_paths: List[str], output_format: str = "markdown"):
    """
    Compares benchmark JSON files and prints a report.
    First file is baseline, others will be compared against it.
    """
    if len(file_paths) < 2:
        print("Error: Need at least 2 files to compare.")
        return

    try:
        data_list = [_load_benchmark_file(f) for f in file_paths]
    except Exception as e:
        print(f"Error loading benchmark files: {e}")
        return

    base_data = data_list[0]
    others_data = data_list[1:]

    # Use filenames as labels if multiple comparisons, else just "New"
    other_labels = [os.path.basename(p) for p in file_paths[1:]]

    base_e2e = base_data.get("total_duration_ms", 0)

    base_durations, base_order, base_counts = consolidate_steps(
        base_data.get("steps", [])
    )

    others_processed = []
    for d in others_data:
        dur, order, counts = consolidate_steps(d.get("steps", []))
        others_processed.append((dur, order, counts))

    combined_order = []
    # Collect all unique stages maintaining order from newest to baseline
    for _, order, _ in reversed(others_processed):
        for name in order:
            if name not in combined_order:
                combined_order.append(name)
    for name in base_order:
        if name not in combined_order:
            combined_order.append(name)

    if output_format == "markdown":
        print("### Performance Comparison Report\n")

        if len(others_data) == 1:
            _print_single_comparison_report(
                others_data,
                base_e2e,
                combined_order,
                base_durations,
                others_processed,
                base_counts,
            )
        else:
            _print_multi_comparison_report(
                base_e2e,
                others_data,
                other_labels,
                combined_order,
                base_durations,
                others_processed,
            )

        print("\n")
        # Metadata
        print("<details>")
        print("<summary>Metadata</summary>\n")
        print(f"- Baseline Commit: `{base_data.get('commit_hash', 'N/A')}`")
        for i, d in enumerate(others_data):
            label = "New" if len(others_data) == 1 else other_labels[i]
            print(f"- {label} Commit: `{d.get('commit_hash', 'N/A')}`")
        print(f"- Timestamp: {datetime.now().isoformat()}")
        print("</details>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare sglang-diffusion performance JSON files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="List of JSON files. First is baseline, others are compared against it.",
    )
    args = parser.parse_args()

    compare_benchmarks(args.files)
