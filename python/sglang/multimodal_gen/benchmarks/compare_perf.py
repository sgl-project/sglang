import argparse
import json
from datetime import datetime
from typing import Any, Dict

from sglang.multimodal_gen.benchmarks import perf_log_analyze


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

    # --- High-level Summary ---
    base_e2e = base_data.get("total_duration_ms", 0)
    new_e2e = new_data.get("total_duration_ms", 0)

    diff_ms, diff_pct = perf_log_analyze.calculate_diff(base_e2e, new_e2e)

    # Status icon: Improved (Green), Regression (Red), Neutral (Gray)
    # Assuming lower latency is better
    if diff_pct < -2.0:
        status = "✅ (Faster)"
    elif diff_pct > 2.0:
        status = "❌ (Slower)"
    else:
        status = "➖ (Similar)"

    # --- Stage Breakdown ---
    base_durations, base_order, base_counts = perf_log_analyze.consolidate_steps(
        base_data.get("steps", [])
    )
    new_durations, new_order, new_counts = perf_log_analyze.consolidate_steps(new_data.get("steps", []))

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

        s_diff, s_pct = perf_log_analyze.calculate_diff(b_val, n_val)

        # Format count string if aggregated
        count_str = ""
        if stage == "Denoising Loop":
            count_str = (
                f" ({n_count} steps)"
                if n_count == b_count
                else f" ({b_count}->{n_count} steps)"
            )

        # Filter noise: show if diff is > 0.5ms OR if it's a major stage (like Denoising Loop)
        # We always show Denoising Loop or stages with significant duration/diff
        if abs(s_diff) > 0.5 or b_val > 100 or n_val > 100:
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
        print("#### 2. Stage Breakdown (Execution Order)")
        if not stage_rows:
            print("*No significant stage differences found.*")
        else:
            print("| Stage Name | Baseline (ms) | New (ms) | Diff (ms) | Diff (%) | Status |")
            print("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for name, b, n, d, p in stage_rows:
                # Highlight large regressions (> 5%)
                name_str = f"**{name}**" if p > 5.0 else name
                status_emoji = perf_log_analyze.get_perf_status_emoji(b, n)
                print(f"| {name_str} | {b:.2f} | {n:.2f} | {d:+.2f} | {p:+.1f}% | {status_emoji} |")
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
        description="Compare two sglang performance JSON files."
    )
    parser.add_argument("baseline", help="Path to the baseline JSON file")
    parser.add_argument("new", help="Path to the new JSON file")
    args = parser.parse_args()

    compare_benchmarks(args.baseline, args.new)
