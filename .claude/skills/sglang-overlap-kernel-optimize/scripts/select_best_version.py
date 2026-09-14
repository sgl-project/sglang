#!/usr/bin/env python3
"""
Select the best-performing overlap kernel version from all benchmark results.

Reads all `<kernel_name>_result_v*.json` files from the results directory,
filters out versions that failed correctness, and selects the one with the
highest comm_savings_pct (how much of the pure communication time the overlap
kernel manages to hide behind compute). Also generates a summary JSON file.

Comm savings metric:
    comm_savings_pct = (non_overlap_us - overlap_us) / comm_only_us * 100%

The loop terminates when comm_savings_pct > target_pct (default 50%),
meaning the overlap kernel saves more than 50% of the pure communication time.

Usage:
    # Select best version and print progress table
    python3 select_best_version.py \
        --kernel_name allgather_gemm_overlap \
        --results_dir ./results \
        --comm_savings_target_pct 50

    # Also generate summary JSON file
    python3 select_best_version.py \
        --kernel_name allgather_gemm_overlap \
        --results_dir ./results \
        --comm_savings_target_pct 50 \
        --output_summary
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def compute_comm_savings_pct(perf: dict) -> float | None:
    """
    Compute how much of the pure communication time the overlap kernel saves.

    comm_savings_pct = (non_overlap_us - overlap_us) / comm_only_us * 100%

    Returns None if any required field is missing or comm_only_us is zero.
    """
    required = ["non_overlap_us", "overlap_us", "comm_only_us"]
    for field in required:
        if field not in perf or perf[field] is None:
            return None
    if perf["comm_only_us"] == 0:
        return None
    return (perf["non_overlap_us"] - perf["overlap_us"]) / perf["comm_only_us"] * 100.0


def load_result_files(kernel_name: str, results_dir: str) -> list[dict]:
    """Load all result JSON files for the given kernel name, sorted by version number."""
    results = []
    for filename in os.listdir(results_dir):
        if filename.startswith(kernel_name) and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath) as f:
                data = json.load(f)
            results.append(data)

    # Sort by version number (extract numeric part from v1, v2, etc.)
    def version_sort_key(result: dict) -> int:
        version_str = result.get("version", "v0")
        try:
            return int(version_str.replace("v", ""))
        except ValueError:
            return 0

    results.sort(key=version_sort_key)
    return results


def print_progress_table(results: list[dict], comm_target_pct: float) -> None:
    """Print the performance progress table in human-readable format."""
    print("\nIteration | Version | Correctness | Speedup | Overlap (us) | Comm Savings (%) | Key Change")
    print("-" * 105)

    best_savings_pct = -1.0
    best_version = None

    for result in results:
        version = result["version"]
        iteration = result.get("iteration", "?")
        correctness = result["correctness"]["status"]
        tuning = result.get("tuning_changes", "") or ""

        if correctness == "PASSED" and result["performance"] is not None:
            perf = result["performance"]
            speedup = perf["speedup"]
            overlap_us = perf["overlap_us"]
            savings_pct = compute_comm_savings_pct(perf)
            savings_display = f"{savings_pct:.1f}" if savings_pct is not None else "N/A"
            tuning_display = tuning[:40] + ("..." if len(tuning) > 40 else "")
            print(f"    {iteration}     |   {version}    |   {correctness}    |  {speedup:.2f}   |"
                  f"    {overlap_us:.1f}    |       {savings_display}       | {tuning_display}")
            if savings_pct is not None and savings_pct > best_savings_pct:
                best_savings_pct = savings_pct
                best_version = version
        else:
            errors = result.get("errors") or {}
            error_type = errors.get("error_type", "unknown") if errors else "unknown"
            tuning_display = tuning[:40] + ("..." if len(tuning) > 40 else "")
            print(f"    {iteration}     |   {version}    |   {correctness}    |   N/A   |"
                  f"     N/A      |        N/A       | {tuning_display} ({error_type})")

    print("-" * 105)
    if best_version:
        print(f"Best: {best_version} (comm_savings={best_savings_pct:.1f}%)")
        target_met = best_savings_pct >= comm_target_pct
        print(f"Comm savings target: >{comm_target_pct}% — {'ACHIEVED ✓' if target_met else 'NOT ACHIEVED'}")
    else:
        print("Best: NONE (all versions failed correctness)")


def select_best(results: list[dict]) -> dict | None:
    """
    Select the best version following the priority criteria:
    1. Correctness MUST pass
    2. Highest comm_savings_pct
    3. Tiebreaker: highest speedup
    4. Tiebreaker: lowest overlap latency
    """
    # Filter: only correctness-passing versions with performance data
    passing = [r for r in results if r["correctness"]["status"] == "PASSED" and r["performance"] is not None]

    if not passing:
        return None

    # Compute comm_savings_pct for each passing version
    for r in passing:
        r["_comm_savings_pct"] = compute_comm_savings_pct(r["performance"]) or -1.0

    # Sort by: comm_savings_pct DESC, speedup DESC, overlap_us ASC
    passing.sort(key=lambda r: (
        -r["_comm_savings_pct"],
        -r["performance"]["speedup"],
        r["performance"]["overlap_us"],
    ))

    # Clean up temporary field
    best = passing[0]
    best.pop("_comm_savings_pct", None)
    return best


def generate_summary(results: list[dict], best: dict | None,
                     comm_savings_target_pct: float, max_iterations: int,
                     kernel_name: str, results_dir: str) -> dict:
    """Generate the consolidated summary JSON."""
    versions_summary = []
    for result in results:
        entry = {
            "version": result["version"],
            "correctness": result["correctness"]["status"],
            "tuning_changes": result.get("tuning_changes", ""),
        }
        if result["correctness"]["status"] == "PASSED" and result["performance"] is not None:
            perf = result["performance"]
            entry["speedup"] = perf["speedup"]
            entry["overlap_us"] = perf["overlap_us"]
            entry["overlap_efficiency"] = perf["overlap_efficiency"]
            entry["comm_savings_pct"] = compute_comm_savings_pct(perf)
        else:
            entry["speedup"] = None
            entry["overlap_us"] = None
            entry["overlap_efficiency"] = None
            entry["comm_savings_pct"] = None
            errors = result.get("errors") or {}
            entry["error_type"] = errors.get("error_type", "unknown") if errors else "unknown"
        versions_summary.append(entry)

    # Determine convergence status
    passing = [r for r in results if r["correctness"]["status"] == "PASSED" and r["performance"] is not None]
    best_savings = compute_comm_savings_pct(best["performance"]) if best else None

    if not passing:
        convergence = "all_failed"
    elif best_savings is not None and best_savings >= comm_savings_target_pct:
        convergence = "converged"
    elif len(passing) >= 3:
        # Check no-progress: last 3 passing versions show < 5% improvement in comm_savings_pct
        recent_savings = [compute_comm_savings_pct(r["performance"]) or 0.0 for r in passing[-3:]]
        if max(recent_savings) - min(recent_savings) < 5.0:
            convergence = "no_progress"
        else:
            convergence = "max_iterations_reached"
    else:
        convergence = "max_iterations_reached"

    summary = {
        "kernel_name": kernel_name,
        "total_iterations": len(results),
        "comm_savings_target_pct": comm_savings_target_pct,
        "target_iterations": max_iterations,
        "versions": versions_summary,
        "best_version": best["version"] if best else None,
        "best_speedup": best["performance"]["speedup"] if best else None,
        "best_overlap_efficiency": best["performance"]["overlap_efficiency"] if best else None,
        "best_comm_savings_pct": best_savings,
        "convergence_status": convergence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Select the best-performing overlap kernel version from benchmark results."
    )
    parser.add_argument("--kernel_name", required=True, help="Kernel name to filter results")
    parser.add_argument("--results_dir", required=True, default="./results",
                        help="Directory containing result JSON files")
    parser.add_argument("--comm_savings_target_pct", type=float, default=50.0,
                        help="Target comm savings percentage threshold (default: 50%%). "
                             "Loop terminates when (non_overlap - overlap) / comm_only > this %%.")
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="Maximum optimization iterations planned")
    parser.add_argument("--output_summary", action="store_true",
                        help="Also generate <kernel_name>_perf_summary.json in results_dir")

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: results directory '{args.results_dir}' does not exist")
        sys.exit(1)

    results = load_result_files(args.kernel_name, args.results_dir)

    if not results:
        print(f"No result files found for kernel '{args.kernel_name}' in '{args.results_dir}'")
        sys.exit(1)

    # Print progress table
    print_progress_table(results, args.comm_savings_target_pct)

    # Select best version
    best = select_best(results)

    if best:
        best_savings = compute_comm_savings_pct(best["performance"])
        print(f"\nSelected best version: {best['version']}")
        print(f"  Speedup: {best['performance']['speedup']:.2f}x")
        print(f"  Overlap latency: {best['performance']['overlap_us']:.1f} us")
        print(f"  Overlap efficiency: {best['performance']['overlap_efficiency']:.2f}")
        if best_savings is not None:
            print(f"  Comm savings: {best_savings:.1f}% of comm_only time overlapped away")
            target_met = best_savings >= args.comm_savings_target_pct
            print(f"  Comm savings target (>{args.comm_savings_target_pct}%): {'ACHIEVED ✓' if target_met else 'NOT ACHIEVED'}")
    else:
        print("\nNo passing version found — all versions failed correctness.")
        print("Recommendation: manual debugging required; do NOT proceed to integration.")

    # Generate summary file if requested
    if args.output_summary:
        summary = generate_summary(results, best, args.comm_savings_target_pct,
                                   args.max_iterations, args.kernel_name, args.results_dir)
        summary_path = os.path.join(args.results_dir, f"{args.kernel_name}_perf_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
