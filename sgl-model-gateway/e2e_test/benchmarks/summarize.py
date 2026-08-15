"""Generate benchmark summary for GitHub Actions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from results import BenchmarkResult, GPUUtilization


def discover_benchmarks(base_dir: Path) -> list[tuple[Path, str]]:
    """Auto-discover benchmark folders and their result JSON files.

    Returns list of (json_path, label) tuples sorted by folder name.
    """
    results = []
    for folder in base_dir.rglob("benchmark_*"):
        if not folder.is_dir():
            continue
        # Find result JSON (exclude metadata and gpu files)
        for json_file in folder.glob("*.json"):
            if (
                "experiment_metadata" not in json_file.name
                and "gpu_utilization" not in json_file.name
            ):
                # Generate label from folder name: benchmark_cache_aware_pd_grpc -> cache_aware pd grpc
                label = folder.name.replace("benchmark_", "").replace("_", " ")
                results.append((json_file, label))
                break  # One JSON per folder
    return sorted(results, key=lambda x: x[0].parent.name)


def find_gpu_utilization(result_path: Path) -> Path | None:
    """Find GPU utilization JSON in same folder as result."""
    gpu_json = result_path.parent / "gpu_utilization.json"
    return gpu_json if gpu_json.exists() else None


def generate_summary(base_dir: Path) -> str:
    """Generate markdown summary."""
    benchmarks = discover_benchmarks(base_dir)

    if not benchmarks:
        return (
            "## Gateway E2E Genai-Bench Results Summary\n\nNo benchmark results found."
        )

    lines = [
        "## Gateway E2E Genai-Bench Results Summary",
        "",
        "| Scenario | Status | TTFT (s) | E2E Latency (s) | Input Throughput (tok/s) | Output Throughput (tok/s) |",
        "|----------|--------|----------|-----------------|--------------------------|---------------------------|",
    ]

    gpu_sections = []

    for result_path, label in benchmarks:
        try:
            result = BenchmarkResult.from_json(result_path)
        except Exception as e:
            print(f"Warning: Failed to parse {result_path}: {e}", file=sys.stderr)
            lines.append(f"| {label} | ❌ Failed | - | - | - | - |")
            continue

        lines.append(
            f"| {label} | ✅ Success | "
            f"{result.ttft_mean:.2f} | "
            f"{result.e2e_latency_mean:.2f} | "
            f"{result.input_throughput_mean:.0f} | "
            f"{result.output_throughput_mean:.0f} |"
        )

        # GPU utilization
        gpu_path = find_gpu_utilization(result_path)
        if gpu_path:
            gpu = GPUUtilization.from_json(gpu_path)
            if gpu and gpu.per_gpu:
                gpu_lines = [
                    f"### GPU Utilization — {label}",
                    "",
                    f"Overall mean: {gpu.overall_mean:.2f}%",
                    "",
                    "| GPU | Mean (%) | p5 | p10 | p25 | p50 | p75 | p90 | p95 |",
                    "|-----|----------|----|-----|-----|-----|-----|-----|-----|",
                ]
                for gpu_id, stats in sorted(
                    gpu.per_gpu.items(), key=lambda x: int(x[0])
                ):
                    gpu_lines.append(
                        f"| {gpu_id} | {stats.get('mean', 0):.2f} | "
                        f"{stats.get('p5', 0):.2f} | {stats.get('p10', 0):.2f} | "
                        f"{stats.get('p25', 0):.2f} | {stats.get('p50', 0):.2f} | "
                        f"{stats.get('p75', 0):.2f} | {stats.get('p90', 0):.2f} | "
                        f"{stats.get('p95', 0):.2f} |"
                    )
                gpu_sections.append("\n".join(gpu_lines))

    return "\n".join(lines) + "\n" + "\n\n".join(gpu_sections)


def main() -> None:
    """Main entry point."""
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    summary = generate_summary(base_dir)

    # Write to GITHUB_STEP_SUMMARY if available
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(summary)
            f.write("\n")
        print(f"Summary written to {summary_file}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
