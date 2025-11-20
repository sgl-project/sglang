# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dateutil.tz import UTC

import sglang


def get_diffusion_perf_log_dir() -> str:
    """
    Determines the directory for performance logs, centralizing the logic.

    Resolution order:
    1. SGLANG_PERF_LOG_DIR environment variable, if set and not empty.
    2. Default to ~/.cache/sglang/logs if the environment variable is not set.
    3. Returns an empty string if SGLANG_PERF_LOG_DIR is set to an empty string,
       which effectively disables file logging.
    """
    log_dir = os.environ.get("SGLANG_PERF_LOG_DIR")
    if log_dir:
        return os.path.abspath(log_dir)
    if log_dir is None:
        # Not set, use default
        sglang_path = Path(sglang.__file__).resolve()
        # .gitignore
        target_path = (sglang_path.parent / "../../.cache/logs").resolve()
        return str(target_path)
    # Is set, but is an empty string
    return ""


LOG_DIR = get_diffusion_perf_log_dir()

# Configure a specific logger for performance metrics
perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)
perf_logger.propagate = False  # Prevent perf logs from going to the main logger

_perf_logger_initialized = False


class FlushingFileHandler(logging.FileHandler):
    """
    A file handler that flushes after every emit to ensure logs are written immediately.
    This is more performant than opening/closing the file on every write (OnDemandFileHandler),
    but safer than standard buffering for real-time monitoring.
    """

    def emit(self, record):
        super().emit(record)
        self.flush()


def _initialize_perf_logger():
    """Initialize the performance logger with a file handler."""
    global _perf_logger_initialized
    if _perf_logger_initialized or not LOG_DIR:
        return

    try:
        # Ensure the logs directory exists
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # Set up a file handler for the performance logger
        # Use 'a' mode to append to existing logs
        handler = FlushingFileHandler(
            os.path.join(LOG_DIR, "performance.log"), encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        perf_logger.addHandler(handler)
    except (OSError, PermissionError) as e:
        # Use print because logging might not be set up or we want to avoid recursion
        print(f"WARNING: Failed to initialize performance logger: {e}", file=sys.stderr)
        # Disable file logging if initialization fails
        globals()["LOG_DIR"] = ""
    finally:
        _perf_logger_initialized = True


_CACHED_COMMIT_HASH = None


def get_git_commit_hash() -> str:
    """Get the current git commit hash with caching."""
    global _CACHED_COMMIT_HASH
    if _CACHED_COMMIT_HASH is not None:
        return _CACHED_COMMIT_HASH

    try:
        # Try to get it from environment variable first (useful in Docker/CI)
        commit_hash = os.environ.get("SGLANG_GIT_COMMIT")
        if not commit_hash:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .strip()
                .decode("utf-8")
            )
        _CACHED_COMMIT_HASH = commit_hash
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        _CACHED_COMMIT_HASH = "N/A"
        return "N/A"


class PerformanceLogger:
    """
    A utility class for logging performance metrics for Diffusion models.
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.monotonic()
        self.step_timings = []
        self.commit_hash = get_git_commit_hash()
        # Stack for nested measurements
        self._timer_stack = []

    def record_step_start(self):
        """Records the start time of a step (legacy API)."""
        self.step_start_time = time.monotonic()

    def record_step_end(self, step_name: str, step_index: int | None = None):
        """Records the end time of a step and calculates the duration (legacy API)."""
        if not hasattr(self, "step_start_time"):
            return
        duration = time.monotonic() - self.step_start_time
        self.step_timings.append(
            {"name": step_name, "index": step_index, "duration_ms": duration * 1000}
        )

    @contextmanager
    def measure(self, step_name: str, step_index: int | None = None):
        """
        Context manager for measuring execution time of a block.
        Supports nesting.
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.step_timings.append(
                {
                    "name": step_name,
                    "index": step_index,
                    "duration_ms": duration * 1000,
                }
            )

    def dump_to_file(self, file_path: str, tag: str = "manual_dump"):
        """
        Dumps the current metrics to a specific file.
        Useful for generating isolated benchmark reports.
        """
        total_duration = time.monotonic() - self.start_time
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": self.request_id,
            "commit_hash": self.commit_hash,
            "tag": tag,
            "total_duration_ms": total_duration * 1000,
            "steps": self.step_timings,
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2)
            print(f"[Performance] Metrics dumped to: {file_path}")
        except IOError as e:
            print(f"[Performance] Failed to dump metrics to {file_path}: {e}")

    def log_total_duration(self, tag: str):
        """Logs the total duration of the operation and all recorded steps."""
        _initialize_perf_logger()
        total_duration = time.monotonic() - self.start_time
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": self.request_id,
            "commit_hash": self.commit_hash,
            "tag": tag,
            "total_duration_ms": total_duration * 1000,
            "steps": self.step_timings,
        }
        perf_logger.info(json.dumps(log_entry))

    def log_stage_metric(self, stage_name: str, duration_ms: float):
        """Logs a single pipeline stage timing entry."""
        _initialize_perf_logger()
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": self.request_id,
            "commit_hash": self.commit_hash,
            "tag": "pipeline_stage_metric",
            "stage": stage_name,
            "duration_ms": duration_ms,
        }
        perf_logger.info(json.dumps(log_entry))

    def log_stage_metrics(self, stages: Any):
        """
        Persist per-stage execution stats to performance.log.

        Args:
            stages: Either a PipelineLoggingInfo instance or any object exposing
                a mapping of stage metadata via a `stages` attribute/dict.
        """
        _initialize_perf_logger()
        if stages is None:
            return

        if hasattr(stages, "stages"):
            stage_items = getattr(stages, "stages", {}).items()
        elif isinstance(stages, dict):
            stage_items = stages.items()
        else:
            return

        formatted_stages: list[dict[str, Any]] = []
        for name, info in stage_items:
            if not info:
                continue
            entry = {"name": name}
            execution_time = info.get("execution_time")
            if execution_time is not None:
                entry["execution_time_ms"] = execution_time * 1000
            for key, value in info.items():
                if key == "execution_time":
                    continue
                entry[key] = value
            formatted_stages.append(entry)

        if not formatted_stages:
            return

        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": self.request_id,
            "commit_hash": self.commit_hash,
            "tag": "pipeline_stage_metrics",
            "stages": formatted_stages,
        }
        perf_logger.info(json.dumps(log_entry))


# -------------------------------------------------------------------------
#  Benchmark Comparison Utilities
# -------------------------------------------------------------------------


def _load_benchmark_file(file_path: str) -> Dict[str, Any]:
    """Loads a benchmark JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _calculate_diff(base: float, new: float) -> tuple[float, float]:
    """Returns (diff, diff_percent)."""
    diff = new - base
    if base == 0:
        percent = 0.0
    else:
        percent = (diff / base) * 100
    return diff, percent


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

    diff_ms, diff_pct = _calculate_diff(base_e2e, new_e2e)

    # Status icon: Improved (Green), Regression (Red), Neutral (Gray)
    # Assuming lower latency is better
    if diff_pct < -2.0:
        status = "✅ (Faster)"
    elif diff_pct > 2.0:
        status = "❌ (Slower)"
    else:
        status = "➖ (Similar)"

    # Determine significant stage changes
    # We assume 'steps' is a list of dicts: [{'name': ..., 'duration_ms': ...}, ...]
    # We flatten them by name. If multiple steps have same name, we sum them?
    # Or usually steps are unique phases like "TE", "UNet", "VAE".
    # Let's aggregate by name just in case.

    def aggregate_steps(steps: List[Dict]) -> Dict[str, float]:
        agg = {}
        for s in steps:
            name = s.get("name", "unknown")
            dur = s.get("duration_ms", 0)
            agg[name] = agg.get(name, 0) + dur
        return agg

    base_steps = aggregate_steps(base_data.get("steps", []))
    new_steps = aggregate_steps(new_data.get("steps", []))

    # Identify significant changes (e.g. > 1% diff contribution or > 5% absolute change)
    stage_rows = []
    all_stage_names = set(base_steps.keys()) | set(new_steps.keys())

    for stage in sorted(all_stage_names):
        b_val = base_steps.get(stage, 0)
        n_val = new_steps.get(stage, 0)
        s_diff, s_pct = _calculate_diff(b_val, n_val)

        # Filter noise: show if diff is > 0.5ms
        if abs(s_diff) > 0.5:
            stage_rows.append((stage, b_val, n_val, s_diff, s_pct))

    # Sort by absolute diff magnitude (descending)
    stage_rows.sort(key=lambda x: abs(x[3]), reverse=True)

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
            f"| **Throughput** | {1000/base_e2e if base_e2e else 0:.2f} req/s | {1000/new_e2e if new_e2e else 0:.2f} req/s | - | - |"
        )
        print("\n")

        # Detailed Breakdown
        print("#### 2. Stage Breakdown (Top Changes)")
        if not stage_rows:
            print("*No significant stage differences found.*")
        else:
            print("| Stage Name | Baseline (ms) | New (ms) | Diff (ms) | Diff (%) |")
            print("| :--- | :--- | :--- | :--- | :--- |")
            for name, b, n, d, p in stage_rows:
                # Highlight large regressions
                name_str = f"**{name}**" if p > 5.0 else name
                print(f"| {name_str} | {b:.2f} | {n:.2f} | {d:+.2f} | {p:+.1f}% |")
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
