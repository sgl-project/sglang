# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dateutil.tz import UTC

import sglang
import sglang.multimodal_gen.envs as envs

perf_logger: logging.Logger | None = None


class RequestTimings:
    """A lightweight data class to store performance timings for a single request."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.stages: Dict[str, float] = {}
        self.total_duration_ms: float = 0.0

    def record_stage(self, stage_name: str, duration_s: float):
        """Records the duration of a pipeline stage."""
        self.stages[stage_name] = duration_s * 1000  # Store as milliseconds

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the timing data to a dictionary."""
        return {
            "request_id": self.request_id,
            "stages": self.stages,
            "total_duration_ms": self.total_duration_ms,
        }


def get_diffusion_perf_log_dir() -> str:
    """
    Determines the directory for performance logs.
    """
    log_dir = os.environ.get("SGLANG_PERF_LOG_DIR")
    if log_dir:
        return os.path.abspath(log_dir)
    if log_dir is None:
        sglang_path = Path(sglang.__file__).resolve()
        target_path = (sglang_path.parent / "../../.cache/logs").resolve()
        return str(target_path)
    return ""


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def _initialize_perf_logger():
    global perf_logger
    if perf_logger is not None:
        return
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False

    try:
        LOG_DIR = get_diffusion_perf_log_dir()
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        handler = FlushingFileHandler(
            os.path.join(LOG_DIR, "performance.log"), encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        perf_logger.addHandler(handler)
    except (OSError, PermissionError) as e:
        print(f"WARNING: Failed to initialize performance logger: {e}", file=sys.stderr)
        globals()["LOG_DIR"] = ""


@lru_cache(maxsize=1)
def get_git_commit_hash() -> str:
    try:
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
    A utility class for logging performance metrics for all request, categorized by request-id.
    Serves both as a runtime logger (stream to file) and a dump utility.
    """

    @classmethod
    def dump_benchmark_report(
        cls,
        file_path: str,
        timings: "RequestTimings",
        meta: Optional[Dict[str, Any]] = None,
        tag: str = "benchmark_dump",
    ):
        """
        Static method to dump a standardized benchmark report to a file.
        Eliminates duplicate logic in CLI/Client code.
        """
        formatted_steps = [
            {"name": name, "duration_ms": duration_ms}
            for name, duration_ms in timings.stages.items()
        ]

        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": timings.request_id,
            "commit_hash": get_git_commit_hash(),
            "tag": tag,
            "total_duration_ms": timings.total_duration_ms,
            "steps": formatted_steps,
            "meta": meta or {},
        }

        try:
            abs_path = os.path.abspath(file_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Performance] Metrics dumped to: {abs_path}")
        except IOError as e:
            print(f"[Performance] Failed to dump metrics to {abs_path}: {e}")
            # Log error but don't crash the program
            logging.getLogger(__name__).error(f"Dump failed: {e}")

    @classmethod
    def log_request_summary(
        cls,
        timings: "RequestTimings",
        tag: str = "total_inference_time",
    ):
        """Logs the stage metrics and total duration for a completed request."""
        _initialize_perf_logger()

        # 1. Log all stage metrics from timings
        formatted_stages = [
            {"name": name, "execution_time_ms": duration_ms}
            for name, duration_ms in timings.stages.items()
        ]

        if formatted_stages:
            stages_log_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": timings.request_id,
                "commit_hash": get_git_commit_hash(),
                "tag": "pipeline_stage_metrics",
                "stages": formatted_stages,
            }
            if perf_logger is not None:
                perf_logger.info(json.dumps(stages_log_entry))

        # 2. Log total duration
        total_duration_log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": timings.request_id,
            "commit_hash": get_git_commit_hash(),
            "tag": tag,
            "total_duration_ms": timings.total_duration_ms,
        }
        if perf_logger is not None:
            perf_logger.info(json.dumps(total_duration_log_entry))


class StageProfiler:
    """
    A unified context manager for profiling and logging pipeline stage execution.
    It records timing information (usually of a single Stage or a step) into a provided RequestTimings object.
    """

    def __init__(
        self,
        stage_name: str,
        timings: Optional["RequestTimings"],
        simple_log: bool = False,
    ):
        self.stage_name = stage_name
        self.timings = timings
        self.simple_log = simple_log
        self.logger = logging.getLogger(__name__)
        self.start_time = 0.0

        # Check env var at runtime to ensure we pick up changes (e.g. from CLI args)
        self.metrics_enabled = envs.SGLANG_DIFFUSION_STAGE_LOGGING

    def __enter__(self):
        if self.simple_log:
            self.logger.info(f"[{self.stage_name}] started...")

        if (self.metrics_enabled and self.timings) or self.simple_log:
            self.start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not ((self.metrics_enabled and self.timings) or self.simple_log):
            return False

        execution_time_s = time.perf_counter() - self.start_time

        if exc_type:
            self.logger.error(
                "[%s] Error during execution after %.4f ms: %s",
                self.stage_name,
                execution_time_s * 1000,
                exc_val,
            )
            if self.metrics_enabled:
                self.logger.error(
                    "[%s] Traceback: %s",
                    self.stage_name,
                    "".join(traceback.format_tb(exc_tb)),
                )
            return False

        if self.simple_log:
            self.logger.info(
                f"[{self.stage_name}] finished in {execution_time_s:.4f} seconds"
            )

        if self.metrics_enabled and self.timings:
            self.timings.record_stage(self.stage_name, execution_time_s)

        return False
