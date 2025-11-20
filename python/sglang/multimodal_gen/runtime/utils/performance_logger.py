# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import json
import logging
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dateutil.tz import UTC

import sglang
import sglang.multimodal_gen.envs as envs


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


class StageProfiler:
    """
    A unified context manager for profiling and logging pipeline stage execution.
    Handles timing, updating logging_info, and writing to perf_logger.
    """

    def __init__(
        self,
        stage_name: str,
        batch: Any = None,
        logger: logging.Logger = None,
        simple_log: bool = False,
    ):
        self.stage_name = stage_name
        self.batch = batch
        # If simple_log is True, we log start/end unconditionally (like Timer)
        self.simple_log = simple_log
        self.logger = logger or logging.getLogger(__name__)

        # Check if metric recording is enabled
        self.metrics_enabled = envs.SGLANG_DIFFUSION_STAGE_LOGGING

    def __enter__(self):
        if self.simple_log:
            self.logger.info(f"[{self.stage_name}] started...")

        if self.metrics_enabled or self.simple_log:
            self.start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If not tracking time, just return (propagate exception if any)
        if not (self.metrics_enabled or self.simple_log):
            return False

        execution_time = time.perf_counter() - self.start_time

        # Handle Exceptions
        if exc_type:
            self.logger.error(
                "[%s] Error during execution after %.4f ms: %s",
                self.stage_name,
                execution_time * 1000,
                exc_val,
            )
            if self.metrics_enabled:
                self.logger.error(
                    "[%s] Traceback: %s",
                    self.stage_name,
                    "".join(traceback.format_tb(exc_tb)),
                )
            return False  # Propagate exception

        # Success Case
        if self.simple_log:
            self.logger.info(
                f"[{self.stage_name}] finished in {execution_time:.4f} seconds"
            )

        if self.metrics_enabled:
            self._record_metrics(execution_time)

        return False

    def _record_metrics(self, execution_time: float):
        if self.batch is None:
            return

        logging_info = getattr(self.batch, "logging_info", None)
        if logging_info is not None:
            try:
                logging_info.add_stage_execution_time(self.stage_name, execution_time)
            except Exception:
                self.logger.warning(
                    "[%s] Failed to record stage timing on batch.logging_info",
                    self.stage_name,
                    exc_info=True,
                )

        perf_logger = getattr(self.batch, "perf_logger", None)
        if perf_logger is not None:
            try:
                perf_logger.log_stage_metric(self.stage_name, execution_time * 1000)
            except Exception:
                self.logger.warning(
                    "[%s] Failed to log stage metric to performance logger",
                    self.stage_name,
                    exc_info=True,
                )
