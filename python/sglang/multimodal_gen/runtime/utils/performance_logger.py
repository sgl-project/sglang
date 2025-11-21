# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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


class OnDemandFileHandler(logging.Handler):
    """
    A logging handler that opens the file for each log record, writes, and closes it.
    This is less performant than FileHandler but avoids long-lived file handles,
    which can be problematic on certain filesystems like NFS.
    """

    def __init__(self, filename: str, mode: str = "a", encoding: str | None = None):
        super().__init__()
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.terminator = "\n"

    def emit(self, record: logging.LogRecord):
        """Emit a record."""
        try:
            msg = self.format(record)
            with open(
                self.baseFilename, self.mode, encoding=self.encoding, errors="replace"
            ) as f:
                f.write(msg + self.terminator)
        except Exception:
            self.handleError(record)


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
        handler = OnDemandFileHandler(os.path.join(LOG_DIR, "performance.log"))
        handler.setFormatter(logging.Formatter("%(message)s"))
        perf_logger.addHandler(handler)
    except (OSError, PermissionError) as e:
        perf_logger.warning(f"Failed to initialize performance logger: {e}")
        # Disable file logging if initialization fails
        globals()["LOG_DIR"] = ""
    finally:
        _perf_logger_initialized = True


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"


class PerformanceLogger:
    """
    A utility class for logging performance metrics.
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.monotonic()
        self.step_timings = []
        self.commit_hash = get_git_commit_hash()

    def record_step_start(self):
        """Records the start time of a step."""
        self.step_start_time = time.monotonic()

    def record_step_end(self, step_name: str, step_index: int | None = None):
        """Records the end time of a step and calculates the duration."""
        duration = time.monotonic() - self.step_start_time
        self.step_timings.append(
            {"name": step_name, "index": step_index, "duration_ms": duration * 1000}
        )

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
