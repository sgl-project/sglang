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
    finally:
        _perf_logger_initialized = True


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
    A utility class for logging performance metrics.
    Serves both as a runtime logger (stream to file) and a dump utility.
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.monotonic()
        self.step_timings = []
        self.commit_hash = get_git_commit_hash()

    @classmethod
    def dump_benchmark_report(
        cls,
        file_path: str,
        request_id: str,
        total_duration_ms: float,
        logging_info: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        tag: str = "benchmark_dump",
    ):
        """
        Static method to dump a standardized benchmark report to a file.
        Eliminates duplicate logic in CLI/Client code.
        """
        # Convert logging_info stages to steps format
        formatted_steps = []
        if logging_info:
            stages = {}
            if hasattr(logging_info, "stages"):
                stages = logging_info.stages
            elif isinstance(logging_info, dict):
                stages = logging_info.get("stages", {})

            if isinstance(stages, dict):
                for name, info in stages.items():
                    if not info:
                        continue
                    exec_time = info.get("execution_time")
                    if exec_time is not None:
                        formatted_steps.append(
                            {"name": name, "duration_ms": exec_time * 1000}
                        )

        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "commit_hash": get_git_commit_hash(),
            "tag": tag,
            "total_duration_ms": total_duration_ms,
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
    def log_stage_metric(cls, request_id: str, stage_name: str, duration_ms: float):
        """Logs a single pipeline stage timing entry to the global log file."""
        _initialize_perf_logger()
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "commit_hash": get_git_commit_hash(),
            "tag": "pipeline_stage_metric",
            "stage": stage_name,
            "duration_ms": duration_ms,
        }
        perf_logger.info(json.dumps(log_entry))


class StageProfiler:
    """
    A unified context manager for profiling and logging pipeline stage execution.
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
        self.simple_log = simple_log
        self.logger = logger or logging.getLogger(__name__)

        # Check env var at runtime to ensure we pick up changes (e.g. from CLI args)
        self.metrics_enabled = envs.SGLANG_DIFFUSION_STAGE_LOGGING

    def __enter__(self):
        if self.simple_log:
            self.logger.info(f"[{self.stage_name}] started...")

        if self.metrics_enabled or self.simple_log:
            self.start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not (self.metrics_enabled or self.simple_log):
            return False

        execution_time = time.perf_counter() - self.start_time

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
            return False

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

        # 1. Update internal logging info (used for Dump)
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

        # 2. Update global perf logger (used for streaming logs)
        request_id = getattr(self.batch, "request_id", "unknown")
        try:
            PerformanceLogger.log_stage_metric(
                request_id=request_id,
                stage_name=self.stage_name,
                duration_ms=execution_time * 1000,
            )
        except Exception:
            self.logger.warning(
                "[%s] Failed to log stage metric to performance logger",
                self.stage_name,
                exc_info=True,
            )
