# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dateutil.tz import UTC

import sglang
import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    _SGLDiffusionLogger,
    get_is_main_process,
    init_logger,
)

logger = init_logger(__name__)


@dataclasses.dataclass
class RequestTimings:
    """A lightweight data class to store performance timings for a single request."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.stages: Dict[str, float] = {}
        self.steps: list[float] = []
        self.total_duration_ms: float = 0.0

    def record_stage(self, stage_name: str, duration_s: float):
        """Records the duration of a pipeline stage"""
        self.stages[stage_name] = duration_s * 1000  # Store as milliseconds

    def record_steps(self, index: int, duration_s: float):
        """Records the duration of a denoising step"""
        assert index == len(self.steps)
        self.steps.append(duration_s * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the timing data to a dictionary."""
        return {
            "request_id": self.request_id,
            "stages": self.stages,
            "steps": self.steps,
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


@dataclasses.dataclass
class RequestPerfRecord:
    request_id: str

    timestamp: str
    commit_hash: str
    tag: str

    stages: list[dict]
    steps: list[float]
    total_duration_ms: float

    def __init__(
        self,
        request_id,
        commit_hash,
        tag,
        stages,
        steps,
        total_duration_ms,
        timestamp=None,
    ):
        self.request_id = request_id
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now(UTC).isoformat()

        self.commit_hash = commit_hash
        self.tag = tag
        self.stages = stages
        self.steps = steps
        self.total_duration_ms = total_duration_ms


class StageProfiler:
    """
    A unified context manager, records timing information (usually of a single Stage or a step) into a provided RequestTimings object (usually from a Req).
    """

    def __init__(
        self,
        stage_name: str,
        logger: _SGLDiffusionLogger,
        timings: Optional["RequestTimings"],
        simple_log: bool = False,
    ):
        self.stage_name = stage_name
        self.timings = timings
        self.logger = logger
        self.simple_log = simple_log
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
                exc_info=True,
            )
            return False

        if self.simple_log:
            self.logger.info(
                f"[{self.stage_name}] finished in {execution_time_s:.4f} seconds"
            )

        if self.metrics_enabled and self.timings:
            if "denoising_step_" in self.stage_name:
                index = int(self.stage_name[len("denoising_step_") :])
                self.timings.record_steps(index, execution_time_s)
            else:
                self.timings.record_stage(self.stage_name, execution_time_s)

        return False


class PerformanceLogger:
    """
    A global utility class for logging performance metrics for all request, categorized by request-id.

    Serves both as a runtime logger (stream to file) and a dump utility.

    Notice that ""RequestTimings"" stores the performance metrics of a single request
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
            logging.getLogger(__name__).error(f"Dump failed: {e}")

    @classmethod
    def log_request_summary(
        cls,
        timings: "RequestTimings",
        tag: str = "total_inference_time",
    ):
        """logs the stage metrics and total duration for a completed request
        to the performance_log file.
        """
        formatted_stages = [
            {"name": name, "execution_time_ms": duration_ms}
            for name, duration_ms in timings.stages.items()
        ]

        record = RequestPerfRecord(
            timings.request_id,
            commit_hash=get_git_commit_hash(),
            tag="pipeline_stage_metrics",
            stages=formatted_stages,
            steps=timings.steps,
            total_duration_ms=timings.total_duration_ms,
        )

        try:
            if get_is_main_process():
                log_dir = get_diffusion_perf_log_dir()
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                log_file = os.path.join(log_dir, "performance.log")

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dataclasses.asdict(record)) + "\n")

        except (OSError, PermissionError) as e:
            print(f"WARNING: Failed to log performance record: {e}", file=sys.stderr)
