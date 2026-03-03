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

import torch
from dateutil.tz import UTC

import sglang
import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    CYAN,
    RESET,
    _SGLDiffusionLogger,
    get_is_main_process,
    init_logger,
)

logger = init_logger(__name__)


@dataclasses.dataclass
class MemorySnapshot:
    allocated_mb: float  # current allocated memory
    reserved_mb: float  # current reserved memory (actual VRAM)
    peak_allocated_mb: float  # peak allocated since last reset
    peak_reserved_mb: float  # peak reserved since last reset

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocated_mb": round(self.allocated_mb, 2),
            "reserved_mb": round(self.reserved_mb, 2),
            "peak_allocated_mb": round(self.peak_allocated_mb, 2),
            "peak_reserved_mb": round(self.peak_reserved_mb, 2),
        }


@dataclasses.dataclass
class RequestMetrics:
    """Performance metrics for a single request, including timings and memory snapshots."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.stages: Dict[str, float] = {}
        self.steps: list[float] = []
        self.total_duration_ms: float = 0.0
        # memory tracking: {checkpoint_name: MemorySnapshot}
        self.memory_snapshots: Dict[str, MemorySnapshot] = {}

    @property
    def total_duration_s(self) -> float:
        return self.total_duration_ms / 1000.0

    def record_stage(self, stage_name: str, duration_s: float):
        """Records the duration of a pipeline stage"""
        self.stages[stage_name] = duration_s * 1000  # Store as milliseconds

    def record_steps(self, index: int, duration_s: float):
        """Records the duration of a denoising step"""
        assert index == len(self.steps)
        self.steps.append(duration_s * 1000)

    def record_memory_snapshot(self, checkpoint_name: str, snapshot: MemorySnapshot):
        self.memory_snapshots[checkpoint_name] = snapshot

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the metrics data to a dictionary."""
        return {
            "request_id": self.request_id,
            "stages": self.stages,
            "steps": self.steps,
            "total_duration_ms": self.total_duration_ms,
            "memory_snapshots": {
                name: snapshot.to_dict()
                for name, snapshot in self.memory_snapshots.items()
            },
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


def capture_memory_snapshot() -> MemorySnapshot:
    if not torch.get_device_module().is_available():
        return MemorySnapshot(
            allocated_mb=0.0,
            reserved_mb=0.0,
            peak_allocated_mb=0.0,
            peak_reserved_mb=0.0,
        )

    allocated = torch.get_device_module().memory_allocated()
    reserved = torch.get_device_module().memory_reserved()
    peak_allocated = torch.get_device_module().max_memory_allocated()
    peak_reserved = torch.get_device_module().max_memory_reserved()

    return MemorySnapshot(
        allocated_mb=allocated / (1024**2),
        reserved_mb=reserved / (1024**2),
        peak_allocated_mb=peak_allocated / (1024**2),
        peak_reserved_mb=peak_reserved / (1024**2),
    )


@dataclasses.dataclass
class RequestPerfRecord:
    request_id: str

    timestamp: str
    commit_hash: str
    tag: str

    stages: list[dict]
    steps: list[float]
    total_duration_ms: float
    memory_snapshots: dict[str, dict] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        request_id,
        commit_hash,
        tag,
        stages,
        steps,
        total_duration_ms,
        memory_snapshots=None,
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
        self.memory_snapshots = memory_snapshots or {}


class StageProfiler:
    """
    A unified context manager, records performance metrics (usually of a single Stage or a step) into a provided RequestMetrics object (usually from a Req).
    """

    def __init__(
        self,
        stage_name: str,
        logger: _SGLDiffusionLogger,
        metrics: Optional["RequestMetrics"],
        log_stage_start_end: bool = False,
        perf_dump_path_provided: bool = False,
        capture_memory: bool = False,
    ):
        self.stage_name = stage_name
        self.metrics = metrics
        self.logger = logger
        self.start_time = 0.0
        self.log_timing = perf_dump_path_provided or envs.SGLANG_DIFFUSION_STAGE_LOGGING
        self.log_stage_start_end = log_stage_start_end
        self.capture_memory = capture_memory

    def __enter__(self):
        if self.log_stage_start_end:
            msg = f"[{self.stage_name}] started..."
            if self.logger.isEnabledFor(logging.DEBUG):
                msg += f" ({round(current_platform.get_available_gpu_memory(), 2)} GB left)"
            self.logger.info(msg)

        if (self.log_timing and self.metrics) or self.log_stage_start_end:
            if (
                os.environ.get("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "0") == "1"
                and self.stage_name.startswith("denoising_step_")
                and torch.get_device_module().is_available()
            ):
                torch.get_device_module().synchronize()
            self.start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not ((self.log_timing and self.metrics) or self.log_stage_start_end):
            return False

        if (
            os.environ.get("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "0") == "1"
            and self.stage_name.startswith("denoising_step_")
            and torch.get_device_module().is_available()
        ):
            torch.get_device_module().synchronize()
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

        if self.log_stage_start_end:
            self.logger.info(
                f"[{self.stage_name}] finished in {execution_time_s:.4f} seconds",
            )

        if self.log_timing and self.metrics:
            if "denoising_step_" in self.stage_name:
                index = int(self.stage_name[len("denoising_step_") :])
                self.metrics.record_steps(index, execution_time_s)
            else:
                self.metrics.record_stage(self.stage_name, execution_time_s)

            # capture memory snapshot after stage if requested
            if self.capture_memory and torch.get_device_module().is_available():
                snapshot = capture_memory_snapshot()
                self.metrics.record_memory_snapshot(
                    f"after_{self.stage_name}", snapshot
                )

        return False


class PerformanceLogger:
    """
    A global utility class for logging performance metrics for all request, categorized by request-id.

    Serves both as a runtime logger (stream to file) and a dump utility.

    Notice that RequestMetrics stores the performance metrics of a single request
    """

    @classmethod
    def dump_benchmark_report(
        cls,
        file_path: str,
        metrics: "RequestMetrics",
        meta: Optional[Dict[str, Any]] = None,
        tag: str = "benchmark_dump",
    ):
        """
        Static method to dump a standardized benchmark report to a file.
        Eliminates duplicate logic in CLI/Client code.
        """
        formatted_steps = [
            {"name": name, "duration_ms": duration_ms}
            for name, duration_ms in metrics.stages.items()
        ]

        denoise_steps_ms = [
            {"step": idx, "duration_ms": duration_ms}
            for idx, duration_ms in enumerate(metrics.steps)
        ]

        memory_checkpoints = {
            name: snapshot.to_dict()
            for name, snapshot in metrics.memory_snapshots.items()
        }

        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": metrics.request_id,
            "commit_hash": get_git_commit_hash(),
            "tag": tag,
            "total_duration_ms": metrics.total_duration_ms,
            "steps": formatted_steps,
            "denoise_steps_ms": denoise_steps_ms,
            "memory_checkpoints": memory_checkpoints,
            "meta": meta or {},
        }

        try:
            abs_path = os.path.abspath(file_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Metrics dumped to: {CYAN}{abs_path}{RESET}")
        except IOError as e:
            logger.error(f"Failed to dump metrics to {abs_path}: {e}")

    @classmethod
    def log_request_summary(
        cls,
        metrics: "RequestMetrics",
        tag: str = "total_inference_time",
    ):
        """logs the stage metrics and total duration for a completed request
        to the performance_log file.

        Note that this accords to the time spent internally in server, postprocess is not included
        """
        formatted_stages = [
            {"name": name, "execution_time_ms": duration_ms}
            for name, duration_ms in metrics.stages.items()
        ]

        memory_checkpoints = {
            name: snapshot.to_dict()
            for name, snapshot in metrics.memory_snapshots.items()
        }

        record = RequestPerfRecord(
            metrics.request_id,
            commit_hash=get_git_commit_hash(),
            tag="pipeline_stage_metrics",
            stages=formatted_stages,
            steps=metrics.steps,
            total_duration_ms=metrics.total_duration_ms,
            memory_snapshots=memory_checkpoints,
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
