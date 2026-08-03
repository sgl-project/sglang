from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from sglang.srt.managers.io_struct import ProfileReqOutput
from sglang.srt.utils.tensor_bridge import use_mlx

logger = logging.getLogger(__name__)


@dataclass
class MetalCaptureProfiler:
    label: str
    trace_path: Path
    stop_capture: Callable[[], None]
    standalone: bool

    @classmethod
    def start_mlx(cls, trace_path: Path):
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import mlx.core as mx

            mx.metal.start_capture(str(trace_path))
        except RuntimeError as e:
            return None, _capture_error("MLX", e)

        return cls._started(
            label="MLX",
            trace_path=trace_path,
            stop_capture=mx.metal.stop_capture,
            standalone=True,
        )

    @classmethod
    def start_mps(cls, trace_path: Path):
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not hasattr(torch, "mps") or not hasattr(torch.mps, "profiler"):
                raise RuntimeError("torch.mps.profiler is not available")
            context = torch.mps.profiler.metal_capture(str(trace_path))
            context.__enter__()
        except RuntimeError as e:
            return None, _capture_error("MPS", e)

        return cls._started(
            label="MPS",
            trace_path=trace_path,
            stop_capture=lambda: context.__exit__(None, None, None),
            standalone=False,
        )

    @classmethod
    def _started(
        cls,
        *,
        label: str,
        trace_path: Path,
        stop_capture: Callable[[], None],
        standalone: bool,
    ):
        profiler = cls(
            label=label,
            trace_path=trace_path,
            stop_capture=stop_capture,
            standalone=standalone,
        )
        logger.info("%s Metal capture started, saving to %s", label, trace_path)
        return profiler, ProfileReqOutput(success=True, message="Succeeded")

    def stop(self) -> str:
        self.stop_capture()

        logger.info(
            "%s Metal capture stopped. Trace saved to: %s",
            self.label,
            self.trace_path,
        )
        return f" Metal trace: {self.trace_path}"


def _capture_error(label: str, error: RuntimeError) -> ProfileReqOutput:
    return ProfileReqOutput(
        success=False,
        message=(
            f"Failed to start {label} Metal capture: {error}. "
            "Set MTL_CAPTURE_ENABLED=1 in the server's environment "
            "before launching to enable GPU trace capture."
        ),
    )


class MetalTorchProfiler:
    def __init__(
        self,
        *,
        start_metal_capture: Callable[[Path], tuple[Any, ProfileReqOutput]],
        torch_profiler: Optional[Any] = None,
    ):
        self.start_metal_capture = start_metal_capture
        self.torch_profiler = torch_profiler
        self.metal_profiler = None

    def start(self):
        trace_path = _new_temp_gputrace_path()
        self.metal_profiler, result = self.start_metal_capture(trace_path)
        if not result.success:
            raise RuntimeError(result.message)
        if self.torch_profiler is not None:
            try:
                self.torch_profiler.start()
            except Exception:
                self.metal_profiler.stop()
                raise

    def stop(self):
        try:
            if self.torch_profiler is not None:
                self.torch_profiler.stop()
        finally:
            if self.metal_profiler is not None:
                self.metal_profiler.stop()

    def export_chrome_trace(self, path: str):
        if self.torch_profiler is not None:
            self.torch_profiler.export_chrome_trace(path)
        else:
            _write_empty_chrome_trace(path)

        if self.metal_profiler is None:
            return

        final_path = _unique_gputrace_path_for_chrome_trace(path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.metal_profiler.trace_path.exists():
            shutil.move(str(self.metal_profiler.trace_path), str(final_path))
            logger.info("Metal trace saved to: %s", final_path)


def apply_metal_profiler_patches() -> None:
    if getattr(torch.profiler.profile, "_sglang_metal_patched", False):
        return

    original_profile = torch.profiler.profile

    def profile(*args, **kwargs):
        activities = _get_activities(args, kwargs)
        if not _has_cuda_activity(activities):
            return original_profile(*args, **kwargs)

        if use_mlx():
            return MetalTorchProfiler(
                start_metal_capture=MetalCaptureProfiler.start_mlx
            )

        torch_activities = [
            activity for activity in activities if not _is_cuda_activity(activity)
        ]
        torch_profiler = None
        if torch_activities:
            patched_args, patched_kwargs = _replace_activities(
                args, kwargs, torch_activities
            )
            torch_profiler = original_profile(*patched_args, **patched_kwargs)

        return MetalTorchProfiler(
            start_metal_capture=MetalCaptureProfiler.start_mps,
            torch_profiler=torch_profiler,
        )

    profile._sglang_metal_patched = True
    profile._sglang_original_profile = original_profile
    torch.profiler.profile = profile


def _get_activities(args, kwargs):
    if "activities" in kwargs:
        return kwargs["activities"]
    if args:
        return args[0]
    return None


def _replace_activities(args, kwargs, activities):
    kwargs = dict(kwargs)
    if "activities" in kwargs:
        kwargs["activities"] = activities
        return args, kwargs

    if args:
        args = list(args)
        args[0] = activities
        return tuple(args), kwargs

    kwargs["activities"] = activities
    return args, kwargs


def _has_cuda_activity(activities) -> bool:
    if activities is None:
        return False
    return any(_is_cuda_activity(activity) for activity in activities)


def _is_cuda_activity(activity) -> bool:
    return activity == torch.profiler.ProfilerActivity.CUDA


def _new_temp_gputrace_path() -> Path:
    output_dir = Path(os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        candidate = (
            output_dir / f"sglang-metal-{os.getpid()}-{time.time_ns()}-{i}.gputrace"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find an unused Metal trace path in {output_dir}")


def _unique_gputrace_path_for_chrome_trace(path: str) -> Path:
    chrome_path = Path(path).expanduser()
    name = chrome_path.name
    if name.endswith(".trace.json.gz"):
        name = name[: -len(".trace.json.gz")] + ".gputrace"
    else:
        name = chrome_path.stem + ".gputrace"

    base = chrome_path.with_name(name)
    if not base.exists():
        return base

    stem = base.name[: -len(".gputrace")]
    for i in range(100):
        candidate = base.with_name(f"{stem}-{time.time_ns()}-{i}.gputrace")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find an unused Metal trace path for {base}")


def _write_empty_chrome_trace(path: str):
    trace = {"traceEvents": []}
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as f:
            json.dump(trace, f)
    else:
        with open(path, "w") as f:
            json.dump(trace, f)
