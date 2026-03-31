from __future__ import annotations

import gzip
import os

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger

try:
    from sglang.multimodal_gen.runtime.distributed import (
        get_world_group as _get_world_group,
    )
except ImportError:
    # distributed module not available in single-GPU setups
    _get_world_group = None


def is_primary_rank() -> bool:
    """Return True if this is rank 0 (or single-GPU).

    Only rank 0 should call cudaProfilerStart/Stop to avoid redundant
    profiling data in multi-GPU (sequence parallelism, tensor parallelism) setups.
    """
    if _get_world_group is not None:
        try:
            world_group = _get_world_group()
            if world_group is not None:
                return world_group.rank == 0
        except RuntimeError:
            # distributed not yet initialized at call time
            pass
    return True


if current_platform.is_npu():
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.schedule", torch_npu.profiler.schedule],
    ]
    torch_npu._apply_patches(patches)

logger = init_logger(__name__)


def parse_step_range(range_str: str | None) -> tuple[int, int] | None:
    """Parse step range string like '100-150' into (start, end) tuple."""
    if not range_str:
        return None
    try:
        parts = range_str.split("-")
        if len(parts) != 2:
            logger.warning(
                f"Invalid SGLANG_DIFFUSION_PROFILE_STEP_RANGE format: {range_str}. "
                "Expected 'START-END' (e.g., '100-150')"
            )
            return None
        start, end = int(parts[0].strip()), int(parts[1].strip())
        if start >= end:
            logger.warning(
                f"Invalid step range: start ({start}) must be less than end ({end})"
            )
            return None
        return (start, end)
    except ValueError as e:
        logger.warning(f"Failed to parse step range '{range_str}': {e}")
        return None


class DiffusionStepProfiler:
    """
    Global denoising step counter with cudaProfilerApi support.

    This class tracks the global denoising step count across all requests,
    similar to how LLM's scheduler tracks forward_ct. It enables nsys profiling
    with -c cudaProfilerApi by calling cudaProfilerStart/Stop at specific step ranges.

    Usage:
        Set SGLANG_DIFFUSION_PROFILE_STEP_RANGE="100-150" to profile steps 100-150.

        Request 1: steps 0-49   (global steps 0-49)
        Request 2: steps 0-49   (global steps 50-99)
        Request 3: steps 0-49   (global steps 100-149)  <- profiled if range is 100-150

    Run with:
        nsys profile -c cudaProfilerApi -t cuda,nvtx ... python -m sglang.cli.main serve ...

    Note: Currently tracks steps from the standard DenoisingStage only.
    Specialized stages (helios, causal, av, dmd, hunyuan3d) are not yet instrumented.
    """

    _instance: DiffusionStepProfiler | None = None

    def __init__(self):
        self.global_step_count = 0
        self.profile_range = parse_step_range(envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE)
        self.profiling_active = False
        self._profiler_started = False

        if self.profile_range:
            logger.info(
                f"DiffusionStepProfiler initialized with step range: "
                f"{self.profile_range[0]}-{self.profile_range[1]}"
            )

    @classmethod
    def get_instance(cls) -> "DiffusionStepProfiler":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = DiffusionStepProfiler()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def should_profile(self) -> bool:
        """Check if profiling is configured."""
        return self.profile_range is not None

    def step(self) -> bool:
        """
        Increment the global step counter and manage cudaProfiler state.

        Returns:
            True if this step is within the profile range, False otherwise.
        """
        current_step = self.global_step_count
        self.global_step_count += 1

        if not self.profile_range:
            return False

        start, end = self.profile_range
        in_range = start <= current_step < end

        if in_range and not self._profiler_started:
            self._start_cuda_profiler(current_step)
            self._profiler_started = True
            self.profiling_active = True
            if is_primary_rank():
                logger.info(
                    f"Profiling STARTED at global step {current_step}, "
                    f"will stop at step {end}"
                )

        if current_step >= end and self._profiler_started and self.profiling_active:
            if is_primary_rank():
                logger.info(
                    f"Profiling STOPPING at global step {current_step} "
                    f"(captured steps {start}-{current_step - 1})"
                )
            self.profiling_active = False
            self._stop_cuda_profiler(current_step)

        return in_range

    def _start_cuda_profiler(self, step: int):
        """Call cudaProfilerStart to begin capture.

        Only called on the primary rank (rank 0) to avoid redundant
        profiling data and duplicate logs in multi-GPU setups.
        """
        if not torch.cuda.is_available() or not is_primary_rank():
            return
        logger.info(
            f"cudaProfilerStart at global step {step} "
            f"(range: {self.profile_range[0]}-{self.profile_range[1]})"
        )
        torch.cuda.cudart().cudaProfilerStart()

    def _stop_cuda_profiler(self, step: int | None = None):
        """Call cudaProfilerStop to end capture.

        Only called on the primary rank (rank 0) to match cudaProfilerStart.
        """
        if not torch.cuda.is_available() or not is_primary_rank():
            return
        step_info = f" at global step {step}" if step is not None else ""
        logger.info(
            f"cudaProfilerStop{step_info} "
            f"(range: {self.profile_range[0]}-{self.profile_range[1]})"
        )
        torch.cuda.cudart().cudaProfilerStop()

    def get_global_step_count(self) -> int:
        """Get the current global step count."""
        return self.global_step_count

    def is_profiling_active(self) -> bool:
        """Check if profiling is currently active."""
        return self.profiling_active

    def ensure_stopped(self):
        """Ensure cudaProfilerStop is called if profiling was started.

        Call this in a finally block to guarantee cleanup on exceptions.

        Note: after this call, _profiler_started remains True so the range will
        not be re-captured in subsequent requests. This is intentional —
        cudaProfilerApi is designed for a single targeted capture window per
        server run.
        """
        if self._profiler_started and self.profiling_active:
            start, end = self.profile_range
            if is_primary_rank():
                logger.info(
                    f"Profiling early stop at global step {self.global_step_count} "
                    f"(captured steps {start}-{self.global_step_count - 1})"
                )
            self.profiling_active = False
            self._stop_cuda_profiler()

    def log_request_start(self, num_steps: int, request_id: str | None = None):
        """
        Log the start of a new request with its step range.

        Only logs on primary rank (rank 0) to avoid duplicate log messages
        in multi-GPU setups, matching the LLM runtime pattern.

        Args:
            num_steps: Number of denoising steps for this request.
            request_id: Optional request identifier.
        """
        if not self.profile_range:
            return
        if not is_primary_rank():
            return

        start_step = self.global_step_count
        end_step = start_step + num_steps
        req_info = f" (request: {request_id})" if request_id else ""

        profile_start, profile_end = self.profile_range
        will_be_profiled = start_step < profile_end and end_step > profile_start
        status = "WILL BE PROFILED" if will_be_profiled else "not in profile range"
        logger.info(
            f"Request starting{req_info}: global steps {start_step}-{end_step - 1} "
            f"[{status}] (target: {profile_start}-{profile_end})"
        )


class SGLDiffusionProfiler:
    """
    A wrapper around torch.profiler to simplify usage in pipelines.
    Supports both full profiling and scheduled profiling.


    1. if profile_all_stages is on: profile all stages, including all denoising steps
    2. otherwise, if num_profiled_timesteps is specified: profile {num_profiled_timesteps} denoising steps. profile all steps if num_profiled_timesteps==-1
    """

    _instance = None

    def __init__(
        self,
        request_id: str | None = None,
        rank: int = 0,
        full_profile: bool = False,
        num_steps: int | None = None,
        num_inference_steps: int | None = None,
        log_dir: str | None = None,
    ):
        self.request_id = request_id or "profile_trace"
        self.rank = rank
        self.full_profile = full_profile

        self.log_dir = (
            log_dir
            if log_dir is not None
            else os.getenv("SGLANG_TORCH_PROFILER_DIR", "./logs")
        )

        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError:
            pass

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() or (
            hasattr(torch, "musa") and torch.musa.is_available()
        ):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if current_platform.is_npu():
            activities.append(torch_npu.profiler.ProfilerActivity.NPU)

        common_torch_profiler_args = dict(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            on_trace_ready=(
                None
                if not current_platform.is_npu()
                else torch_npu.profiler.tensorboard_trace_handler(self.log_dir)
            ),
        )
        if self.full_profile:
            # profile all stages
            self.profiler = torch.profiler.profile(**common_torch_profiler_args)
            self.profile_mode_id = "full stages"
        else:
            # profile denoising stage only
            warmup = 1
            num_actual_steps = num_inference_steps if num_steps == -1 else num_steps
            self.num_active_steps = num_actual_steps + warmup
            self.profiler = torch.profiler.profile(
                **common_torch_profiler_args,
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=warmup,
                    active=self.num_active_steps,
                    repeat=1,
                ),
            )
            self.profile_mode_id = f"{num_actual_steps} steps"

        logger.info(f"Profiling request: {request_id} for {self.profile_mode_id}...")

        self.has_stopped = False

        SGLDiffusionProfiler._instance = self
        self.start()

    def start(self):
        logger.info("Starting Profiler...")
        self.profiler.start()

    def _step(self):
        self.profiler.step()

    def step_stage(self):
        if self.full_profile:
            self._step()

    def step_denoising_step(self):
        if not self.full_profile:
            if self.num_active_steps >= 0:
                self._step()
                self.num_active_steps -= 1
            else:
                # early exit when enough steps are captured, to reduce the trace file size
                self.stop(dump_rank=0)

    @classmethod
    def get_instance(cls) -> "SGLDiffusionProfiler":
        return cls._instance

    def stop(self, export_trace: bool = True, dump_rank: int | None = None):
        if self.has_stopped:
            return
        self.has_stopped = True
        logger.info("Stopping Profiler...")
        if torch.cuda.is_available() or (
            hasattr(torch, "musa") and torch.musa.is_available()
        ):
            torch.cuda.synchronize()
        if current_platform.is_npu():
            torch.npu.synchronize()
            export_trace = False  # set to false because our internal torch_npu.profiler will generate trace file
        self.profiler.stop()

        if export_trace:
            if dump_rank is not None and dump_rank != self.rank:
                pass
            else:
                self._export_trace()

        SGLDiffusionProfiler._instance = None

    def _export_trace(self):

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            sanitized_profile_mode_id = self.profile_mode_id.replace(" ", "_")
            trace_path = os.path.abspath(
                os.path.join(
                    self.log_dir,
                    f"{self.request_id}-{sanitized_profile_mode_id}-global-rank{self.rank}.trace.json.gz",
                )
            )
            self.profiler.export_chrome_trace(trace_path)

            if self._check_trace_integrity(trace_path):
                logger.info(f"Saved profiler traces to: {CYAN}{trace_path}{RESET}")
            else:
                logger.warning(f"Trace file may be corrupted: {trace_path}")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")

    def _check_trace_integrity(self, trace_path: str) -> bool:
        try:
            if not os.path.exists(trace_path) or os.path.getsize(trace_path) == 0:
                return False

            with gzip.open(trace_path, "rb") as f:
                content = f.read()
                if content.count(b"\x1f\x8b") > 1:
                    logger.warning("Multiple gzip headers detected")
                    return False

            return True
        except Exception as e:
            logger.warning(f"Trace file integrity check failed: {e}")
            return False
