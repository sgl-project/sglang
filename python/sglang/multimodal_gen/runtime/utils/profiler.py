import contextlib
import gzip
import os
from collections.abc import Iterator

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_replica_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger
from sglang.srt.observability.profile_analysis_npu import (
    NPUTraceHandler,
    get_torch_npu,
)
from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches

if current_platform.is_npu():
    torch_npu = get_torch_npu()

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.schedule", torch_npu.profiler.schedule],
    ]
    apply_torch_npu_patches(torch_npu, patches)

logger = init_logger(__name__)


@contextlib.contextmanager
def maybe_record_function(name: str, enabled: bool = True) -> Iterator[None]:
    """Named ``torch.profiler`` span, near-free when no profiler is active.
    ``enabled`` mirrors :func:`maybe_nvtx_range` for per-request gates such
    as warmup exclusion.
    """
    # Gate on the process-wide module flag; the thread-local _profiler_enabled()
    # is False under profile_all_threads=True and on non-initiating threads.
    if not enabled or not torch.autograd.profiler._is_profiler_enabled:
        yield
        return
    with torch.profiler.record_function(name):
        yield


def _resolve_profiler_log_dir(log_dir: str | None) -> str:
    if log_dir is not None:
        return log_dir

    diffusion_profiler_dir = envs.SGLANG_DIFFUSION_TORCH_PROFILER_DIR
    if diffusion_profiler_dir:
        return diffusion_profiler_dir

    return os.getenv("SGLANG_TORCH_PROFILER_DIR", "./logs")


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

        self.log_dir = _resolve_profiler_log_dir(log_dir)

        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError:
            pass

        self._npu_trace_handler = (
            NPUTraceHandler(self.log_dir) if current_platform.is_npu() else None
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() or (
            hasattr(torch, "musa") and torch.musa.is_available()
        ):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if current_platform.is_npu():
            activities.append(torch_npu.profiler.ProfilerActivity.NPU)

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            activities.append(torch.profiler.ProfilerActivity.XPU)

        common_torch_profiler_args = dict(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            on_trace_ready=(
                self._npu_trace_handler.on_trace_ready
                if self._npu_trace_handler is not None
                else None
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
        sync_npu_stop_errors = (
            export_trace
            and self._npu_trace_handler is not None
            and self._npu_trace_handler.npu_auto_analysis_in_daemon
        )
        error = None
        try:
            if torch.cuda.is_available() or (
                hasattr(torch, "musa") and torch.musa.is_available()
            ):
                torch.cuda.synchronize()
            if current_platform.is_npu():
                torch.npu.synchronize()
            self.profiler.stop()
        except Exception as exc:
            if not sync_npu_stop_errors:
                raise
            error = f"rank {self.rank}: {exc}"
            logger.error("NPU profiler stop failed: %s", error)
        finally:
            SGLDiffusionProfiler._instance = None

        if sync_npu_stop_errors:
            errors = [error]
            if model_parallel_is_initialized():
                group = get_replica_group()
                if group.world_size > 1:
                    # Coordinate capture-stop failures only. Background analysis
                    # status is rank-local and never gates continued generation.
                    errors = [None] * group.world_size
                    torch.distributed.all_gather_object(
                        errors, error, group=group.cpu_group
                    )
            if any(errors):
                raise RuntimeError(
                    "NPU profiler stop failed: " + "; ".join(e for e in errors if e)
                )

        if export_trace and (
            self._npu_trace_handler is not None
            or dump_rank is None
            or dump_rank == self.rank
        ):
            self._export_trace()

    def _export_trace(self):
        if self._npu_trace_handler is not None:
            self._npu_trace_handler.export(self.profiler)
            return

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
