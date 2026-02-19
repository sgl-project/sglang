import gzip
import os
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger
from sglang.srt.environ import envs

if current_platform.is_npu():
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.schedule", torch_npu.profiler.schedule],
    ]
    torch_npu._apply_patches(patches)

logger = init_logger(__name__)


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
        activities: list[str] | None = None,
        num_steps: int | None = None,
        num_inference_steps: int | None = None,
        log_dir: str | None = None,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
        is_host: bool = False,
    ):
        self.request_id = request_id or "profile_trace"
        self.rank = rank
        self.full_profile = full_profile
        self.is_host = is_host

        self.log_dir = log_dir or envs.SGLANG_TORCH_PROFILER_DIR.get()

        env_with_stack = envs.SGLANG_PROFILE_WITH_STACK.get()
        env_record_shapes = envs.SGLANG_PROFILE_RECORD_SHAPES.get()

        self.with_stack = with_stack if with_stack is not None else env_with_stack
        self.record_shapes = (
            record_shapes if record_shapes is not None else env_record_shapes
        )

        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError:
            pass

        activities = [torch.profiler.ProfilerActivity.CPU]
        if current_platform.is_cuda_alike():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if current_platform.is_npu():
            activities.append(torch_npu.profiler.ProfilerActivity.NPU)

        common_torch_profiler_args = dict(
            activities=activities,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            on_trace_ready=(
                None
                if not current_platform.is_npu()
                else torch_npu.profiler.tensorboard_trace_handler(self.log_dir)
            ),
        )

        logger.info(
            f"Profiler config: output_dir={self.log_dir}, "
            f"with_stack={self.with_stack}, record_shapes={self.record_shapes}, "
            f"activities={[a.name for a in activities]}"
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

    @staticmethod
    def _resolve_activities(
        activities: list[str] | None,
    ) -> list[torch.profiler.ProfilerActivity]:
        """
        Resolve user-provided activity strings to torch profiler activities.
        """

        def _default() -> list[torch.profiler.ProfilerActivity]:
            ret = [torch.profiler.ProfilerActivity.CPU]
            if current_platform.is_cuda_alike():
                ret.append(torch.profiler.ProfilerActivity.CUDA)
            return ret

        if not activities:
            return _default()

        use_cpu = False
        use_cuda = False
        for a in activities:
            if a is None:
                continue
            s = str(a).strip().lower()
            if s == "cpu":
                use_cpu = True
            elif s in ("gpu", "cuda"):
                if current_platform.is_cuda_alike():
                    use_cuda = True
                else:
                    logger.warning(
                        "Profiler activities requested GPU/CUDA but CUDA is not available; ignoring."
                    )
            else:
                logger.warning(f"Unknown profiler activity: {a!r}; ignoring.")

        ret: list[torch.profiler.ProfilerActivity] = []
        if use_cpu:
            ret.append(torch.profiler.ProfilerActivity.CPU)
        if use_cuda:
            ret.append(torch.profiler.ProfilerActivity.CUDA)

        return ret or _default()

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
        if current_platform.is_cuda_alike():
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

            # Filename format:
            # - Host process: {profile_id}-host.trace.json.gz
            # - GPU Worker:   {profile_id}-rank-{rank}.trace.json.gz
            if self.is_host:
                filename = f"{self.request_id}-host.trace.json.gz"
            else:
                filename = f"{self.request_id}-rank-{self.rank}.trace.json.gz"

            trace_path = str(Path(self.log_dir, filename).resolve())
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
